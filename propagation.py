"""
Intelligent satellite propagation algorithm with adaptive time stepping.

This module implements an adaptive algorithm that skips time intervals
when satellites are definitely not visible, improving computation efficiency.
"""

import math
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional, NamedTuple, Generator
from dataclasses import dataclass

from satellite import (
    Satrec, parse_tle, propagate, datetime_to_gmst, teme_to_ecef,
    calculate_look_angle, is_visible, GroundPosition, LookAngle, Position,
    EARTH_RADIUS_KM, DEG_TO_RAD, RAD_TO_DEG
)
import numpy as np


class VisibilityWindow(NamedTuple):
    """A time window when satellite is visible."""
    start: datetime
    end: datetime
    max_elevation: float  # Peak elevation during pass


@dataclass
class PropagationConfig:
    """Configuration for adaptive propagation."""
    min_elevation: float = 10.0      # Minimum elevation for visibility (degrees)
    coarse_step_seconds: float = 60.0   # Large step when far from visibility
    medium_step_seconds: float = 10.0   # Medium step when approaching visibility
    fine_step_seconds: float = 1.0      # Fine step when near visibility threshold

    # Thresholds for switching step sizes
    far_elevation_threshold: float = -30.0   # Below this: use coarse steps
    near_elevation_threshold: float = 0.0    # Above this: use fine steps


def estimate_satellite_orbital_period(satellite: Satrec) -> float:
    """
    Estimate the orbital period of a satellite in seconds.

    Uses the mean motion from the TLE (revolutions per day).
    """
    # Mean motion is in revolutions per day
    mean_motion_rev_per_day = satellite.no_kozai * 60 * 24 / (2 * math.pi)  # Convert rad/min to rev/day

    if mean_motion_rev_per_day <= 0:
        # Fallback to typical LEO period (~95 minutes)
        return 95 * 60

    period_days = 1.0 / mean_motion_rev_per_day
    period_seconds = period_days * 24 * 3600

    return period_seconds


def calculate_satellite_geometry(
    satellite: Satrec,
    ground: GroundPosition,
    dt: datetime
) -> LookAngle:
    """
    Calculate satellite look angle from ground position.

    Returns:
        LookAngle (elevation, azimuth, range)
    """
    pos, _ = propagate(satellite, dt)
    gmst = datetime_to_gmst(dt)
    sat_ecef = teme_to_ecef(pos, gmst)
    return calculate_look_angle(sat_ecef, ground)


def determine_step_size(
    current_elevation: float,
    config: PropagationConfig
) -> float:
    """
    Determine the optimal time step based on current satellite elevation.

    Strategy:
    - Far below horizon (< -30°): coarse steps (60s) - don't waste time
    - Approaching horizon (-30° to 0°): medium steps (10s) - don't miss the pass
    - Near or above horizon (> 0°): fine steps (1s) - accurate timing

    Note: Elevation rate was previously calculated but analysis showed
    that rate-based prediction doesn't improve performance enough to
    justify the 2x propagation cost. Simple threshold-based stepping
    is more efficient for LEO satellites.
    """
    if current_elevation < config.far_elevation_threshold:
        return config.coarse_step_seconds

    elif current_elevation < config.near_elevation_threshold:
        return config.medium_step_seconds

    else:
        return config.fine_step_seconds


def adaptive_propagate(
    satellite: Satrec,
    ground: GroundPosition,
    start_time: datetime,
    end_time: datetime,
    config: Optional[PropagationConfig] = None
) -> Generator[Tuple[datetime, LookAngle, bool], None, None]:
    """
    Adaptively propagate satellite and yield visibility calculations.

    Uses intelligent time stepping to skip periods when satellite is
    definitely not visible.

    Yields:
        Tuple of (datetime, look_angle, is_visible)
    """
    if config is None:
        config = PropagationConfig()

    current_time = start_time

    while current_time < end_time:
        # Calculate current geometry
        look_angle = calculate_satellite_geometry(satellite, ground, current_time)

        visible = is_visible(look_angle, config.min_elevation)
        yield current_time, look_angle, visible

        # Determine next step size based on elevation
        step = determine_step_size(look_angle.elevation, config)

        current_time = current_time + timedelta(seconds=step)


def find_visibility_windows(
    satellite: Satrec,
    ground: GroundPosition,
    start_time: datetime,
    end_time: datetime,
    config: Optional[PropagationConfig] = None
) -> List[VisibilityWindow]:
    """
    Find all visibility windows for a satellite from a ground position.

    Returns a list of time windows when the satellite is visible.
    """
    if config is None:
        config = PropagationConfig()

    windows = []
    current_window_start = None
    max_elevation = -90.0

    for dt, look_angle, visible in adaptive_propagate(
        satellite, ground, start_time, end_time, config
    ):
        if visible:
            if current_window_start is None:
                current_window_start = dt
            max_elevation = max(max_elevation, look_angle.elevation)
        else:
            if current_window_start is not None:
                # Window just ended
                windows.append(VisibilityWindow(
                    start=current_window_start,
                    end=dt,
                    max_elevation=max_elevation
                ))
                current_window_start = None
                max_elevation = -90.0

    # Handle case where visibility extends past end_time
    if current_window_start is not None:
        windows.append(VisibilityWindow(
            start=current_window_start,
            end=end_time,
            max_elevation=max_elevation
        ))

    return windows


def count_propagation_steps(
    satellite: Satrec,
    ground: GroundPosition,
    start_time: datetime,
    end_time: datetime,
    use_adaptive: bool = True
) -> Tuple[int, float]:
    """
    Count the number of propagation steps needed and measure efficiency.

    Returns:
        Tuple of (step_count, elapsed_time_seconds)
    """
    import time as time_module

    if use_adaptive:
        config = PropagationConfig()
        start = time_module.time()
        count = sum(1 for _ in adaptive_propagate(
            satellite, ground, start_time, end_time, config
        ))
        elapsed = time_module.time() - start
    else:
        # Fixed 1-second steps for comparison
        start = time_module.time()
        count = 0
        current = start_time
        while current < end_time:
            pos, _ = propagate(satellite, current)
            gmst = datetime_to_gmst(current)
            sat_ecef = teme_to_ecef(pos, gmst)
            _ = calculate_look_angle(sat_ecef, ground)
            count += 1
            current += timedelta(seconds=1)
        elapsed = time_module.time() - start

    return count, elapsed


def demo_adaptive_propagation():
    """Demonstrate the adaptive propagation algorithm."""
    print("=" * 70)
    print("ADAPTIVE PROPAGATION ALGORITHM DEMONSTRATION")
    print("=" * 70)

    # ISS TLE
    tle_line1 = "1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990"
    tle_line2 = "2 25544  51.6443 242.2052 0004885 264.6463 206.3557 15.49165514212791"

    satellite = parse_tle(tle_line1, tle_line2)

    # Ground position: London, UK
    ground = GroundPosition(latitude=51.5074, longitude=-0.1278, altitude=0.0)

    # Time range: 12 hours starting from TLE epoch
    start_time = datetime(2020, 2, 14, 4, 27, 33, tzinfo=timezone.utc)
    end_time = start_time + timedelta(hours=12)

    print(f"\nSatellite: ISS (ZARYA)")
    print(f"Ground Station: London, UK ({ground.latitude}, {ground.longitude})")
    print(f"Time Range: {start_time} to {end_time}")
    print(f"Duration: 12 hours")

    # Estimate orbital period
    period = estimate_satellite_orbital_period(satellite)
    print(f"\nEstimated orbital period: {period/60:.1f} minutes")

    # Find visibility windows using adaptive algorithm
    print("\n--- Finding Visibility Windows (Adaptive Algorithm) ---")
    config = PropagationConfig(min_elevation=10.0)

    import time as time_module
    t_start = time_module.time()
    windows = find_visibility_windows(satellite, ground, start_time, end_time, config)
    t_elapsed = time_module.time() - t_start
    print(f"(Completed in {t_elapsed:.2f}s)")

    print(f"\nFound {len(windows)} visibility windows:")
    for i, window in enumerate(windows, 1):
        duration = (window.end - window.start).total_seconds()
        print(f"  {i}. {window.start.strftime('%H:%M:%S')} - {window.end.strftime('%H:%M:%S')} "
              f"(duration: {duration:.0f}s, max el: {window.max_elevation:.1f}°)")

    # Compare efficiency: adaptive vs fixed 1-second steps (10 minute sample)
    print("\n--- Efficiency Comparison (10 minute sample) ---")
    sample_end = start_time + timedelta(minutes=10)

    adaptive_steps, adaptive_time = count_propagation_steps(
        satellite, ground, start_time, sample_end, use_adaptive=True
    )

    # Calculate theoretical fixed steps (1 per second for 10 minutes)
    fixed_steps = 10 * 60  # 600 steps for 10 minutes at 1-second intervals

    print(f"\nFixed 1-second stepping (theoretical):")
    print(f"  Steps: {fixed_steps}")

    print(f"\nAdaptive stepping:")
    print(f"  Steps: {adaptive_steps}")
    print(f"  Time: {adaptive_time:.3f}s")

    reduction = (1 - adaptive_steps / fixed_steps) * 100

    print(f"\nImprovement:")
    print(f"  Step reduction: {reduction:.1f}%")

    return len(windows) > 0


if __name__ == "__main__":
    demo_adaptive_propagation()
