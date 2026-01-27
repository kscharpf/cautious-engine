"""
Long-term satellite coverage calculation.

This module calculates satellite visibility over extended time periods (90 days)
using the adaptive propagation algorithm.
"""

import math
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict, Optional, NamedTuple
from dataclasses import dataclass

from satellite import (
    Satrec, parse_tle, GroundPosition, LookAngle,
    calculate_visibility_at_time
)
from propagation import (
    PropagationConfig, adaptive_propagate, find_visibility_windows,
    VisibilityWindow, estimate_satellite_orbital_period
)


@dataclass
class CoverageResult:
    """Coverage statistics for a single ground position."""
    latitude: float
    longitude: float
    total_visible_seconds: float
    total_time_seconds: float
    coverage_percent: float
    num_passes: int
    passes: List[VisibilityWindow]


@dataclass
class TimeHorizonConfig:
    """Configuration for long-term coverage calculation."""
    duration_days: float = 90.0           # Total simulation duration
    min_elevation: float = 10.0           # Minimum elevation angle (degrees)
    progress_interval_hours: float = 24.0 # How often to report progress


def calculate_point_coverage(
    satellite: Satrec,
    ground: GroundPosition,
    start_time: datetime,
    config: TimeHorizonConfig
) -> CoverageResult:
    """
    Calculate coverage statistics for a single ground point over the time horizon.

    Returns coverage percentage and detailed pass information.
    """
    end_time = start_time + timedelta(days=config.duration_days)
    total_seconds = config.duration_days * 24 * 3600

    # Use adaptive propagation to find visibility windows
    prop_config = PropagationConfig(min_elevation=config.min_elevation)

    windows = find_visibility_windows(
        satellite, ground, start_time, end_time, prop_config
    )

    # Calculate total visible time
    total_visible_seconds = sum(
        (w.end - w.start).total_seconds() for w in windows
    )

    coverage_percent = (total_visible_seconds / total_seconds) * 100

    return CoverageResult(
        latitude=ground.latitude,
        longitude=ground.longitude,
        total_visible_seconds=total_visible_seconds,
        total_time_seconds=total_seconds,
        coverage_percent=coverage_percent,
        num_passes=len(windows),
        passes=windows
    )


def calculate_coverage_with_progress(
    satellite: Satrec,
    ground: GroundPosition,
    start_time: datetime,
    config: TimeHorizonConfig,
    progress_callback=None
) -> CoverageResult:
    """
    Calculate coverage with progress reporting for long simulations.

    Args:
        satellite: SGP4 satellite object
        ground: Ground position
        start_time: Start of simulation
        config: Time horizon configuration
        progress_callback: Optional callback function(percent_complete, elapsed_sim_time)
    """
    end_time = start_time + timedelta(days=config.duration_days)
    total_seconds = config.duration_days * 24 * 3600

    prop_config = PropagationConfig(min_elevation=config.min_elevation)

    all_windows = []
    current_start = start_time
    chunk_duration = timedelta(hours=config.progress_interval_hours)

    chunk_num = 0
    total_chunks = int(math.ceil(config.duration_days * 24 / config.progress_interval_hours))

    while current_start < end_time:
        current_end = min(current_start + chunk_duration, end_time)

        # Find visibility windows in this chunk
        windows = find_visibility_windows(
            satellite, ground, current_start, current_end, prop_config
        )
        all_windows.extend(windows)

        chunk_num += 1
        if progress_callback:
            elapsed_hours = (current_end - start_time).total_seconds() / 3600
            percent = (chunk_num / total_chunks) * 100
            progress_callback(percent, elapsed_hours)

        current_start = current_end

    # Calculate totals
    total_visible_seconds = sum(
        (w.end - w.start).total_seconds() for w in all_windows
    )

    coverage_percent = (total_visible_seconds / total_seconds) * 100

    return CoverageResult(
        latitude=ground.latitude,
        longitude=ground.longitude,
        total_visible_seconds=total_visible_seconds,
        total_time_seconds=total_seconds,
        coverage_percent=coverage_percent,
        num_passes=len(all_windows),
        passes=all_windows
    )


def demo_90_day_coverage():
    """Demonstrate 90-day coverage calculation for a single point."""
    print("=" * 70)
    print("90-DAY SATELLITE COVERAGE CALCULATION")
    print("=" * 70)

    # ISS TLE
    tle_line1 = "1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990"
    tle_line2 = "2 25544  51.6443 242.2052 0004885 264.6463 206.3557 15.49165514212791"

    satellite = parse_tle(tle_line1, tle_line2)

    # Ground position: London, UK
    ground = GroundPosition(latitude=51.5074, longitude=-0.1278, altitude=0.0)

    # Start time at TLE epoch
    start_time = datetime(2020, 2, 14, 4, 27, 33, tzinfo=timezone.utc)

    config = TimeHorizonConfig(
        duration_days=90.0,
        min_elevation=10.0,
        progress_interval_hours=168.0  # Weekly progress updates
    )

    print(f"\nSatellite: ISS (ZARYA)")
    print(f"Ground Station: London, UK ({ground.latitude}, {ground.longitude})")
    print(f"Minimum Elevation: {config.min_elevation}°")
    print(f"Duration: {config.duration_days} days")

    import time as time_module
    print("\nCalculating coverage...")

    def progress(percent, elapsed_hours):
        days = elapsed_hours / 24
        print(f"  Progress: {percent:.0f}% (simulated {days:.0f} days)")

    t_start = time_module.time()
    result = calculate_coverage_with_progress(
        satellite, ground, start_time, config,
        progress_callback=progress
    )
    elapsed = time_module.time() - t_start

    print(f"\n{'='*50}")
    print("RESULTS")
    print('='*50)
    print(f"Total simulation time: {elapsed:.2f} seconds")
    print(f"\nCoverage Statistics:")
    print(f"  Total passes: {result.num_passes}")
    print(f"  Total visible time: {result.total_visible_seconds/3600:.1f} hours")
    print(f"  Coverage: {result.coverage_percent:.2f}%")

    # Show sample of passes
    if result.passes:
        print(f"\nSample passes (first 5):")
        for i, window in enumerate(result.passes[:5], 1):
            duration = (window.end - window.start).total_seconds()
            print(f"  {i}. {window.start.strftime('%Y-%m-%d %H:%M')} - "
                  f"duration: {duration:.0f}s, max el: {window.max_elevation:.1f}°")

        if len(result.passes) > 5:
            print(f"  ... and {len(result.passes) - 5} more passes")

    return result


if __name__ == "__main__":
    demo_90_day_coverage()
