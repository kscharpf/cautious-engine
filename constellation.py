"""
Multi-satellite constellation coverage calculation.

This module extends the single-satellite algorithms to handle
constellations of satellites where visibility from any one
satellite is sufficient.
"""

import math
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict, Optional, NamedTuple, Generator
from dataclasses import dataclass, field
import numpy as np

from satellite import (
    Satrec, parse_tle, GroundPosition, LookAngle, Position,
    propagate, datetime_to_gmst, teme_to_ecef, calculate_look_angle, is_visible,
    geodetic_to_ecef_grid, calculate_visibility_grid_multi_sat
)
from propagation import (
    PropagationConfig, adaptive_propagate, VisibilityWindow,
    estimate_satellite_orbital_period
)


# Sample Starlink TLEs (subset for demonstration)
# These are representative TLEs - in production, use current TLEs from Celestrak
SAMPLE_STARLINK_TLES = [
    ("STARLINK-1007", "1 44713U 19074A   20045.54621927  .00001046  00000-0  76155-4 0  9994",
                      "2 44713  53.0544 269.6478 0001422  83.6137 276.5038 15.06390482 10725"),
    ("STARLINK-1008", "1 44714U 19074B   20045.54621942  .00000983  00000-0  72115-4 0  9995",
                      "2 44714  53.0544 269.6477 0001375  87.3221 272.7955 15.06390358 10724"),
    ("STARLINK-1009", "1 44715U 19074C   20045.20932893  .00000964  00000-0  71138-4 0  9993",
                      "2 44715  53.0549 271.2909 0001301  79.6259 280.4907 15.06385987 10671"),
    ("STARLINK-1010", "1 44716U 19074D   20045.21599629  .00000917  00000-0  68113-4 0  9995",
                      "2 44716  53.0550 271.2575 0001461  77.7393 282.3778 15.06390024 10678"),
    ("STARLINK-1011", "1 44717U 19074E   20045.88266098  .00000952  00000-0  70175-4 0  9998",
                      "2 44717  53.0552 268.0459 0001404  81.7449 278.3720 15.06389419 10781"),
    ("STARLINK-1012", "1 44718U 19074F   20045.54621966  .00001009  00000-0  73699-4 0  9996",
                      "2 44718  53.0548 269.6474 0001292  90.8498 269.2672 15.06383918 10725"),
    ("STARLINK-1013", "1 44719U 19074G   20045.88266101  .00001038  00000-0  75832-4 0  9993",
                      "2 44719  53.0549 268.0457 0001472  77.8571 282.2599 15.06389527 10785"),
    ("STARLINK-1014", "1 44720U 19074H   20045.88266111  .00000999  00000-0  73264-4 0  9997",
                      "2 44720  53.0548 268.0454 0001395  85.2116 274.9055 15.06390135 10781"),
]


@dataclass
class ConstellationConfig:
    """Configuration for constellation coverage calculation."""
    min_elevation: float = 10.0  # Minimum elevation for visibility (degrees)
    time_step_seconds: float = 60.0  # Time step for coverage calculation
    # Adaptive stepping is used internally


@dataclass
class ConstellationCoverage:
    """Coverage results for a constellation at multiple ground points."""
    coverage_map: np.ndarray  # 2D array of coverage percentages
    latitudes: np.ndarray
    longitudes: np.ndarray
    total_time_seconds: float
    num_satellites: int
    average_coverage: float


class Constellation:
    """Represents a satellite constellation."""

    def __init__(self, name: str = "Unnamed"):
        self.name = name
        self.satellites: List[Tuple[str, Satrec]] = []

    def add_satellite(self, name: str, tle_line1: str, tle_line2: str):
        """Add a satellite to the constellation."""
        sat = parse_tle(tle_line1, tle_line2)
        self.satellites.append((name, sat))

    def add_from_tle_list(self, tle_list: List[Tuple[str, str, str]]):
        """Add satellites from a list of (name, line1, line2) tuples."""
        for name, line1, line2 in tle_list:
            self.add_satellite(name, line1, line2)

    @property
    def size(self) -> int:
        return len(self.satellites)

    def check_any_visible(
        self,
        ground: GroundPosition,
        dt: datetime,
        min_elevation: float = 10.0
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if any satellite in the constellation is visible.

        Returns (is_visible, visible_satellite_name or None)
        """
        gmst = datetime_to_gmst(dt)

        for name, sat in self.satellites:
            try:
                pos, _ = propagate(sat, dt)
                sat_ecef = teme_to_ecef(pos, gmst)
                look = calculate_look_angle(sat_ecef, ground)
                if is_visible(look, min_elevation):
                    return True, name
            except ValueError:
                # Propagation error - skip this satellite
                continue

        return False, None

    def count_visible(
        self,
        ground: GroundPosition,
        dt: datetime,
        min_elevation: float = 10.0
    ) -> int:
        """Count how many satellites are visible from a ground position."""
        gmst = datetime_to_gmst(dt)
        count = 0

        for name, sat in self.satellites:
            try:
                pos, _ = propagate(sat, dt)
                sat_ecef = teme_to_ecef(pos, gmst)
                look = calculate_look_angle(sat_ecef, ground)
                if is_visible(look, min_elevation):
                    count += 1
            except ValueError:
                continue

        return count


def calculate_constellation_coverage_at_point(
    constellation: Constellation,
    ground: GroundPosition,
    start_time: datetime,
    duration_days: float,
    config: ConstellationConfig
) -> float:
    """
    Calculate coverage percentage for a constellation at a single point.

    Returns the percentage of time when at least one satellite is visible.
    """
    end_time = start_time + timedelta(days=duration_days)
    total_seconds = duration_days * 24 * 3600

    visible_seconds = 0.0
    current_time = start_time

    while current_time < end_time:
        visible, _ = constellation.check_any_visible(
            ground, current_time, config.min_elevation
        )
        if visible:
            visible_seconds += config.time_step_seconds

        current_time += timedelta(seconds=config.time_step_seconds)

    return (visible_seconds / total_seconds) * 100


def calculate_constellation_coverage_map(
    constellation: Constellation,
    start_time: datetime,
    duration_hours: float,
    resolution_deg: float = 5.0,
    config: Optional[ConstellationConfig] = None,
    progress_callback=None
) -> ConstellationCoverage:
    """
    Calculate coverage map for a constellation over a time period.

    Returns a ConstellationCoverage object with spatial coverage data.
    Uses vectorized computation for significant performance improvement.
    """
    if config is None:
        config = ConstellationConfig()

    # Create lat/lon grid
    lats = np.arange(-90, 90, resolution_deg)
    lons = np.arange(-180, 180, resolution_deg)

    visible_counts = np.zeros((len(lats), len(lons)), dtype=np.int32)

    end_time = start_time + timedelta(hours=duration_hours)
    total_seconds = duration_hours * 3600
    total_steps = int(total_seconds / config.time_step_seconds)

    # Precompute ground ECEF positions once (vectorized)
    ground_x, ground_y, ground_z = geodetic_to_ecef_grid(lats, lons, alt=0.0)

    current_time = start_time
    step = 0

    while current_time < end_time:
        gmst = datetime_to_gmst(current_time)

        # Precompute all satellite positions at this time
        sat_positions = []
        for name, sat in constellation.satellites:
            try:
                pos, _ = propagate(sat, current_time)
                sat_ecef = teme_to_ecef(pos, gmst)
                sat_positions.append(sat_ecef)
            except ValueError:
                continue

        # Vectorized visibility check for all grid points
        visibility = calculate_visibility_grid_multi_sat(
            sat_positions, ground_x, ground_y, ground_z,
            lats, lons, config.min_elevation
        )
        visible_counts += visibility.astype(np.int32)

        step += 1
        if progress_callback and step % 10 == 0:
            progress_callback(step, total_steps, current_time)

        current_time += timedelta(seconds=config.time_step_seconds)

    # Convert counts to percentages
    coverage_map = (visible_counts / max(1, step)) * 100

    return ConstellationCoverage(
        coverage_map=coverage_map.astype(np.float32),
        latitudes=lats,
        longitudes=lons,
        total_time_seconds=total_seconds,
        num_satellites=constellation.size,
        average_coverage=float(np.mean(coverage_map))
    )


def calculate_constellation_coverage_map_scalar(
    constellation: Constellation,
    start_time: datetime,
    duration_hours: float,
    resolution_deg: float = 5.0,
    config: Optional[ConstellationConfig] = None,
    progress_callback=None
) -> ConstellationCoverage:
    """
    Calculate coverage map using scalar (non-vectorized) computation.

    Provided for comparison and validation against vectorized version.
    """
    if config is None:
        config = ConstellationConfig()

    # Create lat/lon grid
    lats = np.arange(-90, 90, resolution_deg)
    lons = np.arange(-180, 180, resolution_deg)

    visible_counts = np.zeros((len(lats), len(lons)), dtype=np.int32)

    end_time = start_time + timedelta(hours=duration_hours)
    total_seconds = duration_hours * 3600
    total_steps = int(total_seconds / config.time_step_seconds)

    current_time = start_time
    step = 0

    while current_time < end_time:
        gmst = datetime_to_gmst(current_time)

        # Precompute all satellite positions at this time
        sat_positions = []
        for name, sat in constellation.satellites:
            try:
                pos, _ = propagate(sat, current_time)
                sat_ecef = teme_to_ecef(pos, gmst)
                sat_positions.append(sat_ecef)
            except ValueError:
                continue

        # Check visibility at each grid point (scalar loop)
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                ground = GroundPosition(lat, lon, 0.0)

                # Check if any satellite is visible
                for sat_ecef in sat_positions:
                    look = calculate_look_angle(sat_ecef, ground)
                    if is_visible(look, config.min_elevation):
                        visible_counts[i, j] += 1
                        break  # At least one visible is enough

        step += 1
        if progress_callback and step % 10 == 0:
            progress_callback(step, total_steps, current_time)

        current_time += timedelta(seconds=config.time_step_seconds)

    # Convert counts to percentages
    coverage_map = (visible_counts / max(1, step)) * 100

    return ConstellationCoverage(
        coverage_map=coverage_map.astype(np.float32),
        latitudes=lats,
        longitudes=lons,
        total_time_seconds=total_seconds,
        num_satellites=constellation.size,
        average_coverage=float(np.mean(coverage_map))
    )


def demo_constellation_coverage():
    """Demonstrate multi-satellite constellation coverage calculation."""
    print("=" * 70)
    print("MULTI-SATELLITE CONSTELLATION COVERAGE DEMONSTRATION")
    print("=" * 70)

    # Create constellation with sample Starlink satellites
    constellation = Constellation("Starlink (Sample)")
    constellation.add_from_tle_list(SAMPLE_STARLINK_TLES)

    print(f"\nConstellation: {constellation.name}")
    print(f"Number of satellites: {constellation.size}")

    # Also add ISS for comparison
    constellation_with_iss = Constellation("Starlink + ISS")
    constellation_with_iss.add_from_tle_list(SAMPLE_STARLINK_TLES)
    constellation_with_iss.add_satellite(
        "ISS",
        "1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990",
        "2 25544  51.6443 242.2052 0004885 264.6463 206.3557 15.49165514212791"
    )

    # Ground positions for testing
    test_points = [
        ("London, UK", GroundPosition(51.5074, -0.1278, 0.0)),
        ("New York, US", GroundPosition(40.7128, -74.0060, 0.0)),
        ("Tokyo, Japan", GroundPosition(35.6762, 139.6503, 0.0)),
        ("Sydney, Australia", GroundPosition(-33.8688, 151.2093, 0.0)),
    ]

    # Time instant
    dt = datetime(2020, 2, 14, 12, 0, 0, tzinfo=timezone.utc)

    print(f"\nTime: {dt}")
    print(f"\n--- Single Time Instant Visibility ---")

    for city, ground in test_points:
        # Check visibility for sample constellation
        visible, sat_name = constellation.check_any_visible(ground, dt, 10.0)
        count = constellation.count_visible(ground, dt, 10.0)
        status = f"Yes ({count} sats, including {sat_name})" if visible else "No"
        print(f"  {city}: {status}")

    import time as time_module
    config = ConstellationConfig(min_elevation=10.0, time_step_seconds=60.0)

    # Compare scalar vs vectorized coverage map calculation
    print(f"\n--- 2-Hour Coverage Map Performance Comparison (5° resolution) ---")

    # Scalar version
    print("\nScalar computation:")
    t_start = time_module.time()
    coverage_scalar = calculate_constellation_coverage_map_scalar(
        constellation,
        dt,
        duration_hours=2.0,
        resolution_deg=5.0,
        config=config,
        progress_callback=None
    )
    scalar_time = time_module.time() - t_start
    print(f"  Time: {scalar_time:.2f}s")
    print(f"  Average coverage: {coverage_scalar.average_coverage:.1f}%")

    # Vectorized version
    print("\nVectorized computation:")
    t_start = time_module.time()
    coverage = calculate_constellation_coverage_map(
        constellation,
        dt,
        duration_hours=2.0,
        resolution_deg=5.0,
        config=config,
        progress_callback=None
    )
    vectorized_time = time_module.time() - t_start
    print(f"  Time: {vectorized_time:.2f}s")
    print(f"  Average coverage: {coverage.average_coverage:.1f}%")

    # Verify results match
    if np.allclose(coverage_scalar.coverage_map, coverage.coverage_map, rtol=1e-5):
        print("\nPASS: Vectorized results match scalar results")
    else:
        diff = np.abs(coverage_scalar.coverage_map - coverage.coverage_map)
        print(f"\nWARNING: Max difference: {np.max(diff):.6f}%")

    # Speedup
    speedup = scalar_time / vectorized_time if vectorized_time > 0 else float('inf')
    print(f"\nVectorization speedup: {speedup:.1f}x")

    print(f"\n--- Coverage Map Results ---")
    print(f"  Grid size: {coverage.latitudes.shape[0]} x {coverage.longitudes.shape[0]}")

    # Find best and worst coverage regions
    max_cov = np.max(coverage.coverage_map)
    min_cov = np.min(coverage.coverage_map)
    max_idx = np.unravel_index(np.argmax(coverage.coverage_map), coverage.coverage_map.shape)
    min_idx = np.unravel_index(np.argmin(coverage.coverage_map), coverage.coverage_map.shape)

    print(f"\nCoverage extremes:")
    print(f"  Best: {max_cov:.1f}% at ({coverage.latitudes[max_idx[0]]:.0f}°, {coverage.longitudes[max_idx[1]]:.0f}°)")
    print(f"  Worst: {min_cov:.1f}% at ({coverage.latitudes[min_idx[0]]:.0f}°, {coverage.longitudes[min_idx[1]]:.0f}°)")

    # Single-point 24-hour coverage
    print(f"\n--- 24-Hour Point Coverage (London) ---")
    london = GroundPosition(51.5074, -0.1278, 0.0)

    t_start = time_module.time()
    coverage_percent = calculate_constellation_coverage_at_point(
        constellation, london, dt, duration_days=1.0, config=config
    )
    elapsed = time_module.time() - t_start

    print(f"  Coverage: {coverage_percent:.2f}%")
    print(f"  Computation time: {elapsed:.2f}s")

    return coverage


if __name__ == "__main__":
    demo_constellation_coverage()
