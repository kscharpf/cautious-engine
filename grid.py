"""
Adaptive grid algorithm for multi-point satellite coverage calculation.

This module implements a coarse-to-fine grid approach that:
1. Starts with coarse grid over Earth
2. Identifies cells at visibility boundaries
3. Refines those cells for higher resolution
"""

import math
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict, Set, Optional, NamedTuple
from dataclasses import dataclass, field
import numpy as np

from satellite import (
    Satrec, parse_tle, GroundPosition, LookAngle,
    propagate, datetime_to_gmst, teme_to_ecef, calculate_look_angle, is_visible,
    calculate_visibility_grid, geodetic_to_ecef_grid, calculate_look_angles_grid
)
from propagation import PropagationConfig


@dataclass
class GridCell:
    """A cell in the adaptive grid."""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    coverage_sum: float = 0.0  # Sum of coverage fractions at sample points
    sample_count: int = 0
    needs_refinement: bool = False
    refined: bool = False

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.lat_min + self.lat_max) / 2,
                (self.lon_min + self.lon_max) / 2)

    @property
    def lat_size(self) -> float:
        return self.lat_max - self.lat_min

    @property
    def lon_size(self) -> float:
        return self.lon_max - self.lon_min


@dataclass
class GridConfig:
    """Configuration for adaptive grid algorithm."""
    initial_resolution_deg: float = 10.0  # Starting grid resolution
    min_resolution_deg: float = 2.0       # Minimum refinement resolution
    min_elevation: float = 10.0           # Minimum elevation for visibility
    lat_bounds: Tuple[float, float] = (-90.0, 90.0)
    lon_bounds: Tuple[float, float] = (-180.0, 180.0)


def create_initial_grid(config: GridConfig) -> List[GridCell]:
    """Create the initial coarse grid over Earth."""
    cells = []
    lat = config.lat_bounds[0]
    while lat < config.lat_bounds[1]:
        lat_next = min(lat + config.initial_resolution_deg, config.lat_bounds[1])
        lon = config.lon_bounds[0]
        while lon < config.lon_bounds[1]:
            lon_next = min(lon + config.initial_resolution_deg, config.lon_bounds[1])
            cells.append(GridCell(lat, lat_next, lon, lon_next))
            lon = lon_next
        lat = lat_next
    return cells


def sample_cell_corners(cell: GridCell) -> List[GroundPosition]:
    """Get the 4 corners and center of a grid cell as ground positions."""
    return [
        GroundPosition(cell.lat_min, cell.lon_min, 0.0),
        GroundPosition(cell.lat_min, cell.lon_max, 0.0),
        GroundPosition(cell.lat_max, cell.lon_min, 0.0),
        GroundPosition(cell.lat_max, cell.lon_max, 0.0),
        GroundPosition(cell.center[0], cell.center[1], 0.0),
    ]


def check_visibility_at_points(
    satellite: Satrec,
    points: List[GroundPosition],
    dt: datetime,
    min_elevation: float
) -> List[bool]:
    """Check visibility at multiple points for a single time instant."""
    # Propagate satellite once
    pos, _ = propagate(satellite, dt)
    gmst = datetime_to_gmst(dt)
    sat_ecef = teme_to_ecef(pos, gmst)

    results = []
    for point in points:
        look = calculate_look_angle(sat_ecef, point)
        results.append(is_visible(look, min_elevation))
    return results


def classify_cell_visibility(
    satellite: Satrec,
    cell: GridCell,
    dt: datetime,
    min_elevation: float
) -> Tuple[bool, bool, bool]:
    """
    Classify a cell's visibility state.

    Returns:
        (all_visible, all_hidden, needs_refinement)
    """
    corners = sample_cell_corners(cell)
    visibility = check_visibility_at_points(satellite, corners, dt, min_elevation)

    visible_count = sum(visibility)
    total = len(visibility)

    all_visible = visible_count == total
    all_hidden = visible_count == 0
    needs_refinement = not (all_visible or all_hidden)  # Mixed visibility

    return all_visible, all_hidden, needs_refinement


def subdivide_cell(cell: GridCell) -> List[GridCell]:
    """Subdivide a cell into 4 smaller cells."""
    mid_lat = (cell.lat_min + cell.lat_max) / 2
    mid_lon = (cell.lon_min + cell.lon_max) / 2

    return [
        GridCell(cell.lat_min, mid_lat, cell.lon_min, mid_lon),
        GridCell(cell.lat_min, mid_lat, mid_lon, cell.lon_max),
        GridCell(mid_lat, cell.lat_max, cell.lon_min, mid_lon),
        GridCell(mid_lat, cell.lat_max, mid_lon, cell.lon_max),
    ]


def adaptive_visibility_grid(
    satellite: Satrec,
    dt: datetime,
    config: GridConfig,
    max_refinement_iterations: int = 3
) -> Tuple[List[GridCell], Dict[str, float]]:
    """
    Calculate visibility over Earth using adaptive grid refinement.

    Returns:
        List of final grid cells with visibility classification
        Dict of statistics
    """
    # Create initial coarse grid
    cells = create_initial_grid(config)
    stats = {
        'initial_cells': len(cells),
        'refinement_iterations': 0,
        'visibility_checks': 0,
    }

    for iteration in range(max_refinement_iterations):
        cells_to_refine = []
        final_cells = []

        for cell in cells:
            all_vis, all_hid, needs_refine = classify_cell_visibility(
                satellite, cell, dt, config.min_elevation
            )
            stats['visibility_checks'] += 5  # 5 points per cell

            if needs_refine and cell.lat_size > config.min_resolution_deg:
                cells_to_refine.append(cell)
            else:
                cell.needs_refinement = needs_refine
                # Calculate visibility percentage
                if all_vis:
                    cell.coverage_sum = 1.0
                elif all_hid:
                    cell.coverage_sum = 0.0
                else:
                    # Edge cell at minimum resolution - estimate 50%
                    cell.coverage_sum = 0.5
                cell.sample_count = 1
                final_cells.append(cell)

        if not cells_to_refine:
            cells = final_cells
            break

        # Refine cells that need it
        new_cells = []
        for cell in cells_to_refine:
            new_cells.extend(subdivide_cell(cell))

        cells = final_cells + new_cells
        stats['refinement_iterations'] += 1

    stats['final_cells'] = len(cells)
    stats['visible_cells'] = sum(1 for c in cells if c.coverage_sum > 0.5)
    stats['boundary_cells'] = sum(1 for c in cells if 0 < c.coverage_sum < 1)

    return cells, stats


def calculate_instant_coverage_map(
    satellite: Satrec,
    dt: datetime,
    resolution_deg: float = 2.0,
    min_elevation: float = 10.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate a visibility map for a single instant using uniform grid.

    Returns a 2D numpy array where 1 = visible, 0 = not visible.
    Uses vectorized computation for significant performance improvement.
    """
    # Create lat/lon grid
    lats = np.arange(-90, 90, resolution_deg)
    lons = np.arange(-180, 180, resolution_deg)

    # Use vectorized visibility calculation
    visibility, _, _, _ = calculate_visibility_grid(
        satellite, dt, lats, lons, min_elevation
    )

    # Convert boolean to float32 for compatibility
    visibility_map = visibility.astype(np.float32)

    return visibility_map, lats, lons


def calculate_instant_coverage_map_scalar(
    satellite: Satrec,
    dt: datetime,
    resolution_deg: float = 2.0,
    min_elevation: float = 10.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate a visibility map using scalar (non-vectorized) computation.

    Provided for comparison and validation against vectorized version.
    """
    # Create lat/lon grid
    lats = np.arange(-90, 90, resolution_deg)
    lons = np.arange(-180, 180, resolution_deg)

    visibility_map = np.zeros((len(lats), len(lons)), dtype=np.float32)

    # Propagate satellite once
    pos, _ = propagate(satellite, dt)
    gmst = datetime_to_gmst(dt)
    sat_ecef = teme_to_ecef(pos, gmst)

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            ground = GroundPosition(lat, lon, 0.0)
            look = calculate_look_angle(sat_ecef, ground)
            if is_visible(look, min_elevation):
                visibility_map[i, j] = 1.0

    return visibility_map, lats, lons


def demo_adaptive_grid():
    """Demonstrate the adaptive grid algorithm."""
    print("=" * 70)
    print("ADAPTIVE GRID VISIBILITY ALGORITHM DEMONSTRATION")
    print("=" * 70)

    # ISS TLE
    tle_line1 = "1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990"
    tle_line2 = "2 25544  51.6443 242.2052 0004885 264.6463 206.3557 15.49165514212791"

    satellite = parse_tle(tle_line1, tle_line2)

    # Time instant
    dt = datetime(2020, 2, 14, 12, 0, 0, tzinfo=timezone.utc)

    print(f"\nSatellite: ISS (ZARYA)")
    print(f"Time: {dt}")

    import time as time_module

    # Test 1: Scalar (non-vectorized) uniform grid
    print("\n--- Scalar Uniform Grid (2° resolution) ---")

    t_start = time_module.time()
    vis_map_scalar, lats, lons = calculate_instant_coverage_map_scalar(
        satellite, dt, resolution_deg=2.0, min_elevation=10.0
    )
    scalar_time = time_module.time() - t_start

    uniform_checks = len(lats) * len(lons)
    visible_points_scalar = np.sum(vis_map_scalar)

    print(f"Grid size: {len(lats)} x {len(lons)} = {uniform_checks} points")
    print(f"Visible points: {int(visible_points_scalar)}")
    print(f"Time: {scalar_time:.3f}s")

    # Test 2: Vectorized uniform grid
    print("\n--- Vectorized Uniform Grid (2° resolution) ---")

    t_start = time_module.time()
    vis_map_vec, lats, lons = calculate_instant_coverage_map(
        satellite, dt, resolution_deg=2.0, min_elevation=10.0
    )
    vectorized_time = time_module.time() - t_start

    visible_points_vec = np.sum(vis_map_vec)

    print(f"Grid size: {len(lats)} x {len(lons)} = {uniform_checks} points")
    print(f"Visible points: {int(visible_points_vec)}")
    print(f"Time: {vectorized_time:.3f}s")

    # Verify results match
    if np.allclose(vis_map_scalar, vis_map_vec):
        print("PASS: Vectorized results match scalar results")
    else:
        diff_count = np.sum(vis_map_scalar != vis_map_vec)
        print(f"WARNING: {diff_count} points differ between scalar and vectorized")

    # Speedup from vectorization
    vec_speedup = scalar_time / vectorized_time if vectorized_time > 0 else float('inf')
    print(f"\nVectorization speedup: {vec_speedup:.1f}x")

    # Test 3: Adaptive grid
    print("\n--- Adaptive Grid (10° -> 2.5° refinement) ---")

    config = GridConfig(
        initial_resolution_deg=10.0,
        min_resolution_deg=2.5,
        min_elevation=10.0
    )

    t_start = time_module.time()
    cells, stats = adaptive_visibility_grid(satellite, dt, config)
    adaptive_time = time_module.time() - t_start

    print(f"Initial cells: {stats['initial_cells']}")
    print(f"Final cells: {stats['final_cells']}")
    print(f"Refinement iterations: {stats['refinement_iterations']}")
    print(f"Visibility checks: {stats['visibility_checks']}")
    print(f"Visible cells: {stats['visible_cells']}")
    print(f"Boundary cells: {stats['boundary_cells']}")
    print(f"Time: {adaptive_time:.3f}s")

    # Compare efficiency
    print("\n--- Efficiency Summary ---")
    print(f"Scalar uniform:     {scalar_time:.3f}s (baseline)")
    print(f"Vectorized uniform: {vectorized_time:.3f}s ({vec_speedup:.1f}x faster)")
    print(f"Adaptive grid:      {adaptive_time:.3f}s")

    reduction = (1 - stats['visibility_checks'] / uniform_checks) * 100
    print(f"\nAdaptive check reduction: {reduction:.1f}%")

    return cells, stats


if __name__ == "__main__":
    demo_adaptive_grid()
