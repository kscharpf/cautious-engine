#!/usr/bin/env python3
"""
Command-line interface for satellite coverage calculations.

Usage examples:
    python cli.py --tle-file satellites.txt --lat 40.0 --lon -74.0 --duration 24
    python cli.py --celestrak starlink --lat 51.5 --lon -0.1 --duration 48 -v
    python cli.py --tle-file sats.txt --coverage-map --resolution 5 -o result.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Optional

from tle_loader import (
    parse_tle_file,
    parse_tle_text,
    fetch_tle_from_url_sync,
    fetch_celestrak_preset_sync,
    get_celestrak_presets,
    TLEParseError,
    TLEFetchError,
)
from constellation import (
    Constellation,
    ConstellationConfig,
    calculate_constellation_coverage_at_point,
    calculate_constellation_coverage_map,
)
from satellite import GroundPosition


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog='ralph-cli',
        description='Satellite coverage calculator CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --celestrak iss --lat 51.5 --lon -0.1 --duration 24
  %(prog)s --tle-file my_sats.txt --lat 40 --lon -74 --duration 12 -v
  %(prog)s --celestrak gps --coverage-map --resolution 10 -o coverage.json
  %(prog)s --tle-url "https://example.com/satellites.txt" --lat 0 --lon 0

Available CelesTrak presets:
  iss, starlink, gps, weather, noaa, goes, galileo, glonass, beidou, iridium, planet, spire, active
""",
    )

    # TLE input (mutually exclusive)
    tle_group = parser.add_mutually_exclusive_group(required=True)
    tle_group.add_argument(
        '--tle-file',
        metavar='PATH',
        help='Path to TLE file',
    )
    tle_group.add_argument(
        '--celestrak',
        metavar='PRESET',
        help='CelesTrak preset name (e.g., iss, starlink, gps)',
    )
    tle_group.add_argument(
        '--tle-url',
        metavar='URL',
        help='URL to fetch TLE data from',
    )
    tle_group.add_argument(
        '--list-presets',
        action='store_true',
        help='List available CelesTrak presets and exit',
    )

    # Ground position
    parser.add_argument(
        '--lat', '--latitude',
        type=float,
        metavar='DEG',
        help='Ground station latitude (-90 to 90 degrees)',
    )
    parser.add_argument(
        '--lon', '--longitude',
        type=float,
        metavar='DEG',
        help='Ground station longitude (-180 to 180 degrees)',
    )
    parser.add_argument(
        '--alt', '--altitude',
        type=float,
        default=0.0,
        metavar='KM',
        help='Ground station altitude in km (default: 0)',
    )

    # Time parameters
    parser.add_argument(
        '--duration',
        type=float,
        default=24.0,
        metavar='HOURS',
        help='Simulation duration in hours (default: 24)',
    )
    parser.add_argument(
        '--start',
        type=str,
        metavar='ISO',
        help='Start time in ISO format (default: now)',
    )

    # Coverage options
    parser.add_argument(
        '--min-elevation',
        type=float,
        default=10.0,
        metavar='DEG',
        help='Minimum elevation angle in degrees (default: 10)',
    )
    parser.add_argument(
        '--coverage-map',
        action='store_true',
        help='Calculate coverage map instead of point coverage',
    )
    parser.add_argument(
        '--resolution',
        type=float,
        default=5.0,
        metavar='DEG',
        help='Grid resolution in degrees for coverage map (default: 5)',
    )

    # Output options
    parser.add_argument(
        '-o', '--output',
        metavar='PATH',
        help='Output file path (default: stdout)',
    )
    parser.add_argument(
        '--format',
        choices=['text', 'json', 'csv'],
        default='text',
        help='Output format (default: text)',
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output',
    )

    return parser


def list_presets():
    """List available CelesTrak presets."""
    presets = get_celestrak_presets()
    print("Available CelesTrak presets:\n")
    for name, url in sorted(presets.items()):
        print(f"  {name:12} {url}")
    print()


def load_tles(args) -> list:
    """Load TLEs from the specified source."""
    if args.tle_file:
        if args.verbose:
            print(f"Loading TLEs from file: {args.tle_file}")
        return parse_tle_file(args.tle_file)

    elif args.celestrak:
        if args.verbose:
            print(f"Fetching TLEs from CelesTrak preset: {args.celestrak}")
        return fetch_celestrak_preset_sync(args.celestrak)

    elif args.tle_url:
        if args.verbose:
            print(f"Fetching TLEs from URL: {args.tle_url}")
        text = fetch_tle_from_url_sync(args.tle_url)
        return parse_tle_text(text)

    return []


def format_point_result(result: dict, fmt: str) -> str:
    """Format point coverage result."""
    if fmt == 'json':
        return json.dumps(result, indent=2)

    elif fmt == 'csv':
        lines = ['latitude,longitude,duration_hours,coverage_percent,num_satellites']
        lines.append(f"{result['latitude']},{result['longitude']},{result['duration_hours']},{result['coverage_percent']:.2f},{result['num_satellites']}")
        return '\n'.join(lines)

    else:  # text
        lines = [
            "=" * 50,
            "SATELLITE COVERAGE RESULTS",
            "=" * 50,
            "",
            f"Ground Position: {result['latitude']:.4f}°, {result['longitude']:.4f}°",
            f"Duration: {result['duration_hours']} hours",
            f"Satellites: {result['num_satellites']}",
            f"Min Elevation: {result['min_elevation']}°",
            "",
            f"Coverage: {result['coverage_percent']:.2f}%",
            "",
        ]
        return '\n'.join(lines)


def format_map_result(result: dict, fmt: str) -> str:
    """Format coverage map result."""
    if fmt == 'json':
        return json.dumps(result, indent=2)

    elif fmt == 'csv':
        lines = ['latitude,longitude,coverage_percent']
        coverage = result['coverage_map']
        lats = result['latitudes']
        lons = result['longitudes']
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                lines.append(f"{lat},{lon},{coverage[i][j]:.2f}")
        return '\n'.join(lines)

    else:  # text
        lines = [
            "=" * 50,
            "SATELLITE COVERAGE MAP RESULTS",
            "=" * 50,
            "",
            f"Grid: {len(result['latitudes'])} x {len(result['longitudes'])} points",
            f"Resolution: {result['resolution']}°",
            f"Duration: {result['duration_hours']} hours",
            f"Satellites: {result['num_satellites']}",
            f"Min Elevation: {result['min_elevation']}°",
            "",
            f"Average Coverage: {result['average_coverage']:.2f}%",
            f"Min Coverage: {result['min_coverage']:.2f}%",
            f"Max Coverage: {result['max_coverage']:.2f}%",
            "",
        ]
        return '\n'.join(lines)


def run_point_coverage(constellation: Constellation, args) -> dict:
    """Run point coverage calculation."""
    if args.lat is None or args.lon is None:
        raise ValueError("--lat and --lon are required for point coverage")

    ground = GroundPosition(args.lat, args.lon, args.alt)

    if args.start:
        start_time = datetime.fromisoformat(args.start.replace('Z', '+00:00'))
    else:
        start_time = datetime.now(timezone.utc)

    config = ConstellationConfig(
        min_elevation=args.min_elevation,
        time_step_seconds=60.0,
    )

    if args.verbose:
        print(f"Calculating coverage for ({args.lat}, {args.lon})...")
        print(f"Duration: {args.duration} hours, Min elevation: {args.min_elevation}°")

    coverage_percent = calculate_constellation_coverage_at_point(
        constellation,
        ground,
        start_time,
        duration_days=args.duration / 24.0,
        config=config,
    )

    return {
        'latitude': args.lat,
        'longitude': args.lon,
        'altitude_km': args.alt,
        'duration_hours': args.duration,
        'start_time': start_time.isoformat(),
        'coverage_percent': coverage_percent,
        'num_satellites': constellation.size,
        'min_elevation': args.min_elevation,
    }


def run_coverage_map(constellation: Constellation, args) -> dict:
    """Run coverage map calculation."""
    if args.start:
        start_time = datetime.fromisoformat(args.start.replace('Z', '+00:00'))
    else:
        start_time = datetime.now(timezone.utc)

    config = ConstellationConfig(
        min_elevation=args.min_elevation,
        time_step_seconds=60.0,
    )

    def progress_callback(step, total, current_time):
        if args.verbose:
            pct = (step / total) * 100
            print(f"  Progress: {pct:.0f}% ({step}/{total} steps)")

    if args.verbose:
        print(f"Calculating coverage map at {args.resolution}° resolution...")
        print(f"Duration: {args.duration} hours, Min elevation: {args.min_elevation}°")

    coverage = calculate_constellation_coverage_map(
        constellation,
        start_time,
        duration_hours=args.duration,
        resolution_deg=args.resolution,
        config=config,
        progress_callback=progress_callback if args.verbose else None,
    )

    return {
        'resolution': args.resolution,
        'duration_hours': args.duration,
        'start_time': start_time.isoformat(),
        'num_satellites': constellation.size,
        'min_elevation': args.min_elevation,
        'average_coverage': float(coverage.average_coverage),
        'min_coverage': float(coverage.coverage_map.min()),
        'max_coverage': float(coverage.coverage_map.max()),
        'latitudes': coverage.latitudes.tolist(),
        'longitudes': coverage.longitudes.tolist(),
        'coverage_map': coverage.coverage_map.tolist(),
    }


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle --list-presets
    if args.list_presets:
        list_presets()
        return 0

    try:
        # Load TLEs
        tles = load_tles(args)

        if not tles:
            print("Error: No TLEs loaded", file=sys.stderr)
            return 1

        if args.verbose:
            print(f"Loaded {len(tles)} satellites")

        # Create constellation
        constellation = Constellation("CLI Constellation")
        constellation.add_from_tle_list(tles)

        if args.verbose:
            print(f"Constellation size: {constellation.size}")
            for name, _ in constellation.satellites[:5]:
                print(f"  - {name}")
            if constellation.size > 5:
                print(f"  ... and {constellation.size - 5} more")

        # Run calculation
        if args.coverage_map:
            result = run_coverage_map(constellation, args)
            output = format_map_result(result, args.format)
        else:
            if args.lat is None or args.lon is None:
                print("Error: --lat and --lon are required for point coverage", file=sys.stderr)
                print("Use --coverage-map for global coverage calculation", file=sys.stderr)
                return 1
            result = run_point_coverage(constellation, args)
            output = format_point_result(result, args.format)

        # Output result
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            if args.verbose:
                print(f"Results written to: {args.output}")
        else:
            print(output)

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except TLEParseError as e:
        print(f"TLE Parse Error: {e}", file=sys.stderr)
        return 1
    except TLEFetchError as e:
        print(f"TLE Fetch Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
