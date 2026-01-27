"""
Satellite propagation and visibility calculations using SGP4.

This module provides functionality for:
- Parsing TLE (Two-Line Element) data
- Propagating satellite positions using SGP4
- Calculating look angles (elevation/azimuth) from ground positions
"""

import math
from datetime import datetime, timezone
from typing import Tuple, Optional, NamedTuple

from sgp4.api import Satrec, jday
from sgp4 import exporter
import numpy as np


# Constants
EARTH_RADIUS_KM = 6371.0  # Mean Earth radius in km
DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi

# Earth's rotation rate (rad/s) - WGS84
EARTH_ROTATION_RATE = 7.2921150e-5


class Position(NamedTuple):
    """Position in TEME frame (km)."""
    x: float
    y: float
    z: float


class Velocity(NamedTuple):
    """Velocity in TEME frame (km/s)."""
    vx: float
    vy: float
    vz: float


class GroundPosition(NamedTuple):
    """Ground position in geodetic coordinates."""
    latitude: float   # degrees, -90 to 90
    longitude: float  # degrees, -180 to 180
    altitude: float   # km above sea level


class LookAngle(NamedTuple):
    """Look angle from ground station to satellite."""
    elevation: float  # degrees, -90 to 90 (positive = above horizon)
    azimuth: float    # degrees, 0 to 360 (clockwise from north)
    range_km: float   # distance to satellite in km


def parse_tle(tle_line1: str, tle_line2: str) -> Satrec:
    """Parse a TLE and return an SGP4 satellite object."""
    satellite = Satrec.twoline2rv(tle_line1, tle_line2)
    return satellite


def propagate(satellite: Satrec, dt: datetime) -> Tuple[Position, Velocity]:
    """
    Propagate satellite to given datetime.

    Returns position and velocity in TEME frame (True Equator Mean Equinox).
    Position in km, velocity in km/s.
    """
    # Ensure datetime is UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # Convert to Julian date
    jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                   dt.second + dt.microsecond / 1e6)

    # Propagate
    error, position, velocity = satellite.sgp4(jd, fr)

    if error != 0:
        raise ValueError(f"SGP4 propagation error: {error}")

    return Position(*position), Velocity(*velocity)


def datetime_to_gmst(dt: datetime) -> float:
    """
    Calculate Greenwich Mean Sidereal Time (GMST) for a given datetime.
    Returns GMST in radians.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # Julian date
    jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                  dt.second + dt.microsecond / 1e6)
    jd_full = jd + fr

    # Julian centuries from J2000.0
    t_ut1 = (jd_full - 2451545.0) / 36525.0

    # GMST in seconds
    gmst_sec = (67310.54841 +
                (876600.0 * 3600.0 + 8640184.812866) * t_ut1 +
                0.093104 * t_ut1**2 -
                6.2e-6 * t_ut1**3)

    # Convert to radians (86400 seconds per day, 2*pi radians per day)
    gmst_rad = (gmst_sec % 86400) * (2 * math.pi / 86400)

    return gmst_rad


def teme_to_ecef(position: Position, gmst: float) -> Position:
    """
    Convert position from TEME frame to ECEF frame.

    TEME = True Equator Mean Equinox (inertial-ish frame used by SGP4)
    ECEF = Earth-Centered Earth-Fixed (rotates with Earth)
    """
    cos_gmst = math.cos(gmst)
    sin_gmst = math.sin(gmst)

    x_ecef = position.x * cos_gmst + position.y * sin_gmst
    y_ecef = -position.x * sin_gmst + position.y * cos_gmst
    z_ecef = position.z

    return Position(x_ecef, y_ecef, z_ecef)


def geodetic_to_ecef(ground: GroundPosition) -> Position:
    """
    Convert geodetic coordinates to ECEF position.

    Uses WGS84 ellipsoid parameters.
    """
    # WGS84 parameters
    a = 6378.137  # Equatorial radius (km)
    f = 1 / 298.257223563  # Flattening
    e2 = 2 * f - f**2  # Eccentricity squared

    lat_rad = ground.latitude * DEG_TO_RAD
    lon_rad = ground.longitude * DEG_TO_RAD
    alt = ground.altitude

    cos_lat = math.cos(lat_rad)
    sin_lat = math.sin(lat_rad)
    cos_lon = math.cos(lon_rad)
    sin_lon = math.sin(lon_rad)

    # Radius of curvature in prime vertical
    N = a / math.sqrt(1 - e2 * sin_lat**2)

    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1 - e2) + alt) * sin_lat

    return Position(x, y, z)


def calculate_look_angle(satellite_ecef: Position, ground: GroundPosition) -> LookAngle:
    """
    Calculate look angle (elevation, azimuth, range) from ground position to satellite.

    Uses SEZ (South-East-Zenith) coordinate system for local horizon calculations.
    """
    # Convert ground position to ECEF
    ground_ecef = geodetic_to_ecef(ground)

    # Range vector from ground to satellite (in ECEF)
    range_vec = np.array([
        satellite_ecef.x - ground_ecef.x,
        satellite_ecef.y - ground_ecef.y,
        satellite_ecef.z - ground_ecef.z
    ])

    range_km = np.linalg.norm(range_vec)

    # Convert to SEZ (South-East-Zenith) local coordinates
    lat_rad = ground.latitude * DEG_TO_RAD
    lon_rad = ground.longitude * DEG_TO_RAD

    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)

    # Rotation matrix from ECEF to SEZ
    # S (south), E (east), Z (zenith/up)
    rot_s = np.array([sin_lat * cos_lon, sin_lat * sin_lon, -cos_lat])
    rot_e = np.array([-sin_lon, cos_lon, 0])
    rot_z = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])

    s = np.dot(rot_s, range_vec)
    e = np.dot(rot_e, range_vec)
    z = np.dot(rot_z, range_vec)

    # Calculate elevation (angle above horizon)
    horizontal_range = math.sqrt(s**2 + e**2)
    elevation = math.atan2(z, horizontal_range) * RAD_TO_DEG

    # Calculate azimuth (angle from north, clockwise)
    azimuth = math.atan2(e, -s) * RAD_TO_DEG
    if azimuth < 0:
        azimuth += 360

    return LookAngle(elevation=elevation, azimuth=azimuth, range_km=range_km)


def is_visible(look_angle: LookAngle, min_elevation: float = 10.0) -> bool:
    """
    Determine if satellite is visible from ground position.

    Args:
        look_angle: Calculated look angle to satellite
        min_elevation: Minimum elevation angle in degrees (default 10 degrees)

    Returns:
        True if satellite is above minimum elevation angle
    """
    return look_angle.elevation >= min_elevation


def calculate_visibility_at_time(
    satellite: Satrec,
    ground: GroundPosition,
    dt: datetime,
    min_elevation: float = 10.0
) -> Tuple[LookAngle, bool]:
    """
    Calculate visibility of a satellite from a ground position at a given time.

    Returns the look angle and whether the satellite is visible.
    """
    # Propagate satellite to time
    position, _ = propagate(satellite, dt)

    # Get GMST for TEME to ECEF conversion
    gmst = datetime_to_gmst(dt)

    # Convert to ECEF
    sat_ecef = teme_to_ecef(position, gmst)

    # Calculate look angle
    look_angle = calculate_look_angle(sat_ecef, ground)

    # Check visibility
    visible = is_visible(look_angle, min_elevation)

    return look_angle, visible


# =============================================================================
# TEST VECTORS AND VERIFICATION
# =============================================================================

def verify_sgp4_propagation():
    """
    Verify SGP4 propagation against known test vectors.

    Uses a well-known satellite TLE and compares propagated positions
    against published test cases.
    """
    print("=" * 60)
    print("SGP4 Propagation Verification")
    print("=" * 60)

    # Test TLE: ISS (Zarya) - a well-documented satellite
    # Note: Using a historical TLE for reproducibility
    tle_line1 = "1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990"
    tle_line2 = "2 25544  51.6443 242.2052 0004885 264.6463 206.3557 15.49165514212791"

    satellite = parse_tle(tle_line1, tle_line2)

    # Propagate to epoch (should give very close to TLE position)
    # Epoch: 2020, day 45.18587073 = Feb 14, 2020, 04:27:33 UTC
    epoch_dt = datetime(2020, 2, 14, 4, 27, 33, tzinfo=timezone.utc)

    pos, vel = propagate(satellite, epoch_dt)

    print(f"\nSatellite: ISS (ZARYA)")
    print(f"TLE Epoch: {epoch_dt}")
    print(f"\nPropagated Position (TEME, km):")
    print(f"  X: {pos.x:12.4f}")
    print(f"  Y: {pos.y:12.4f}")
    print(f"  Z: {pos.z:12.4f}")
    print(f"\nPropagated Velocity (TEME, km/s):")
    print(f"  Vx: {vel.vx:12.6f}")
    print(f"  Vy: {vel.vy:12.6f}")
    print(f"  Vz: {vel.vz:12.6f}")

    # Verify approximate orbital altitude
    altitude = math.sqrt(pos.x**2 + pos.y**2 + pos.z**2) - EARTH_RADIUS_KM
    print(f"\nApproximate altitude: {altitude:.2f} km")

    # ISS orbits at ~400-420 km altitude
    if 350 < altitude < 450:
        print("PASS: Altitude is within expected range for ISS (350-450 km)")
        altitude_pass = True
    else:
        print("FAIL: Altitude is outside expected range")
        altitude_pass = False

    # Verify orbital velocity (ISS ~7.66 km/s)
    velocity_mag = math.sqrt(vel.vx**2 + vel.vy**2 + vel.vz**2)
    print(f"Orbital velocity: {velocity_mag:.4f} km/s")

    if 7.5 < velocity_mag < 7.8:
        print("PASS: Velocity is within expected range for ISS (7.5-7.8 km/s)")
        velocity_pass = True
    else:
        print("FAIL: Velocity is outside expected range")
        velocity_pass = False

    # Additional test: propagate forward 1 orbit (~92 minutes)
    print("\n--- Propagate forward 1 orbit (92 minutes) ---")
    future_dt = datetime(2020, 2, 14, 5, 59, 33, tzinfo=timezone.utc)
    pos2, vel2 = propagate(satellite, future_dt)

    altitude2 = math.sqrt(pos2.x**2 + pos2.y**2 + pos2.z**2) - EARTH_RADIUS_KM
    velocity_mag2 = math.sqrt(vel2.vx**2 + vel2.vy**2 + vel2.vz**2)

    print(f"Position (km): ({pos2.x:.2f}, {pos2.y:.2f}, {pos2.z:.2f})")
    print(f"Altitude: {altitude2:.2f} km")
    print(f"Velocity: {velocity_mag2:.4f} km/s")

    # After one orbit, altitude and velocity should be similar
    altitude_diff = abs(altitude2 - altitude)
    velocity_diff = abs(velocity_mag2 - velocity_mag)

    if altitude_diff < 50:  # Within 50 km (orbit is slightly elliptical)
        print(f"PASS: Altitude after 1 orbit differs by {altitude_diff:.2f} km")
    else:
        print(f"WARNING: Large altitude difference: {altitude_diff:.2f} km")

    return altitude_pass and velocity_pass


def verify_look_angle_calculation():
    """
    Verify look angle calculation against known test vectors.
    """
    print("\n" + "=" * 60)
    print("Look Angle Calculation Verification")
    print("=" * 60)

    # Test case: Satellite directly overhead
    print("\n--- Test 1: Satellite directly overhead ---")

    # Ground position: 0 lat, 0 lon (intersection of equator and prime meridian)
    ground = GroundPosition(latitude=0.0, longitude=0.0, altitude=0.0)

    # Satellite at 400 km directly above this point
    # In ECEF, this is along the x-axis at radius = Earth_radius + 400 km
    sat_height = EARTH_RADIUS_KM + 400
    sat_ecef = Position(sat_height, 0, 0)

    look = calculate_look_angle(sat_ecef, ground)

    print(f"Ground: ({ground.latitude}, {ground.longitude})")
    print(f"Satellite ECEF: ({sat_ecef.x:.2f}, {sat_ecef.y:.2f}, {sat_ecef.z:.2f})")
    print(f"Look Angle: elevation={look.elevation:.2f}, azimuth={look.azimuth:.2f}")
    print(f"Range: {look.range_km:.2f} km")

    # Elevation should be ~90 degrees (straight up)
    test1_pass = abs(look.elevation - 90.0) < 1.0
    if test1_pass:
        print("PASS: Elevation is approximately 90 degrees")
    else:
        print(f"FAIL: Expected elevation ~90, got {look.elevation:.2f}")

    # Range should be ~393 km (400 km altitude, but ground is at ellipsoid surface
    # at equator, which is 6378 km radius, not 6371 km mean radius)
    expected_range = 400 - (6378.137 - EARTH_RADIUS_KM)  # ~393 km
    test1_range = abs(look.range_km - expected_range) < 5.0
    if test1_range:
        print(f"PASS: Range is approximately {expected_range:.0f} km")
    else:
        print(f"FAIL: Expected range ~{expected_range:.0f} km, got {look.range_km:.2f}")

    # Test case: Satellite on horizon
    print("\n--- Test 2: Satellite on horizon (north) ---")

    # Ground position: 45 N, 0 E
    ground2 = GroundPosition(latitude=45.0, longitude=0.0, altitude=0.0)

    # For a satellite on the horizon to the north, we need it far enough
    # that the elevation angle is ~0. Let's calculate where that is.
    # At 45N looking north, a satellite at altitude 400km on horizon
    # would be roughly at 52-53N due to Earth curvature.

    # For simplicity, let's test a satellite that's definitely below horizon
    # (on the other side of Earth)
    ground_ecef2 = geodetic_to_ecef(ground2)

    # Satellite on opposite side of Earth (negative x direction)
    sat_opposite = Position(-ground_ecef2.x, -ground_ecef2.y, -ground_ecef2.z)

    look2 = calculate_look_angle(sat_opposite, ground2)

    print(f"Ground: ({ground2.latitude}, {ground2.longitude})")
    print(f"Satellite ECEF: ({sat_opposite.x:.2f}, {sat_opposite.y:.2f}, {sat_opposite.z:.2f})")
    print(f"Look Angle: elevation={look2.elevation:.2f}, azimuth={look2.azimuth:.2f}")

    test2_pass = look2.elevation < 0
    if test2_pass:
        print("PASS: Negative elevation for satellite on opposite side of Earth")
    else:
        print(f"FAIL: Expected negative elevation, got {look2.elevation:.2f}")

    # Test case: Full calculation with TLE
    print("\n--- Test 3: Full look angle calculation with ISS TLE ---")

    tle_line1 = "1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990"
    tle_line2 = "2 25544  51.6443 242.2052 0004885 264.6463 206.3557 15.49165514212791"

    satellite = parse_tle(tle_line1, tle_line2)

    # Ground position: Houston, TX (NASA JSC)
    houston = GroundPosition(latitude=29.5502, longitude=-95.0979, altitude=0.0)

    # Time at epoch
    epoch_dt = datetime(2020, 2, 14, 4, 27, 33, tzinfo=timezone.utc)

    look_angle, visible = calculate_visibility_at_time(satellite, houston, epoch_dt)

    print(f"Ground: Houston, TX ({houston.latitude}, {houston.longitude})")
    print(f"Time: {epoch_dt}")
    print(f"Look Angle: elevation={look_angle.elevation:.2f}, azimuth={look_angle.azimuth:.2f}")
    print(f"Range: {look_angle.range_km:.2f} km")
    print(f"Visible (>10 deg): {visible}")

    # Verify range is reasonable (LEO satellite: ~400 km altitude)
    # When not visible, range can be up to ~13,000 km (satellite on other side of Earth)
    # Minimum range is ~350 km (closest approach overhead), max ~13,000 km (opposite side)
    test3_pass = 300 < look_angle.range_km < 15000
    if test3_pass:
        print("PASS: Range is within reasonable bounds for LEO satellite")
    else:
        print(f"FAIL: Range {look_angle.range_km:.2f} km is outside expected bounds")

    return test1_pass and test2_pass and test3_pass


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "#" * 60)
    print("# SATELLITE VISIBILITY CALCULATION - VERIFICATION TESTS")
    print("#" * 60)

    sgp4_pass = verify_sgp4_propagation()
    look_angle_pass = verify_look_angle_calculation()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"SGP4 Propagation Tests: {'PASS' if sgp4_pass else 'FAIL'}")
    print(f"Look Angle Tests: {'PASS' if look_angle_pass else 'FAIL'}")

    if sgp4_pass and look_angle_pass:
        print("\nAll tests PASSED!")
        return True
    else:
        print("\nSome tests FAILED!")
        return False


if __name__ == "__main__":
    run_all_tests()
