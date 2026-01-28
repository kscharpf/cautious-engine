"""
FastAPI backend for satellite coverage visualization.

Provides REST API endpoints for calculating satellite positions and coverage.
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict
from dataclasses import dataclass

from fastapi import FastAPI, Query, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import numpy as np

from satellite import (
    Satrec, parse_tle, propagate, datetime_to_gmst, teme_to_ecef,
    calculate_look_angle, geodetic_to_ecef, is_visible, GroundPosition,
    GroundGridCache, propagate_constellation_batch, calculate_visibility_numba,
    compute_coverage_map_optimized, NUMBA_AVAILABLE
)
from constellation import (
    Constellation, SAMPLE_STARLINK_TLES, ConstellationConfig,
    calculate_constellation_coverage_at_point
)
from propagation import (
    find_visibility_windows, PropagationConfig, calculate_satellite_geometry
)
from tle_loader import (
    parse_tle_text, fetch_tle_from_url, fetch_celestrak_preset,
    get_celestrak_presets, TLEParseError, TLEFetchError
)

app = FastAPI(title="Satellite Coverage Visualizer")

# Session storage for user constellations
# In production, use Redis or a database
SESSION_STORE: Dict[str, dict] = {}


def get_default_constellation() -> Constellation:
    """Create the default constellation with ISS and Starlink samples."""
    constellation = Constellation("Default Constellation")
    constellation.add_satellite(
        "ISS (ZARYA)",
        "1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990",
        "2 25544  51.6443 242.2052 0004885 264.6463 206.3557 15.49165514212791"
    )
    constellation.add_from_tle_list(SAMPLE_STARLINK_TLES)
    return constellation


# Default constellation (for backwards compatibility)
DEFAULT_CONSTELLATION = get_default_constellation()


def get_constellation(session_id: Optional[str] = None) -> Constellation:
    """Get constellation for a session, or the default."""
    if session_id and session_id in SESSION_STORE:
        return SESSION_STORE[session_id]['constellation']
    return DEFAULT_CONSTELLATION


# Pydantic models for request/response
class LoadTLERequest(BaseModel):
    tle_text: Optional[str] = None
    celestrak_preset: Optional[str] = None
    tle_url: Optional[str] = None


class CoverageRequest(BaseModel):
    latitude: float
    longitude: float
    altitude_km: float = 0.0
    duration_hours: float = 24.0
    min_elevation: float = 10.0


class SessionInfo(BaseModel):
    session_id: str
    source: str
    satellite_count: int
    satellites: List[str]


@app.get("/")
async def root():
    """Serve the main visualization page."""
    return HTMLResponse(content=get_html_page(), status_code=200)


@app.get("/api/satellites")
async def get_satellites(session_id: Optional[str] = None):
    """Get list of satellites in the constellation."""
    constellation = get_constellation(session_id)
    return {
        "satellites": [name for name, _ in constellation.satellites],
        "count": constellation.size,
        "session_id": session_id
    }


@app.get("/api/positions")
async def get_satellite_positions(
    timestamp: Optional[str] = None,
    offset_minutes: int = 0,
    session_id: Optional[str] = None
):
    """Get current positions of all satellites."""
    constellation = get_constellation(session_id)

    if timestamp:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    else:
        dt = datetime(2020, 2, 14, 12, 0, 0, tzinfo=timezone.utc)

    dt = dt + timedelta(minutes=offset_minutes)
    gmst = datetime_to_gmst(dt)

    positions = []
    for name, sat in constellation.satellites:
        try:
            pos, vel = propagate(sat, dt)
            sat_ecef = teme_to_ecef(pos, gmst)

            r = np.sqrt(sat_ecef.x**2 + sat_ecef.y**2 + sat_ecef.z**2)
            lat = np.arcsin(sat_ecef.z / r) * 180 / np.pi
            lon = np.arctan2(sat_ecef.y, sat_ecef.x) * 180 / np.pi
            alt = r - 6371

            positions.append({
                "name": name,
                "latitude": float(lat),
                "longitude": float(lon),
                "altitude_km": float(alt),
                "velocity_km_s": float(np.sqrt(vel.vx**2 + vel.vy**2 + vel.vz**2))
            })
        except ValueError:
            continue

    return {
        "timestamp": dt.isoformat(),
        "positions": positions,
        "session_id": session_id
    }


@app.get("/api/coverage/instant")
async def get_instant_coverage(
    timestamp: Optional[str] = None,
    offset_minutes: int = 0,
    resolution: float = 5.0,
    min_elevation: float = 10.0,
    session_id: Optional[str] = None
):
    """Get coverage map for a single instant."""
    constellation = get_constellation(session_id)

    if timestamp:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    else:
        dt = datetime(2020, 2, 14, 12, 0, 0, tzinfo=timezone.utc)

    dt = dt + timedelta(minutes=offset_minutes)
    gmst = datetime_to_gmst(dt)

    lats = np.arange(-90, 90, resolution)
    lons = np.arange(-180, 180, resolution)

    sat_positions = []
    for name, sat in constellation.satellites:
        try:
            pos, _ = propagate(sat, dt)
            sat_ecef = teme_to_ecef(pos, gmst)
            sat_positions.append(sat_ecef)
        except ValueError:
            continue

    coverage = []
    for lat in lats:
        row = []
        for lon in lons:
            ground = GroundPosition(float(lat), float(lon), 0.0)
            visible = False
            for sat_ecef in sat_positions:
                look = calculate_look_angle(sat_ecef, ground)
                if is_visible(look, min_elevation):
                    visible = True
                    break
            row.append(1 if visible else 0)
        coverage.append(row)

    return {
        "timestamp": dt.isoformat(),
        "resolution": resolution,
        "latitudes": lats.tolist(),
        "longitudes": lons.tolist(),
        "coverage": coverage,
        "session_id": session_id
    }


@app.get("/api/coverage/point")
async def get_point_coverage(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    timestamp: Optional[str] = None,
    duration_hours: float = 24.0,
    step_minutes: float = 1.0,
    min_elevation: float = 10.0,
    session_id: Optional[str] = None
):
    """Get time-series coverage for a specific point."""
    constellation = get_constellation(session_id)

    if timestamp:
        start = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    else:
        start = datetime(2020, 2, 14, 0, 0, 0, tzinfo=timezone.utc)

    ground = GroundPosition(latitude, longitude, 0.0)
    end = start + timedelta(hours=duration_hours)

    timeline = []
    current = start
    visible_count = 0
    total_count = 0

    while current < end:
        gmst = datetime_to_gmst(current)

        any_visible = False
        visible_sats = []

        for name, sat in constellation.satellites:
            try:
                pos, _ = propagate(sat, current)
                sat_ecef = teme_to_ecef(pos, gmst)
                look = calculate_look_angle(sat_ecef, ground)
                if is_visible(look, min_elevation):
                    any_visible = True
                    visible_sats.append({
                        "name": name,
                        "elevation": round(look.elevation, 1),
                        "azimuth": round(look.azimuth, 1)
                    })
            except ValueError:
                continue

        timeline.append({
            "time": current.isoformat(),
            "visible": any_visible,
            "satellites": visible_sats
        })

        if any_visible:
            visible_count += 1
        total_count += 1

        current += timedelta(minutes=step_minutes)

    coverage_percent = (visible_count / total_count * 100) if total_count > 0 else 0

    return {
        "latitude": latitude,
        "longitude": longitude,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "coverage_percent": round(coverage_percent, 2),
        "timeline": timeline[:100],
        "session_id": session_id
    }


# ============================================================================
# COVERAGE MAP WITH PERCENTAGE GRADATIONS
# ============================================================================

@app.get("/api/coverage/map")
async def get_coverage_map(
    timestamp: Optional[str] = None,
    offset_minutes: int = 0,
    duration_hours: float = 24.0,
    step_minutes: float = 5.0,
    min_elevation: float = 10.0,
    session_id: Optional[str] = None,
    parallel_workers: int = 4
):
    """
    Calculate coverage percentage map over a time period.

    Returns coverage percentages (0-100) for each 1° lat/lon grid cell.
    Also returns suggested gradation breaks based on data distribution.

    Uses high-performance optimizations:
    - Precomputed ground grid cache
    - Vectorized multi-satellite visibility
    - Numba JIT compilation (if available)
    - Parallel time step processing
    """
    import time as time_module
    calc_start = time_module.perf_counter()

    constellation = get_constellation(session_id)

    if timestamp:
        start_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    else:
        start_dt = datetime(2020, 2, 14, 0, 0, 0, tzinfo=timezone.utc)

    start_dt = start_dt + timedelta(minutes=offset_minutes)
    end_dt = start_dt + timedelta(hours=duration_hours)

    # Create 1° resolution grid
    lats = np.arange(-90, 90, 1.0)
    lons = np.arange(-180, 180, 1.0)

    # Use optimized coverage calculation
    visible_counts, total_steps = compute_coverage_map_optimized(
        satellites=constellation.satellites,
        start_time=start_dt,
        end_time=end_dt,
        step_minutes=step_minutes,
        lats=lats,
        lons=lons,
        min_elevation=min_elevation,
        use_numba=NUMBA_AVAILABLE,
        n_workers=parallel_workers
    )

    # Convert counts to percentages
    if total_steps > 0:
        coverage_pct = (visible_counts / total_steps) * 100
    else:
        coverage_pct = np.zeros_like(visible_counts, dtype=np.float32)

    # Calculate statistics for dynamic gradation
    flat_coverage = coverage_pct.flatten()
    non_zero = flat_coverage[flat_coverage > 0]

    # Determine gradation breaks
    if len(non_zero) == 0:
        gradations = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    else:
        min_cov = float(np.min(non_zero))
        max_cov = float(np.max(non_zero))
        range_cov = max_cov - min_cov

        if range_cov < 20:
            percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            gradations = [float(np.percentile(non_zero, p)) for p in percentiles]
            gradations = sorted(list(set([round(g, 1) for g in gradations])))
        else:
            gradations = list(range(0, 101, 10))

    # Calculate distribution for legend
    distribution = {}
    for i in range(len(gradations) - 1):
        low, high = gradations[i], gradations[i + 1]
        count = np.sum((coverage_pct >= low) & (coverage_pct < high))
        distribution[f"{low:.0f}-{high:.0f}"] = int(count)

    calc_elapsed = time_module.perf_counter() - calc_start

    return {
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "duration_hours": duration_hours,
        "step_minutes": step_minutes,
        "total_steps": total_steps,
        "latitudes": lats.tolist(),
        "longitudes": lons.tolist(),
        "coverage": coverage_pct.round(1).tolist(),
        "gradations": gradations,
        "distribution": distribution,
        "stats": {
            "min": float(np.min(coverage_pct)),
            "max": float(np.max(coverage_pct)),
            "mean": float(np.mean(coverage_pct)),
            "median": float(np.median(coverage_pct)),
            "non_zero_min": float(np.min(non_zero)) if len(non_zero) > 0 else 0,
            "non_zero_count": int(np.sum(coverage_pct > 0))
        },
        "performance": {
            "calculation_time_seconds": round(calc_elapsed, 2),
            "numba_enabled": NUMBA_AVAILABLE,
            "parallel_workers": parallel_workers
        },
        "session_id": session_id
    }


# ============================================================================
# PASS SCHEDULE
# ============================================================================

@app.get("/api/passes")
async def get_pass_schedule(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    altitude_km: float = 0.0,
    duration_hours: float = 24.0,
    min_elevation: float = 10.0,
    session_id: Optional[str] = None
):
    """
    Get pass schedule for all satellites over a ground location.

    Returns a list of satellite passes sorted by start time, including:
    - Satellite name
    - AOS (Acquisition of Signal) time and azimuth
    - LOS (Loss of Signal) time and azimuth
    - TCA (Time of Closest Approach) with max elevation
    - Pass duration
    """
    constellation = get_constellation(session_id)
    ground = GroundPosition(latitude, longitude, altitude_km)

    start_time = datetime.now(timezone.utc)
    end_time = start_time + timedelta(hours=duration_hours)

    config = PropagationConfig(min_elevation=min_elevation)

    all_passes = []

    for sat_name, sat in constellation.satellites:
        try:
            windows = find_visibility_windows(sat, ground, start_time, end_time, config)

            for window in windows:
                # Get AOS (start) geometry
                aos_look = calculate_satellite_geometry(sat, ground, window.start)

                # Get LOS (end) geometry
                los_look = calculate_satellite_geometry(sat, ground, window.end)

                # Find TCA (max elevation time) - simple approach: midpoint refined
                # For more accuracy, we could do a binary search
                duration_sec = (window.end - window.start).total_seconds()
                tca_time = window.start + timedelta(seconds=duration_sec / 2)

                # Refine TCA by checking a few points around midpoint
                best_elev = window.max_elevation
                best_time = tca_time
                for offset in [-duration_sec/4, 0, duration_sec/4]:
                    check_time = window.start + timedelta(seconds=duration_sec/2 + offset)
                    if window.start <= check_time <= window.end:
                        check_look = calculate_satellite_geometry(sat, ground, check_time)
                        if check_look.elevation > best_elev:
                            best_elev = check_look.elevation
                            best_time = check_time

                tca_look = calculate_satellite_geometry(sat, ground, best_time)

                all_passes.append({
                    "satellite": sat_name,
                    "aos": {
                        "time": window.start.isoformat(),
                        "azimuth": round(aos_look.azimuth, 1),
                        "azimuth_compass": azimuth_to_compass(aos_look.azimuth)
                    },
                    "tca": {
                        "time": best_time.isoformat(),
                        "elevation": round(best_elev, 1),
                        "azimuth": round(tca_look.azimuth, 1)
                    },
                    "los": {
                        "time": window.end.isoformat(),
                        "azimuth": round(los_look.azimuth, 1),
                        "azimuth_compass": azimuth_to_compass(los_look.azimuth)
                    },
                    "duration_seconds": round(duration_sec),
                    "duration_formatted": format_duration(duration_sec),
                    "max_elevation": round(window.max_elevation, 1)
                })

        except ValueError:
            # Propagation error for this satellite
            continue

    # Sort by AOS time
    all_passes.sort(key=lambda p: p["aos"]["time"])

    return {
        "latitude": latitude,
        "longitude": longitude,
        "altitude_km": altitude_km,
        "start": start_time.isoformat(),
        "end": end_time.isoformat(),
        "duration_hours": duration_hours,
        "min_elevation": min_elevation,
        "total_passes": len(all_passes),
        "passes": all_passes,
        "session_id": session_id
    }


def azimuth_to_compass(azimuth: float) -> str:
    """Convert azimuth angle to compass direction."""
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    index = round(azimuth / 22.5) % 16
    return directions[index]


def format_duration(seconds: float) -> str:
    """Format duration in seconds to MM:SS string."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


# ============================================================================
# NEW ENDPOINTS: TLE Loading and Session Management
# ============================================================================

@app.get("/api/celestrak/presets")
async def get_celestrak_presets_endpoint():
    """List available CelesTrak presets."""
    presets = get_celestrak_presets()
    return {
        "presets": [
            {"name": name, "url": url}
            for name, url in sorted(presets.items())
        ]
    }


@app.post("/api/constellation/load")
async def load_constellation(request: LoadTLERequest):
    """
    Load a constellation from TLE text, CelesTrak preset, or URL.

    Returns a session ID to use for subsequent requests.
    """
    try:
        # Determine source and load TLEs
        if request.tle_text:
            tles = parse_tle_text(request.tle_text)
            source = "text"
        elif request.celestrak_preset:
            tles = await fetch_celestrak_preset(request.celestrak_preset)
            source = f"celestrak:{request.celestrak_preset}"
        elif request.tle_url:
            text = await fetch_tle_from_url(request.tle_url)
            tles = parse_tle_text(text)
            source = f"url:{request.tle_url}"
        else:
            raise HTTPException(status_code=400, detail="No TLE source provided")

        if not tles:
            raise HTTPException(status_code=400, detail="No valid TLEs found")

        # Create constellation
        constellation = Constellation(f"User Constellation ({source})")
        constellation.add_from_tle_list(tles)

        # Create session
        session_id = str(uuid.uuid4())
        SESSION_STORE[session_id] = {
            'constellation': constellation,
            'source': source,
            'created_at': datetime.now(timezone.utc).isoformat(),
        }

        return {
            "session_id": session_id,
            "source": source,
            "satellite_count": constellation.size,
            "satellites": [name for name, _ in constellation.satellites[:50]],
            "truncated": constellation.size > 50
        }

    except TLEParseError as e:
        raise HTTPException(status_code=400, detail=f"TLE parse error: {e}")
    except TLEFetchError as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch TLEs: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/constellation/upload")
async def upload_constellation(file: UploadFile = File(...)):
    """
    Upload a TLE file and create a session.

    Accepts .txt or .tle files.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Check file extension
    if not file.filename.lower().endswith(('.txt', '.tle')):
        raise HTTPException(status_code=400, detail="File must be .txt or .tle")

    try:
        content = await file.read()
        text = content.decode('utf-8')

        tles = parse_tle_text(text)

        if not tles:
            raise HTTPException(status_code=400, detail="No valid TLEs found in file")

        # Create constellation
        constellation = Constellation(f"Uploaded: {file.filename}")
        constellation.add_from_tle_list(tles)

        # Create session
        session_id = str(uuid.uuid4())
        SESSION_STORE[session_id] = {
            'constellation': constellation,
            'source': f"upload:{file.filename}",
            'created_at': datetime.now(timezone.utc).isoformat(),
        }

        return {
            "session_id": session_id,
            "source": f"upload:{file.filename}",
            "satellite_count": constellation.size,
            "satellites": [name for name, _ in constellation.satellites[:50]],
            "truncated": constellation.size > 50
        }

    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text")
    except TLEParseError as e:
        raise HTTPException(status_code=400, detail=f"TLE parse error: {e}")


@app.get("/api/constellation/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a session's constellation."""
    if session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found")

    session = SESSION_STORE[session_id]
    constellation = session['constellation']

    return {
        "session_id": session_id,
        "source": session['source'],
        "created_at": session['created_at'],
        "satellite_count": constellation.size,
        "satellites": [name for name, _ in constellation.satellites[:50]],
        "truncated": constellation.size > 50
    }


@app.post("/api/coverage/calculate/{session_id}")
async def calculate_session_coverage(session_id: str, request: CoverageRequest):
    """Calculate coverage for a session's constellation at a specific point."""
    if session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found")

    constellation = SESSION_STORE[session_id]['constellation']
    ground = GroundPosition(request.latitude, request.longitude, request.altitude_km)
    start_time = datetime.now(timezone.utc)

    config = ConstellationConfig(
        min_elevation=request.min_elevation,
        time_step_seconds=60.0,
    )

    coverage_percent = calculate_constellation_coverage_at_point(
        constellation,
        ground,
        start_time,
        duration_days=request.duration_hours / 24.0,
        config=config,
    )

    return {
        "session_id": session_id,
        "latitude": request.latitude,
        "longitude": request.longitude,
        "altitude_km": request.altitude_km,
        "duration_hours": request.duration_hours,
        "min_elevation": request.min_elevation,
        "coverage_percent": round(coverage_percent, 2),
        "satellite_count": constellation.size,
        "start_time": start_time.isoformat()
    }


@app.delete("/api/constellation/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found")

    del SESSION_STORE[session_id]
    return {"message": "Session deleted", "session_id": session_id}


def get_html_page():
    """Generate the HTML page for visualization."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Satellite Coverage Visualizer</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
        }
        .container {
            display: flex;
            height: 100vh;
        }
        .sidebar {
            width: 360px;
            background: #16213e;
            padding: 20px;
            overflow-y: auto;
        }
        .map-container {
            flex: 1;
            position: relative;
        }
        #map {
            width: 100%;
            height: 100%;
        }
        h1 {
            font-size: 1.5rem;
            margin-bottom: 20px;
            color: #00d9ff;
        }
        h2 {
            font-size: 1rem;
            margin: 20px 0 10px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .control-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-size: 0.9rem;
        }
        input[type="range"], input[type="number"], input[type="text"], select, textarea {
            width: 100%;
            background: #0f3460;
            border: 1px solid #1a1a2e;
            padding: 8px;
            border-radius: 4px;
            color: #fff;
            font-size: 0.9rem;
        }
        textarea {
            min-height: 100px;
            resize: vertical;
            font-family: monospace;
        }
        input[type="range"] {
            padding: 0;
            height: 8px;
            -webkit-appearance: none;
            border-radius: 4px;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            background: #00d9ff;
            border-radius: 50%;
            cursor: pointer;
        }
        select {
            cursor: pointer;
        }
        button {
            width: 100%;
            padding: 12px;
            background: #00d9ff;
            border: none;
            border-radius: 4px;
            color: #1a1a2e;
            font-weight: bold;
            cursor: pointer;
            margin-top: 10px;
        }
        button:hover {
            background: #00b8d4;
        }
        button:disabled {
            background: #555;
            cursor: not-allowed;
        }
        button.secondary {
            background: #0f3460;
            color: #00d9ff;
            border: 1px solid #00d9ff;
        }
        button.secondary:hover {
            background: #1a4a70;
        }
        .stats {
            background: #0f3460;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }
        .stat-value {
            color: #00d9ff;
            font-weight: bold;
        }
        .satellite-list {
            max-height: 150px;
            overflow-y: auto;
            background: #0f3460;
            border-radius: 8px;
            padding: 10px;
        }
        .satellite-item {
            padding: 6px;
            border-bottom: 1px solid #1a1a2e;
            font-size: 0.85rem;
        }
        .satellite-item:last-child {
            border-bottom: none;
        }
        .satellite-name {
            color: #00d9ff;
        }
        .legend {
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: rgba(22, 33, 62, 0.9);
            padding: 15px;
            border-radius: 8px;
            z-index: 1000;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin-bottom: 5px;
        }
        .legend-color {
            width: 20px;
            height: 20px;
            margin-right: 10px;
            border-radius: 4px;
        }
        .loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(22, 33, 62, 0.95);
            padding: 20px 40px;
            border-radius: 8px;
            z-index: 2000;
        }
        .click-info {
            background: #0f3460;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
        }
        .tabs {
            display: flex;
            gap: 5px;
            margin-bottom: 10px;
        }
        .tab {
            flex: 1;
            padding: 8px;
            background: #0f3460;
            border: none;
            border-radius: 4px 4px 0 0;
            color: #888;
            cursor: pointer;
            font-size: 0.85rem;
        }
        .tab.active {
            background: #1a4a70;
            color: #00d9ff;
        }
        .tab-content {
            display: none;
            background: #0f3460;
            padding: 15px;
            border-radius: 0 0 8px 8px;
        }
        .tab-content.active {
            display: block;
        }
        .session-info {
            background: #0a2a4e;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            font-size: 0.85rem;
        }
        .session-info.active {
            border-left: 3px solid #00d9ff;
        }
        .status-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.75rem;
            margin-left: 5px;
        }
        .status-badge.default {
            background: #444;
        }
        .status-badge.custom {
            background: #00d9ff;
            color: #1a1a2e;
        }
        .file-input-wrapper {
            position: relative;
            overflow: hidden;
            display: inline-block;
            width: 100%;
        }
        .file-input-wrapper input[type=file] {
            position: absolute;
            left: 0;
            top: 0;
            opacity: 0;
            cursor: pointer;
            width: 100%;
            height: 100%;
        }
        .file-input-btn {
            display: block;
            width: 100%;
            padding: 12px;
            background: #0f3460;
            border: 2px dashed #00d9ff;
            border-radius: 4px;
            color: #00d9ff;
            text-align: center;
            cursor: pointer;
        }
        .file-input-btn:hover {
            background: #1a4a70;
        }
        .error-message {
            color: #ff6b6b;
            font-size: 0.85rem;
            margin-top: 10px;
        }
        .success-message {
            color: #51cf66;
            font-size: 0.85rem;
            margin-top: 10px;
        }
        .pass-list {
            max-height: 400px;
            overflow-y: auto;
            background: #0f3460;
            border-radius: 8px;
            padding: 10px;
        }
        .pass-item {
            background: #0a2a4e;
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 10px;
            border-left: 3px solid #00d9ff;
        }
        .pass-item:last-child {
            margin-bottom: 0;
        }
        .pass-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .pass-satellite {
            font-weight: bold;
            color: #00d9ff;
            font-size: 0.95rem;
        }
        .pass-elevation {
            background: #00d9ff;
            color: #1a1a2e;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        .pass-elevation.high {
            background: #51cf66;
        }
        .pass-elevation.medium {
            background: #fcc419;
        }
        .pass-elevation.low {
            background: #ff6b6b;
        }
        .pass-details {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            font-size: 0.8rem;
        }
        .pass-detail {
            text-align: center;
        }
        .pass-detail-label {
            color: #888;
            font-size: 0.7rem;
            text-transform: uppercase;
        }
        .pass-detail-value {
            color: #fff;
            font-weight: bold;
        }
        .pass-detail-sub {
            color: #666;
            font-size: 0.7rem;
        }
        .pass-summary {
            background: #0a2a4e;
            padding: 10px;
            border-radius: 6px;
            margin-bottom: 10px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h1>Satellite Coverage</h1>

            <h2>TLE Source</h2>
            <div class="tabs">
                <button class="tab active" data-tab="paste">Paste</button>
                <button class="tab" data-tab="upload">Upload</button>
                <button class="tab" data-tab="celestrak">CelesTrak</button>
            </div>

            <div class="tab-content active" id="tab-paste">
                <textarea id="tle-text" placeholder="Paste TLE data here...
ISS (ZARYA)
1 25544U 98067A...
2 25544 51.6443..."></textarea>
                <button id="load-tle-btn">Load TLEs</button>
            </div>

            <div class="tab-content" id="tab-upload">
                <div class="file-input-wrapper">
                    <div class="file-input-btn" id="file-label">Choose .txt or .tle file</div>
                    <input type="file" id="tle-file" accept=".txt,.tle">
                </div>
                <button id="upload-tle-btn" disabled>Upload TLEs</button>
            </div>

            <div class="tab-content" id="tab-celestrak">
                <label>Select Constellation:</label>
                <select id="celestrak-select">
                    <option value="">-- Select --</option>
                </select>
                <button id="fetch-celestrak-btn">Fetch from CelesTrak</button>
            </div>

            <div id="tle-status"></div>

            <div class="session-info" id="session-info">
                <strong>Source:</strong> <span id="session-source">Default</span>
                <span class="status-badge default" id="session-badge">Default</span><br>
                <strong>Satellites:</strong> <span id="session-count">-</span>
            </div>

            <h2>Time Controls</h2>
            <div class="control-group">
                <label>Time Offset: <span id="offset-value">0</span> minutes</label>
                <input type="range" id="time-offset" min="-720" max="720" value="0">
            </div>

            <div class="control-group">
                <button id="play-btn">Play Animation</button>
            </div>

            <h2>Coverage Map Settings</h2>
            <div class="control-group">
                <label>Min Elevation: <span id="elev-value">10</span>&deg;</label>
                <input type="range" id="min-elevation" min="0" max="45" value="10">
            </div>

            <div class="control-group">
                <label>Coverage Duration: <span id="duration-value">24</span>h</label>
                <input type="range" id="coverage-duration" min="1" max="72" value="24">
            </div>

            <div class="control-group">
                <label>Time Step: <span id="step-value">5</span> min</label>
                <input type="range" id="time-step" min="1" max="30" value="5">
            </div>

            <button id="refresh-btn">Calculate Coverage Map</button>
            <p style="font-size:0.75rem;color:#666;margin-top:5px;">1&deg; resolution, may take a moment</p>

            <h2>Point Analysis</h2>
            <div class="click-info" id="coverage-calc">
                <label>Latitude:</label>
                <input type="number" id="calc-lat" step="0.01" placeholder="Click map or enter">
                <label style="margin-top:8px;">Longitude:</label>
                <input type="number" id="calc-lon" step="0.01" placeholder="Click map or enter">
                <label style="margin-top:8px;">Duration (hours):</label>
                <input type="number" id="calc-duration" value="24" min="1" max="168">
                <div style="display:flex;gap:10px;margin-top:10px;">
                    <button id="calc-coverage-btn" style="flex:1;">Coverage</button>
                    <button id="calc-passes-btn" style="flex:1;">Passes</button>
                </div>
                <div id="coverage-result"></div>
            </div>

            <div id="pass-schedule" style="display:none;">
                <h2>Pass Schedule</h2>
                <div class="pass-list" id="pass-list">
                    <!-- Populated by JS -->
                </div>
            </div>

            <h2>Satellites</h2>
            <div class="satellite-list" id="satellite-list">
                Loading...
            </div>

            <h2>Statistics</h2>
            <div class="stats" id="stats">
                <div class="stat-row">
                    <span>Satellites:</span>
                    <span class="stat-value" id="stat-sats">-</span>
                </div>
                <div class="stat-row">
                    <span>Grid Cells:</span>
                    <span class="stat-value" id="stat-cells">-</span>
                </div>
                <div class="stat-row">
                    <span>Avg Coverage:</span>
                    <span class="stat-value" id="stat-avg">-</span>
                </div>
                <div class="stat-row">
                    <span>Min/Max:</span>
                    <span class="stat-value" id="stat-minmax">-</span>
                </div>
            </div>

            <button id="reset-btn" class="secondary" style="margin-top:20px;">Reset to Default</button>
        </div>

        <div class="map-container">
            <div id="map"></div>
            <div class="legend" id="legend">
                <div style="font-weight:bold;margin-bottom:10px;">Coverage %</div>
                <div id="legend-gradient" style="display:flex;flex-direction:column;gap:2px;">
                    <!-- Populated by JS -->
                </div>
                <div class="legend-item" style="margin-top:10px;border-top:1px solid #444;padding-top:8px;">
                    <div class="legend-color" style="background: #ff4444;"></div>
                    <span>Satellite</span>
                </div>
            </div>
            <div class="loading" id="loading" style="display: none;">
                Loading coverage data...
            </div>
        </div>
    </div>

    <script>
        // State
        let currentSessionId = null;
        let map;
        let coverageLayer = null;
        let satelliteMarkers = [];
        let animationInterval = null;
        let isPlaying = false;
        let selectedMarker = null;
        let currentGradations = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

        // Color scale for coverage percentages (from red/orange to green/cyan)
        const COVERAGE_COLORS = [
            { pct: 0,   color: { r: 40,  g: 40,  b: 40  } },  // Dark gray (no coverage)
            { pct: 10,  color: { r: 139, g: 0,   b: 0   } },  // Dark red
            { pct: 20,  color: { r: 178, g: 34,  b: 34  } },  // Firebrick
            { pct: 30,  color: { r: 255, g: 69,  b: 0   } },  // Orange red
            { pct: 40,  color: { r: 255, g: 140, b: 0   } },  // Dark orange
            { pct: 50,  color: { r: 255, g: 215, b: 0   } },  // Gold
            { pct: 60,  color: { r: 154, g: 205, b: 50  } },  // Yellow green
            { pct: 70,  color: { r: 50,  g: 205, b: 50  } },  // Lime green
            { pct: 80,  color: { r: 0,   g: 206, b: 209 } },  // Dark turquoise
            { pct: 90,  color: { r: 0,   g: 191, b: 255 } },  // Deep sky blue
            { pct: 100, color: { r: 0,   g: 217, b: 255 } },  // Cyan
        ];

        function getColorForPercent(pct, gradations) {
            // Find which gradation bucket this percentage falls into
            let bucket = 0;
            for (let i = 0; i < gradations.length - 1; i++) {
                if (pct >= gradations[i] && pct < gradations[i + 1]) {
                    bucket = i;
                    break;
                }
                if (pct >= gradations[gradations.length - 1]) {
                    bucket = gradations.length - 2;
                }
            }

            // Map bucket to color scale
            const colorIdx = Math.floor((bucket / (gradations.length - 1)) * (COVERAGE_COLORS.length - 1));
            const c = COVERAGE_COLORS[Math.min(colorIdx, COVERAGE_COLORS.length - 1)].color;
            return `rgba(${c.r}, ${c.g}, ${c.b}, 0.7)`;
        }

        function updateLegend(gradations, stats) {
            const legend = document.getElementById('legend-gradient');
            legend.innerHTML = '';

            // Create legend items from high to low
            for (let i = gradations.length - 2; i >= 0; i--) {
                const low = gradations[i];
                const high = gradations[i + 1];
                const midPct = (low + high) / 2;
                const color = getColorForPercent(midPct, gradations);

                const item = document.createElement('div');
                item.className = 'legend-item';
                item.innerHTML = `
                    <div class="legend-color" style="background: ${color};"></div>
                    <span>${low.toFixed(0)}-${high.toFixed(0)}%</span>
                `;
                legend.appendChild(item);
            }
        }

        // Initialize map
        function initMap() {
            map = L.map('map').setView([20, 0], 2);
            L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
                attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
                maxZoom: 19
            }).addTo(map);

            // Map click handler for coverage calculator
            map.on('click', (e) => {
                const { lat, lng } = e.latlng;
                document.getElementById('calc-lat').value = lat.toFixed(4);
                document.getElementById('calc-lon').value = lng.toFixed(4);

                // Add/move marker
                if (selectedMarker) {
                    selectedMarker.setLatLng(e.latlng);
                } else {
                    selectedMarker = L.marker(e.latlng, {
                        icon: L.divIcon({
                            className: 'selected-point',
                            html: '<div style="background:#00d9ff;width:12px;height:12px;border-radius:50%;border:2px solid white;"></div>',
                            iconSize: [16, 16],
                            iconAnchor: [8, 8]
                        })
                    }).addTo(map);
                }
            });

            // Initialize legend with default gradations
            updateLegend(currentGradations, null);
        }

        // Tab switching
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                tab.classList.add('active');
                document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
            });
        });

        // Load CelesTrak presets
        async function loadCelestrakPresets() {
            try {
                const response = await fetch('/api/celestrak/presets');
                const data = await response.json();
                const select = document.getElementById('celestrak-select');
                data.presets.forEach(preset => {
                    const option = document.createElement('option');
                    option.value = preset.name;
                    option.textContent = preset.name.charAt(0).toUpperCase() + preset.name.slice(1);
                    select.appendChild(option);
                });
            } catch (error) {
                console.error('Failed to load presets:', error);
            }
        }

        // Update session info display
        function updateSessionInfo(source, count, isCustom = false) {
            document.getElementById('session-source').textContent = source;
            document.getElementById('session-count').textContent = count;
            const badge = document.getElementById('session-badge');
            badge.textContent = isCustom ? 'Custom' : 'Default';
            badge.className = 'status-badge ' + (isCustom ? 'custom' : 'default');
        }

        // Show status message
        function showStatus(message, isError = false) {
            const status = document.getElementById('tle-status');
            status.innerHTML = `<div class="${isError ? 'error-message' : 'success-message'}">${message}</div>`;
            setTimeout(() => { status.innerHTML = ''; }, 5000);
        }

        // Load TLEs from text
        document.getElementById('load-tle-btn').addEventListener('click', async () => {
            const text = document.getElementById('tle-text').value.trim();
            if (!text) {
                showStatus('Please paste TLE data first', true);
                return;
            }

            const btn = document.getElementById('load-tle-btn');
            btn.disabled = true;
            btn.textContent = 'Loading...';

            try {
                const response = await fetch('/api/constellation/load', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ tle_text: text })
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to load TLEs');
                }

                const data = await response.json();
                currentSessionId = data.session_id;
                updateSessionInfo(data.source, data.satellite_count, true);
                showStatus(`Loaded ${data.satellite_count} satellites`);

                // Refresh display
                loadSatellites();
                updatePositions();
                updateCoverage();

            } catch (error) {
                showStatus(error.message, true);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Load TLEs';
            }
        });

        // File upload
        document.getElementById('tle-file').addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                document.getElementById('file-label').textContent = file.name;
                document.getElementById('upload-tle-btn').disabled = false;
            }
        });

        document.getElementById('upload-tle-btn').addEventListener('click', async () => {
            const fileInput = document.getElementById('tle-file');
            const file = fileInput.files[0];
            if (!file) return;

            const btn = document.getElementById('upload-tle-btn');
            btn.disabled = true;
            btn.textContent = 'Uploading...';

            try {
                const formData = new FormData();
                formData.append('file', file);

                const response = await fetch('/api/constellation/upload', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to upload TLEs');
                }

                const data = await response.json();
                currentSessionId = data.session_id;
                updateSessionInfo(data.source, data.satellite_count, true);
                showStatus(`Uploaded ${data.satellite_count} satellites`);

                // Refresh display
                loadSatellites();
                updatePositions();
                updateCoverage();

            } catch (error) {
                showStatus(error.message, true);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Upload TLEs';
            }
        });

        // CelesTrak fetch
        document.getElementById('fetch-celestrak-btn').addEventListener('click', async () => {
            const preset = document.getElementById('celestrak-select').value;
            if (!preset) {
                showStatus('Please select a constellation', true);
                return;
            }

            const btn = document.getElementById('fetch-celestrak-btn');
            btn.disabled = true;
            btn.textContent = 'Fetching...';

            try {
                const response = await fetch('/api/constellation/load', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ celestrak_preset: preset })
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to fetch TLEs');
                }

                const data = await response.json();
                currentSessionId = data.session_id;
                updateSessionInfo(data.source, data.satellite_count, true);
                showStatus(`Fetched ${data.satellite_count} satellites from CelesTrak`);

                // Refresh display
                loadSatellites();
                updatePositions();
                updateCoverage();

            } catch (error) {
                showStatus(error.message, true);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Fetch from CelesTrak';
            }
        });

        // Reset to default
        document.getElementById('reset-btn').addEventListener('click', () => {
            currentSessionId = null;
            updateSessionInfo('Default', '-', false);
            loadSatellites();
            updatePositions();
            updateCoverage();
            showStatus('Reset to default constellation');
        });

        // Calculate coverage for selected point
        document.getElementById('calc-coverage-btn').addEventListener('click', async () => {
            const lat = parseFloat(document.getElementById('calc-lat').value);
            const lon = parseFloat(document.getElementById('calc-lon').value);
            const duration = parseFloat(document.getElementById('calc-duration').value);

            if (isNaN(lat) || isNaN(lon)) {
                document.getElementById('coverage-result').innerHTML =
                    '<div class="error-message">Please enter valid coordinates or click on the map</div>';
                return;
            }

            const btn = document.getElementById('calc-coverage-btn');
            btn.disabled = true;
            btn.textContent = 'Calculating...';
            document.getElementById('coverage-result').innerHTML = 'Calculating coverage...';

            try {
                let response;
                if (currentSessionId) {
                    response = await fetch(`/api/coverage/calculate/${currentSessionId}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            latitude: lat,
                            longitude: lon,
                            duration_hours: duration,
                            min_elevation: parseFloat(document.getElementById('min-elevation').value)
                        })
                    });
                } else {
                    response = await fetch(
                        `/api/coverage/point?latitude=${lat}&longitude=${lon}&duration_hours=${duration}&min_elevation=${document.getElementById('min-elevation').value}`
                    );
                }

                if (!response.ok) throw new Error('Calculation failed');

                const data = await response.json();
                document.getElementById('coverage-result').innerHTML = `
                    <div style="margin-top:10px;padding:10px;background:#0a2a4e;border-radius:4px;">
                        <strong>Coverage: ${data.coverage_percent}%</strong><br>
                        <small>Duration: ${duration}h | Satellites: ${data.satellite_count || data.timeline?.length || '-'}</small>
                    </div>
                `;
            } catch (error) {
                document.getElementById('coverage-result').innerHTML =
                    '<div class="error-message">Failed to calculate coverage</div>';
            } finally {
                btn.disabled = false;
                btn.textContent = 'Coverage';
            }
        });

        // Get pass schedule for selected point
        document.getElementById('calc-passes-btn').addEventListener('click', async () => {
            const lat = parseFloat(document.getElementById('calc-lat').value);
            const lon = parseFloat(document.getElementById('calc-lon').value);
            const duration = parseFloat(document.getElementById('calc-duration').value);
            const minElev = parseFloat(document.getElementById('min-elevation').value);

            if (isNaN(lat) || isNaN(lon)) {
                document.getElementById('coverage-result').innerHTML =
                    '<div class="error-message">Please enter valid coordinates or click on the map</div>';
                return;
            }

            const btn = document.getElementById('calc-passes-btn');
            btn.disabled = true;
            btn.textContent = 'Loading...';

            try {
                let url = `/api/passes?latitude=${lat}&longitude=${lon}&duration_hours=${duration}&min_elevation=${minElev}`;
                if (currentSessionId) {
                    url += `&session_id=${currentSessionId}`;
                }

                const response = await fetch(url);
                if (!response.ok) throw new Error('Failed to get passes');

                const data = await response.json();
                displayPasses(data);

            } catch (error) {
                document.getElementById('coverage-result').innerHTML =
                    '<div class="error-message">Failed to get pass schedule</div>';
                document.getElementById('pass-schedule').style.display = 'none';
            } finally {
                btn.disabled = false;
                btn.textContent = 'Passes';
            }
        });

        // Display pass schedule
        function displayPasses(data) {
            const container = document.getElementById('pass-schedule');
            const list = document.getElementById('pass-list');

            if (data.passes.length === 0) {
                document.getElementById('coverage-result').innerHTML =
                    '<div style="margin-top:10px;padding:10px;background:#0a2a4e;border-radius:4px;">No passes found in the next ' + data.duration_hours + ' hours</div>';
                container.style.display = 'none';
                return;
            }

            // Show summary in coverage result area
            document.getElementById('coverage-result').innerHTML = `
                <div style="margin-top:10px;padding:10px;background:#0a2a4e;border-radius:4px;">
                    <strong>${data.total_passes} passes</strong> in next ${data.duration_hours}h<br>
                    <small>Min elevation: ${data.min_elevation}&deg;</small>
                </div>
            `;

            // Build pass list HTML
            let html = '';

            data.passes.forEach((pass, idx) => {
                const aosTime = new Date(pass.aos.time);
                const losTime = new Date(pass.los.time);
                const tcaTime = new Date(pass.tca.time);

                // Determine elevation class
                let elevClass = 'low';
                if (pass.max_elevation >= 60) elevClass = 'high';
                else if (pass.max_elevation >= 30) elevClass = 'medium';

                html += `
                    <div class="pass-item">
                        <div class="pass-header">
                            <span class="pass-satellite">${pass.satellite}</span>
                            <span class="pass-elevation ${elevClass}">${pass.max_elevation}&deg; max</span>
                        </div>
                        <div class="pass-details">
                            <div class="pass-detail">
                                <div class="pass-detail-label">AOS</div>
                                <div class="pass-detail-value">${aosTime.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</div>
                                <div class="pass-detail-sub">${pass.aos.azimuth_compass} (${pass.aos.azimuth}&deg;)</div>
                            </div>
                            <div class="pass-detail">
                                <div class="pass-detail-label">TCA</div>
                                <div class="pass-detail-value">${tcaTime.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</div>
                                <div class="pass-detail-sub">${pass.tca.elevation}&deg; el</div>
                            </div>
                            <div class="pass-detail">
                                <div class="pass-detail-label">LOS</div>
                                <div class="pass-detail-value">${losTime.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</div>
                                <div class="pass-detail-sub">${pass.los.azimuth_compass} (${pass.los.azimuth}&deg;)</div>
                            </div>
                        </div>
                        <div style="text-align:center;margin-top:8px;font-size:0.75rem;color:#666;">
                            Duration: ${pass.duration_formatted} | ${aosTime.toLocaleDateString()}
                        </div>
                    </div>
                `;
            });

            list.innerHTML = html;
            container.style.display = 'block';
        }

        // Load satellites
        async function loadSatellites() {
            const url = currentSessionId
                ? `/api/satellites?session_id=${currentSessionId}`
                : '/api/satellites';

            try {
                const response = await fetch(url);
                const data = await response.json();

                const list = document.getElementById('satellite-list');
                list.innerHTML = data.satellites.slice(0, 50).map(name =>
                    `<div class="satellite-item"><span class="satellite-name">&#128752;</span> ${name}</div>`
                ).join('');

                if (data.count > 50) {
                    list.innerHTML += `<div class="satellite-item" style="color:#888;">... and ${data.count - 50} more</div>`;
                }

                document.getElementById('stat-sats').textContent = data.count;

                if (!currentSessionId) {
                    updateSessionInfo('Default', data.count, false);
                }
            } catch (error) {
                console.error('Failed to load satellites:', error);
            }
        }

        // Update satellite positions
        async function updatePositions() {
            const offset = document.getElementById('time-offset').value;
            const url = currentSessionId
                ? `/api/positions?offset_minutes=${offset}&session_id=${currentSessionId}`
                : `/api/positions?offset_minutes=${offset}`;

            try {
                const response = await fetch(url);
                const data = await response.json();

                // Clear existing markers
                satelliteMarkers.forEach(m => map.removeLayer(m));
                satelliteMarkers = [];

                // Add new markers
                data.positions.forEach(sat => {
                    const marker = L.circleMarker([sat.latitude, sat.longitude], {
                        radius: 8,
                        fillColor: '#ff4444',
                        color: '#fff',
                        weight: 2,
                        fillOpacity: 0.8
                    }).addTo(map);

                    marker.bindPopup(`
                        <b>${sat.name}</b><br>
                        Lat: ${sat.latitude.toFixed(2)}&deg;<br>
                        Lon: ${sat.longitude.toFixed(2)}&deg;<br>
                        Alt: ${sat.altitude_km.toFixed(0)} km<br>
                        Vel: ${sat.velocity_km_s.toFixed(2)} km/s
                    `);

                    satelliteMarkers.push(marker);
                });
            } catch (error) {
                console.error('Failed to update positions:', error);
            }
        }

        // Update coverage map with percentage gradations
        async function updateCoverage() {
            const loading = document.getElementById('loading');
            loading.style.display = 'block';
            loading.textContent = 'Calculating coverage map (this may take a moment)...';

            const offset = document.getElementById('time-offset').value;
            const duration = document.getElementById('coverage-duration').value;
            const stepMin = document.getElementById('time-step').value;
            const minElev = document.getElementById('min-elevation').value;

            let url = `/api/coverage/map?offset_minutes=${offset}&duration_hours=${duration}&step_minutes=${stepMin}&min_elevation=${minElev}`;
            if (currentSessionId) {
                url += `&session_id=${currentSessionId}`;
            }

            try {
                const response = await fetch(url);
                const data = await response.json();

                // Remove old coverage layer
                if (coverageLayer) {
                    map.removeLayer(coverageLayer);
                }

                // Update gradations from server response
                currentGradations = data.gradations;
                updateLegend(currentGradations, data.stats);

                // Create coverage polygons with gradient colors
                const polygons = [];
                const res = 1.0; // 1 degree resolution

                data.coverage.forEach((row, i) => {
                    row.forEach((pct, j) => {
                        if (pct > 0) {
                            const lat = data.latitudes[i];
                            const lon = data.longitudes[j];
                            const color = getColorForPercent(pct, currentGradations);

                            const bounds = [
                                [lat, lon],
                                [lat, lon + res],
                                [lat + res, lon + res],
                                [lat + res, lon]
                            ];

                            const poly = L.polygon(bounds, {
                                color: 'transparent',
                                fillColor: color,
                                fillOpacity: 0.7,
                                weight: 0
                            });

                            // Add popup with coverage info
                            poly.bindPopup(`
                                <b>Coverage: ${pct.toFixed(1)}%</b><br>
                                Lat: ${lat.toFixed(0)}&deg; to ${(lat+1).toFixed(0)}&deg;<br>
                                Lon: ${lon.toFixed(0)}&deg; to ${(lon+1).toFixed(0)}&deg;
                            `);

                            polygons.push(poly);
                        }
                    });
                });

                coverageLayer = L.layerGroup(polygons).addTo(map);

                // Update stats
                const totalCells = data.coverage.length * data.coverage[0].length;
                const nonZeroCells = data.stats.non_zero_count;

                document.getElementById('stat-cells').textContent = `${nonZeroCells.toLocaleString()} / ${totalCells.toLocaleString()}`;
                document.getElementById('stat-avg').textContent = data.stats.mean.toFixed(1) + '%';
                document.getElementById('stat-minmax').textContent = `${data.stats.min.toFixed(0)}% / ${data.stats.max.toFixed(0)}%`;

            } catch (error) {
                console.error('Error loading coverage:', error);
                loading.textContent = 'Error loading coverage data';
            }

            loading.style.display = 'none';
        }

        // Event listeners
        document.getElementById('time-offset').addEventListener('input', (e) => {
            document.getElementById('offset-value').textContent = e.target.value;
            updatePositions();
        });

        document.getElementById('min-elevation').addEventListener('input', (e) => {
            document.getElementById('elev-value').textContent = e.target.value;
        });

        document.getElementById('coverage-duration').addEventListener('input', (e) => {
            document.getElementById('duration-value').textContent = e.target.value;
        });

        document.getElementById('time-step').addEventListener('input', (e) => {
            document.getElementById('step-value').textContent = e.target.value;
        });

        document.getElementById('refresh-btn').addEventListener('click', () => {
            updateCoverage();
            updatePositions();
        });

        document.getElementById('play-btn').addEventListener('click', () => {
            const btn = document.getElementById('play-btn');
            const slider = document.getElementById('time-offset');

            if (isPlaying) {
                clearInterval(animationInterval);
                btn.textContent = 'Play Animation';
                isPlaying = false;
            } else {
                btn.textContent = 'Pause';
                isPlaying = true;

                animationInterval = setInterval(() => {
                    let val = parseInt(slider.value) + 5;
                    if (val > 720) val = -720;
                    slider.value = val;
                    document.getElementById('offset-value').textContent = val;
                    updatePositions();
                }, 200);
            }
        });

        // Initialize
        initMap();
        loadCelestrakPresets();
        loadSatellites();
        updatePositions();
        updateCoverage();
    </script>
</body>
</html>
'''


if __name__ == "__main__":
    import uvicorn
    print("Starting Satellite Coverage Visualizer...")
    print("Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)
