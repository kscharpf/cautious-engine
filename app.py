"""
FastAPI backend for satellite coverage visualization.

Provides REST API endpoints for calculating satellite positions and coverage.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional
from dataclasses import dataclass

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np

from satellite import (
    Satrec, parse_tle, propagate, datetime_to_gmst, teme_to_ecef,
    calculate_look_angle, geodetic_to_ecef, is_visible, GroundPosition
)
from constellation import Constellation, SAMPLE_STARLINK_TLES

app = FastAPI(title="Satellite Coverage Visualizer")

# Sample constellation (initialized once)
CONSTELLATION = Constellation("Demo Constellation")

# Add ISS
CONSTELLATION.add_satellite(
    "ISS (ZARYA)",
    "1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990",
    "2 25544  51.6443 242.2052 0004885 264.6463 206.3557 15.49165514212791"
)

# Add Starlink satellites
CONSTELLATION.add_from_tle_list(SAMPLE_STARLINK_TLES)


@app.get("/")
async def root():
    """Serve the main visualization page."""
    return HTMLResponse(content=get_html_page(), status_code=200)


@app.get("/api/satellites")
async def get_satellites():
    """Get list of satellites in the constellation."""
    return {
        "satellites": [name for name, _ in CONSTELLATION.satellites],
        "count": CONSTELLATION.size
    }


@app.get("/api/positions")
async def get_satellite_positions(
    timestamp: Optional[str] = None,
    offset_minutes: int = 0
):
    """
    Get current positions of all satellites.

    Args:
        timestamp: ISO format timestamp (default: now)
        offset_minutes: Offset from timestamp in minutes
    """
    if timestamp:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    else:
        # Use a fixed time for reproducibility in demo
        dt = datetime(2020, 2, 14, 12, 0, 0, tzinfo=timezone.utc)

    dt = dt + timedelta(minutes=offset_minutes)
    gmst = datetime_to_gmst(dt)

    positions = []
    for name, sat in CONSTELLATION.satellites:
        try:
            pos, vel = propagate(sat, dt)
            sat_ecef = teme_to_ecef(pos, gmst)

            # Convert ECEF to lat/lon/alt
            # Simple conversion (not accounting for ellipsoid perfectly)
            r = np.sqrt(sat_ecef.x**2 + sat_ecef.y**2 + sat_ecef.z**2)
            lat = np.arcsin(sat_ecef.z / r) * 180 / np.pi
            lon = np.arctan2(sat_ecef.y, sat_ecef.x) * 180 / np.pi
            alt = r - 6371  # Approximate altitude in km

            positions.append({
                "name": name,
                "latitude": float(lat),
                "longitude": float(lon),
                "altitude_km": float(alt),
                "velocity_km_s": float(np.sqrt(vel.vx**2 + vel.vy**2 + vel.vz**2))
            })
        except ValueError as e:
            # Propagation error
            continue

    return {
        "timestamp": dt.isoformat(),
        "positions": positions
    }


@app.get("/api/coverage/instant")
async def get_instant_coverage(
    timestamp: Optional[str] = None,
    offset_minutes: int = 0,
    resolution: float = 5.0,
    min_elevation: float = 10.0
):
    """
    Get coverage map for a single instant.

    Returns a grid of visibility values (0 or 1).
    """
    if timestamp:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    else:
        dt = datetime(2020, 2, 14, 12, 0, 0, tzinfo=timezone.utc)

    dt = dt + timedelta(minutes=offset_minutes)
    gmst = datetime_to_gmst(dt)

    # Create grid
    lats = np.arange(-90, 90, resolution)
    lons = np.arange(-180, 180, resolution)

    # Get satellite positions
    sat_positions = []
    for name, sat in CONSTELLATION.satellites:
        try:
            pos, _ = propagate(sat, dt)
            sat_ecef = teme_to_ecef(pos, gmst)
            sat_positions.append(sat_ecef)
        except ValueError:
            continue

    # Calculate visibility grid
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
        "coverage": coverage
    }


@app.get("/api/coverage/point")
async def get_point_coverage(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    timestamp: Optional[str] = None,
    duration_hours: float = 24.0,
    step_minutes: float = 1.0,
    min_elevation: float = 10.0
):
    """
    Get time-series coverage for a specific point.
    """
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

        # Check each satellite
        any_visible = False
        visible_sats = []

        for name, sat in CONSTELLATION.satellites:
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
        "timeline": timeline[:100]  # Limit response size
    }


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
            width: 320px;
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
        input[type="range"], input[type="number"] {
            width: 100%;
            background: #0f3460;
            border: none;
            padding: 8px;
            border-radius: 4px;
            color: #fff;
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
            max-height: 200px;
            overflow-y: auto;
            background: #0f3460;
            border-radius: 8px;
            padding: 10px;
        }
        .satellite-item {
            padding: 8px;
            border-bottom: 1px solid #1a1a2e;
            font-size: 0.9rem;
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
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h1>🛰️ Satellite Coverage</h1>

            <h2>Time Controls</h2>
            <div class="control-group">
                <label>Time Offset: <span id="offset-value">0</span> minutes</label>
                <input type="range" id="time-offset" min="-720" max="720" value="0">
            </div>

            <div class="control-group">
                <button id="play-btn">▶ Play Animation</button>
            </div>

            <h2>Display Options</h2>
            <div class="control-group">
                <label>Min Elevation: <span id="elev-value">10</span>°</label>
                <input type="range" id="min-elevation" min="0" max="45" value="10">
            </div>

            <div class="control-group">
                <label>Grid Resolution: <span id="res-value">5</span>°</label>
                <input type="range" id="resolution" min="2" max="15" value="5">
            </div>

            <button id="refresh-btn">🔄 Refresh Coverage</button>

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
                    <span>Visible Points:</span>
                    <span class="stat-value" id="stat-visible">-</span>
                </div>
                <div class="stat-row">
                    <span>Coverage:</span>
                    <span class="stat-value" id="stat-coverage">-</span>
                </div>
            </div>

            <h2>Click Map for Details</h2>
            <div class="click-info" id="click-info">
                Click anywhere on the map to see coverage details for that location.
            </div>
        </div>

        <div class="map-container">
            <div id="map"></div>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: rgba(0, 217, 255, 0.6);"></div>
                    <span>Coverage Area</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ff4444;"></div>
                    <span>Satellite Position</span>
                </div>
            </div>
            <div class="loading" id="loading" style="display: none;">
                Loading coverage data...
            </div>
        </div>
    </div>

    <script>
        // Initialize map
        const map = L.map('map').setView([20, 0], 2);
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '© OpenStreetMap contributors © CARTO',
            maxZoom: 19
        }).addTo(map);

        // Layers
        let coverageLayer = null;
        let satelliteMarkers = [];
        let animationInterval = null;
        let isPlaying = false;

        // Load satellites
        async function loadSatellites() {
            const response = await fetch('/api/satellites');
            const data = await response.json();

            const list = document.getElementById('satellite-list');
            list.innerHTML = data.satellites.map(name =>
                `<div class="satellite-item"><span class="satellite-name">🛰️</span> ${name}</div>`
            ).join('');

            document.getElementById('stat-sats').textContent = data.count;
        }

        // Update satellite positions
        async function updatePositions() {
            const offset = document.getElementById('time-offset').value;
            const response = await fetch(`/api/positions?offset_minutes=${offset}`);
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
                    Lat: ${sat.latitude.toFixed(2)}°<br>
                    Lon: ${sat.longitude.toFixed(2)}°<br>
                    Alt: ${sat.altitude_km.toFixed(0)} km<br>
                    Vel: ${sat.velocity_km_s.toFixed(2)} km/s
                `);

                satelliteMarkers.push(marker);
            });
        }

        // Update coverage map
        async function updateCoverage() {
            const loading = document.getElementById('loading');
            loading.style.display = 'block';

            const offset = document.getElementById('time-offset').value;
            const resolution = document.getElementById('resolution').value;
            const minElev = document.getElementById('min-elevation').value;

            try {
                const response = await fetch(
                    `/api/coverage/instant?offset_minutes=${offset}&resolution=${resolution}&min_elevation=${minElev}`
                );
                const data = await response.json();

                // Remove old coverage layer
                if (coverageLayer) {
                    map.removeLayer(coverageLayer);
                }

                // Create coverage polygons
                const polygons = [];
                let visibleCount = 0;
                const res = parseFloat(resolution);

                data.coverage.forEach((row, i) => {
                    row.forEach((val, j) => {
                        if (val === 1) {
                            visibleCount++;
                            const lat = data.latitudes[i];
                            const lon = data.longitudes[j];

                            const bounds = [
                                [lat, lon],
                                [lat, lon + res],
                                [lat + res, lon + res],
                                [lat + res, lon]
                            ];

                            polygons.push(L.polygon(bounds, {
                                color: 'rgba(0, 217, 255, 0.3)',
                                fillColor: 'rgba(0, 217, 255, 0.4)',
                                fillOpacity: 0.4,
                                weight: 0
                            }));
                        }
                    });
                });

                coverageLayer = L.layerGroup(polygons).addTo(map);

                // Update stats
                const totalCells = data.coverage.length * data.coverage[0].length;
                const coveragePercent = (visibleCount / totalCells * 100).toFixed(1);

                document.getElementById('stat-visible').textContent = visibleCount;
                document.getElementById('stat-coverage').textContent = coveragePercent + '%';

            } catch (error) {
                console.error('Error loading coverage:', error);
            }

            loading.style.display = 'none';
        }

        // Handle map clicks
        map.on('click', async (e) => {
            const { lat, lng } = e.latlng;
            const info = document.getElementById('click-info');
            info.innerHTML = `Loading data for (${lat.toFixed(2)}°, ${lng.toFixed(2)}°)...`;

            try {
                const response = await fetch(
                    `/api/coverage/point?latitude=${lat}&longitude=${lng}&duration_hours=24`
                );
                const data = await response.json();

                const visibleSats = data.timeline.filter(t => t.visible).length;

                info.innerHTML = `
                    <b>Location:</b> ${lat.toFixed(2)}°, ${lng.toFixed(2)}°<br>
                    <b>24h Coverage:</b> ${data.coverage_percent}%<br>
                    <b>Visible samples:</b> ${visibleSats}/${data.timeline.length}
                `;
            } catch (error) {
                info.innerHTML = 'Error loading data';
            }
        });

        // Event listeners
        document.getElementById('time-offset').addEventListener('input', (e) => {
            document.getElementById('offset-value').textContent = e.target.value;
            updatePositions();
        });

        document.getElementById('min-elevation').addEventListener('input', (e) => {
            document.getElementById('elev-value').textContent = e.target.value;
        });

        document.getElementById('resolution').addEventListener('input', (e) => {
            document.getElementById('res-value').textContent = e.target.value;
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
                btn.textContent = '▶ Play Animation';
                isPlaying = false;
            } else {
                btn.textContent = '⏸ Pause';
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

        // Initial load
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
