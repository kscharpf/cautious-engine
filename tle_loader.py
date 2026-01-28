"""
TLE (Two-Line Element) loading and parsing utilities.

This module provides functionality for:
- Parsing TLE data from files and text
- Validating TLE line formats
- Fetching TLEs from URLs (including CelesTrak)
"""

import re
from pathlib import Path
from typing import List, Tuple, Optional
import httpx


# CelesTrak preset URLs for common satellite groups
CELESTRAK_PRESETS = {
    "iss": "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=TLE",
    "starlink": "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=TLE",
    "gps": "https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=TLE",
    "weather": "https://celestrak.org/NORAD/elements/gp.php?GROUP=weather&FORMAT=TLE",
    "noaa": "https://celestrak.org/NORAD/elements/gp.php?GROUP=noaa&FORMAT=TLE",
    "goes": "https://celestrak.org/NORAD/elements/gp.php?GROUP=goes&FORMAT=TLE",
    "galileo": "https://celestrak.org/NORAD/elements/gp.php?GROUP=galileo&FORMAT=TLE",
    "glonass": "https://celestrak.org/NORAD/elements/gp.php?GROUP=glo-ops&FORMAT=TLE",
    "beidou": "https://celestrak.org/NORAD/elements/gp.php?GROUP=beidou&FORMAT=TLE",
    "iridium": "https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-NEXT&FORMAT=TLE",
    "planet": "https://celestrak.org/NORAD/elements/gp.php?GROUP=planet&FORMAT=TLE",
    "spire": "https://celestrak.org/NORAD/elements/gp.php?GROUP=spire&FORMAT=TLE",
    "active": "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=TLE",
}


class TLEParseError(Exception):
    """Raised when TLE parsing fails."""
    pass


class TLEFetchError(Exception):
    """Raised when fetching TLEs from URL fails."""
    pass


def validate_tle_lines(line1: str, line2: str, strict: bool = False) -> bool:
    """
    Validate TLE line format.

    Args:
        line1: First line of TLE (starts with '1')
        line2: Second line of TLE (starts with '2')
        strict: If True, also validate checksums (default: False)

    Returns:
        True if both lines are valid TLE format
    """
    line1 = line1.strip()
    line2 = line2.strip()

    # Check line lengths
    if len(line1) != 69 or len(line2) != 69:
        return False

    # Check line numbers
    if not line1.startswith('1 ') or not line2.startswith('2 '):
        return False

    # Check that NORAD catalog numbers match
    try:
        norad1 = line1[2:7].strip()
        norad2 = line2[2:7].strip()
        if norad1 != norad2:
            return False
    except (IndexError, ValueError):
        return False

    # Strict mode: also validate checksums
    if strict:
        def calc_checksum(line: str) -> int:
            """Calculate TLE line checksum (modulo 10)."""
            checksum = 0
            for char in line[:-1]:  # Exclude the checksum digit itself
                if char.isdigit():
                    checksum += int(char)
                elif char == '-':
                    checksum += 1
            return checksum % 10

        try:
            expected_checksum1 = int(line1[-1])
            expected_checksum2 = int(line2[-1])

            if calc_checksum(line1) != expected_checksum1:
                return False
            if calc_checksum(line2) != expected_checksum2:
                return False
        except ValueError:
            return False

    return True


def parse_tle_text(text: str, validate: bool = True, strict_checksum: bool = False) -> List[Tuple[str, str, str]]:
    """
    Parse TLE data from text content.

    Handles both 3-line format (name + line1 + line2) and 2-line format.

    Args:
        text: Text containing TLE data
        validate: Whether to validate TLE format (line length, numbers)
        strict_checksum: Whether to also validate TLE checksums

    Returns:
        List of (name, line1, line2) tuples

    Raises:
        TLEParseError: If parsing fails
    """
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]

    if not lines:
        raise TLEParseError("Empty TLE data")

    tles = []
    i = 0

    while i < len(lines):
        # Check if current line is TLE line 1
        if lines[i].startswith('1 ') and len(lines[i]) == 69:
            # Two-line format (no name)
            if i + 1 >= len(lines):
                raise TLEParseError(f"Missing TLE line 2 at line {i + 1}")

            line1 = lines[i]
            line2 = lines[i + 1]

            if not line2.startswith('2 '):
                raise TLEParseError(f"Invalid TLE line 2 at line {i + 2}")

            if validate and not validate_tle_lines(line1, line2, strict=strict_checksum):
                raise TLEParseError(f"TLE validation failed at line {i + 1}")

            # Extract NORAD catalog number as name
            name = f"NORAD-{line1[2:7].strip()}"
            tles.append((name, line1, line2))
            i += 2

        elif not lines[i].startswith('1 ') and not lines[i].startswith('2 '):
            # Three-line format (name line)
            if i + 2 >= len(lines):
                raise TLEParseError(f"Incomplete TLE at line {i + 1}")

            name = lines[i]
            line1 = lines[i + 1]
            line2 = lines[i + 2]

            if not line1.startswith('1 '):
                raise TLEParseError(f"Invalid TLE line 1 at line {i + 2}")
            if not line2.startswith('2 '):
                raise TLEParseError(f"Invalid TLE line 2 at line {i + 3}")

            if validate and not validate_tle_lines(line1, line2, strict=strict_checksum):
                raise TLEParseError(f"TLE validation failed for {name}")

            tles.append((name.strip(), line1, line2))
            i += 3

        else:
            raise TLEParseError(f"Unexpected line format at line {i + 1}: {lines[i][:50]}...")

    return tles


def parse_tle_file(path: str, validate: bool = True, strict_checksum: bool = False) -> List[Tuple[str, str, str]]:
    """
    Parse TLE data from a file.

    Args:
        path: Path to the TLE file
        validate: Whether to validate TLE format
        strict_checksum: Whether to also validate TLE checksums

    Returns:
        List of (name, line1, line2) tuples

    Raises:
        TLEParseError: If parsing fails
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"TLE file not found: {path}")

    text = file_path.read_text()
    return parse_tle_text(text, validate=validate, strict_checksum=strict_checksum)


async def fetch_tle_from_url(url: str, timeout: float = 30.0) -> str:
    """
    Fetch TLE data from a URL asynchronously.

    Args:
        url: URL to fetch TLE data from
        timeout: Request timeout in seconds

    Returns:
        TLE text content

    Raises:
        TLEFetchError: If fetch fails
    """
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text
    except httpx.TimeoutException:
        raise TLEFetchError(f"Timeout fetching TLE from {url}")
    except httpx.HTTPStatusError as e:
        raise TLEFetchError(f"HTTP error {e.response.status_code} fetching TLE from {url}")
    except httpx.RequestError as e:
        raise TLEFetchError(f"Failed to fetch TLE from {url}: {e}")


def fetch_tle_from_url_sync(url: str, timeout: float = 30.0) -> str:
    """
    Fetch TLE data from a URL synchronously.

    Args:
        url: URL to fetch TLE data from
        timeout: Request timeout in seconds

    Returns:
        TLE text content

    Raises:
        TLEFetchError: If fetch fails
    """
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.text
    except httpx.TimeoutException:
        raise TLEFetchError(f"Timeout fetching TLE from {url}")
    except httpx.HTTPStatusError as e:
        raise TLEFetchError(f"HTTP error {e.response.status_code} fetching TLE from {url}")
    except httpx.RequestError as e:
        raise TLEFetchError(f"Failed to fetch TLE from {url}: {e}")


async def fetch_celestrak_preset(preset_name: str) -> List[Tuple[str, str, str]]:
    """
    Fetch TLEs from a CelesTrak preset.

    Args:
        preset_name: Name of the preset (e.g., 'iss', 'starlink', 'gps')

    Returns:
        List of (name, line1, line2) tuples

    Raises:
        ValueError: If preset doesn't exist
        TLEFetchError: If fetch fails
    """
    preset_name = preset_name.lower()

    if preset_name not in CELESTRAK_PRESETS:
        available = ', '.join(sorted(CELESTRAK_PRESETS.keys()))
        raise ValueError(f"Unknown CelesTrak preset: {preset_name}. Available: {available}")

    url = CELESTRAK_PRESETS[preset_name]
    text = await fetch_tle_from_url(url)
    return parse_tle_text(text)


def fetch_celestrak_preset_sync(preset_name: str) -> List[Tuple[str, str, str]]:
    """
    Fetch TLEs from a CelesTrak preset synchronously.

    Args:
        preset_name: Name of the preset (e.g., 'iss', 'starlink', 'gps')

    Returns:
        List of (name, line1, line2) tuples

    Raises:
        ValueError: If preset doesn't exist
        TLEFetchError: If fetch fails
    """
    preset_name = preset_name.lower()

    if preset_name not in CELESTRAK_PRESETS:
        available = ', '.join(sorted(CELESTRAK_PRESETS.keys()))
        raise ValueError(f"Unknown CelesTrak preset: {preset_name}. Available: {available}")

    url = CELESTRAK_PRESETS[preset_name]
    text = fetch_tle_from_url_sync(url)
    return parse_tle_text(text)


def get_celestrak_presets() -> dict:
    """
    Get available CelesTrak presets and their URLs.

    Returns:
        Dictionary of preset names to URLs
    """
    return CELESTRAK_PRESETS.copy()


if __name__ == "__main__":
    # Demo the module
    print("TLE Loader Module Demo")
    print("=" * 50)

    # Show available presets
    print("\nAvailable CelesTrak presets:")
    for name, url in sorted(CELESTRAK_PRESETS.items()):
        print(f"  {name}: {url}")

    # Example TLE text parsing
    sample_tle = """ISS (ZARYA)
1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990
2 25544  51.6443 242.2052 0004885 264.6463 206.3557 15.49165514212791"""

    print("\nParsing sample TLE:")
    try:
        tles = parse_tle_text(sample_tle)
        for name, line1, line2 in tles:
            print(f"  Name: {name}")
            print(f"  Line 1: {line1[:30]}...")
            print(f"  Line 2: {line2[:30]}...")
    except TLEParseError as e:
        print(f"  Error: {e}")

    # Validate TLE lines
    print("\nTLE validation:")
    valid_line1 = "1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990"
    valid_line2 = "2 25544  51.6443 242.2052 0004885 264.6463 206.3557 15.49165514212791"
    print(f"  Valid TLE: {validate_tle_lines(valid_line1, valid_line2)}")

    invalid_line1 = "1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9999"  # Bad checksum
    print(f"  Invalid checksum: {validate_tle_lines(invalid_line1, valid_line2)}")
