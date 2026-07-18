"""
Current weather: NWS station observations first, Open-Meteo as fallback.

Why two sources: Open-Meteo's "current" is model-interpolated forecast data —
it reported "partly cloudy, 0.0 precip" during an actual thunderstorm
(2026-07-18). NWS gives real station observations (free, no key), but only
covers the US; Open-Meteo covers everywhere. Geocoding is Open-Meteo's, cached
per location; the NWS station is likewise resolved once and cached.

Result string matches the training format: "62°F, light rain".
"""
import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
NWS_POINTS_URL = "https://api.weather.gov/points/{lat},{lon}"
NWS_OBS_URL = "https://api.weather.gov/stations/{station}/observations/latest"
NWS_HEADERS = {"User-Agent": "desktop-helper (personal assistant)"}
TIMEOUT = 10

# WMO weather interpretation codes → the plain-language conditions the model
# saw in training data
_CONDITIONS = [
    ((0,), "sunny"),
    ((1, 2), "partly cloudy"),
    ((3,), "overcast"),
    ((45, 48), "foggy"),
    ((51, 53, 55, 56, 57), "light rain"),
    ((61, 63, 80, 81), "rainy"),
    ((65, 82), "heavy rain"),
    ((66, 67), "freezing rain"),
    ((71, 73, 75, 77, 85, 86), "snowy"),
    ((95, 96, 99), "stormy"),
]


def condition_text(code: int) -> str:
    for codes, text in _CONDITIONS:
        if code in codes:
            return text
    return "unsettled"


_geocode_cache: dict[str, tuple[float, float]] = {}


def _geocode(location: str) -> tuple[float, float]:
    if location in _geocode_cache:
        return _geocode_cache[location]
    resp = requests.get(GEOCODE_URL, params={"name": location, "count": 1}, timeout=TIMEOUT)
    resp.raise_for_status()
    results = resp.json().get("results")
    if not results:
        raise ValueError(f"location not found: {location}")
    coords = (results[0]["latitude"], results[0]["longitude"])
    _geocode_cache[location] = coords
    return coords


# NWS textDescription → the training-condition vocabulary
def nws_condition(text: str) -> str:
    t = text.lower()
    if "thunder" in t:
        return "stormy"
    if "fog" in t or "mist" in t or "haze" in t:
        return "foggy"
    if "drizzle" in t or "light rain" in t:
        return "light rain"
    if "heavy rain" in t:
        return "heavy rain"
    if "rain" in t or "shower" in t:
        return "rainy"
    if "snow" in t or "sleet" in t or "ice" in t or "freezing" in t:
        return "snowy"
    if "partly" in t or "few clouds" in t or "scattered" in t:
        return "partly cloudy"
    if "cloud" in t or "overcast" in t:
        return "overcast"
    if "clear" in t or "sunny" in t or "fair" in t:
        return "sunny"
    return t  # unmapped description, verbatim lowercase


_station_cache: dict[tuple[float, float], str] = {}


def _nws_current(lat: float, lon: float) -> str | None:
    """Real station observation, or None (non-US point, missing data, etc.)."""
    try:
        key = (lat, lon)
        if key not in _station_cache:
            point = requests.get(NWS_POINTS_URL.format(lat=lat, lon=lon),
                                 headers=NWS_HEADERS, timeout=TIMEOUT).json()
            stations = requests.get(point["properties"]["observationStations"],
                                    headers=NWS_HEADERS, timeout=TIMEOUT).json()
            _station_cache[key] = \
                stations["features"][0]["properties"]["stationIdentifier"]
        obs = requests.get(NWS_OBS_URL.format(station=_station_cache[key]),
                           headers=NWS_HEADERS, timeout=TIMEOUT).json()["properties"]
        celsius = obs["temperature"]["value"]
        description = obs["textDescription"]
        if celsius is None or not description:
            return None
        return f"{round(celsius * 9 / 5 + 32)}°F, {nws_condition(description)}"
    except Exception:
        return None


def current(location: str) -> str:
    lat, lon = _geocode(location)
    observed = _nws_current(lat, lon)
    if observed is not None:
        return observed

    resp = requests.get(
        FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weather_code",
            "temperature_unit": "fahrenheit",
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()["current"]
    temp = round(data["temperature_2m"])
    return f"{temp}°F, {condition_text(data['weather_code'])}"
