"""
Current weather via Open-Meteo (free, no API key).

Two calls: geocode the configured location name once (cached for the process),
then fetch current conditions. Result string matches the training format:
"62°F, light rain".
"""
import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
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


def current(location: str) -> str:
    lat, lon = _geocode(location)
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
