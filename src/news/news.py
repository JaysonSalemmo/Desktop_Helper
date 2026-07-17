"""
Headlines from the RSS feeds configured in config.json (`news.rss_feeds`).

Parsed with stdlib ElementTree — handles both RSS (<item><title>) and Atom
(<entry><title>). Headlines are taken round-robin across feeds so one feed
doesn't crowd out the others. Result string matches the training format:
headlines joined by "; ".
"""
import xml.etree.ElementTree as ET

import requests

TIMEOUT = 10
_ATOM = "{http://www.w3.org/2005/Atom}"


def parse_titles(xml_text: str) -> list[str]:
    root = ET.fromstring(xml_text)
    items = root.findall(".//item/title") or root.findall(f".//{_ATOM}entry/{_ATOM}title")
    return [t.text.strip() for t in items if t.text and t.text.strip()]


def _fetch_titles(url: str) -> list[str]:
    try:
        resp = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "desktop-helper/0.1"})
        resp.raise_for_status()
        return parse_titles(resp.text)
    except Exception:
        return []  # a dead feed shouldn't kill the whole headline run


def headlines(feed_urls: list[str], max_headlines: int = 5) -> list[str]:
    per_feed = [_fetch_titles(url) for url in feed_urls]
    picked: list[str] = []
    i = 0
    while len(picked) < max_headlines and any(feed[i:] for feed in per_feed):
        for feed in per_feed:
            if i < len(feed) and len(picked) < max_headlines:
                picked.append(feed[i])
        i += 1
    return picked
