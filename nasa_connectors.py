"""
nasa_connectors.py
===================
API wrappers and scraping utilities for all four NASA data sources used by
the Knowledge Assistant:

    1. api.nasa.gov       -> APOD, Mars Rover Photos, NeoWs (asteroids)
    2. data.nasa.gov      -> Open-data dataset metadata (Socrata API)
    3. earthdata.nasa.gov -> Collection metadata via the CMR search API
    4. nasa.gov/news      -> RSS feed (feedparser) + HTML scrape fallback (BeautifulSoup)

Every public function returns a `List[Dict[str, Any]]` of "raw records" —
plain dictionaries with a predictable shape — so the ingestion layer
(`ingest.py`) can normalize them into LangChain `Document` objects without
needing to know the quirks of each upstream API.

Design notes
------------
* Every network call is wrapped in try/except and returns an empty list on
  failure rather than raising, so a single flaky source never crashes the
  whole ingestion run. Failures are logged via the standard `logging` module.
* A small retry-with-backoff helper (`_request_with_retries`) centralizes
  timeout/rate-limit handling for all `requests`-based calls.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import feedparser
import requests
from bs4 import BeautifulSoup

from config import (
    DATA_NASA_GOV_URL,
    EARTHDATA_CMR_URL,
    MAX_RETRIES,
    NASA_APOD_URL,
    NASA_MARS_PHOTOS_URL,
    NASA_NEOWS_URL,
    NASA_NEWS_HTML_URL,
    NASA_NEWS_RSS_URL,
    REQUEST_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# --------------------------------------------------------------------------- #
# Shared HTTP helper
# --------------------------------------------------------------------------- #
def _request_with_retries(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    max_retries: int = MAX_RETRIES,
) -> Optional[requests.Response]:
    """
    Perform a GET request with basic retry/backoff handling.

    Handles:
        - Connection timeouts (requests.exceptions.Timeout)
        - Generic connection errors (requests.exceptions.ConnectionError)
        - HTTP 429 (rate limit) with a short backoff before retrying
        - Other non-200 status codes (logged, returns None)

    Returns
    -------
    Optional[requests.Response]
        The successful response object, or None if all retries were exhausted.
    """
    for attempt in range(1, max_retries + 2):  # +1 initial attempt
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code == 429:
                wait = 2 ** attempt
                logger.warning(
                    "Rate limited (429) on %s — backing off %ss (attempt %s/%s)",
                    url,
                    wait,
                    attempt,
                    max_retries + 1,
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout:
            logger.warning(
                "Timeout on %s (attempt %s/%s)", url, attempt, max_retries + 1
            )
        except requests.exceptions.ConnectionError as exc:
            logger.warning(
                "Connection error on %s: %s (attempt %s/%s)",
                url,
                exc,
                attempt,
                max_retries + 1,
            )
        except requests.exceptions.HTTPError as exc:
            logger.error("HTTP error on %s: %s", url, exc)
            return None
        except requests.exceptions.RequestException as exc:
            logger.error("Unexpected request error on %s: %s", url, exc)
            return None
        time.sleep(1)
    logger.error("Exhausted retries for %s", url)
    return None


# --------------------------------------------------------------------------- #
# 1. api.nasa.gov — APOD
# --------------------------------------------------------------------------- #
def fetch_apod(api_key: str, days: int = 5) -> List[Dict[str, Any]]:
    """
    Fetch the Astronomy Picture of the Day for the last `days` days.

    Parameters
    ----------
    api_key : str
        NASA API key (or "DEMO_KEY").
    days : int
        Number of most-recent days to fetch (APOD supports a date range).

    Returns
    -------
    List[Dict[str, Any]]
        One record per day, each containing title/explanation/date/url/media_type.
    """
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=max(days - 1, 0))

    params = {
        "api_key": api_key,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }
    response = _request_with_retries(NASA_APOD_URL, params=params)
    if response is None:
        return []

    try:
        payload = response.json()
    except ValueError:
        logger.error("APOD response was not valid JSON")
        return []

    # Single-day responses come back as a dict, not a list.
    records = payload if isinstance(payload, list) else [payload]

    results: List[Dict[str, Any]] = []
    for item in records:
        if "error" in item:
            logger.warning("APOD API error: %s", item.get("error"))
            continue
        results.append(
            {
                "source": "APOD",
                "category": "astronomy_picture_of_the_day",
                "title": item.get("title", "Untitled APOD entry"),
                "text": item.get("explanation", ""),
                "url": item.get("url", ""),
                "date": item.get("date", ""),
                "media_type": item.get("media_type", ""),
            }
        )
    return results


# --------------------------------------------------------------------------- #
# 1. api.nasa.gov — Mars Rover Photos
# --------------------------------------------------------------------------- #
def fetch_mars_rover_photos(
    api_key: str, sol: int = 1000, rover: str = "curiosity", page: int = 1
) -> List[Dict[str, Any]]:
    """
    Fetch Mars rover photo metadata for a given Martian sol (day).

    Parameters
    ----------
    api_key : str
        NASA API key.
    sol : int
        Martian sol (day since landing) to query.
    rover : str
        Rover name (curiosity, opportunity, spirit, perseverance).
    page : int
        Pagination page (25 photos per page).

    Returns
    -------
    List[Dict[str, Any]]
        One record per photo, with camera/rover/date metadata. The raw image
        is NOT downloaded — only metadata + image URL, keeping ingestion light.
    """
    url = NASA_MARS_PHOTOS_URL.replace("curiosity", rover)
    params = {"api_key": api_key, "sol": sol, "page": page}
    response = _request_with_retries(url, params=params)
    if response is None:
        return []

    try:
        payload = response.json()
    except ValueError:
        logger.error("Mars Rover Photos response was not valid JSON")
        return []

    photos = payload.get("photos", [])
    results: List[Dict[str, Any]] = []
    for photo in photos:
        camera = photo.get("camera", {})
        rover_info = photo.get("rover", {})
        summary_text = (
            f"Photo taken by the {rover_info.get('name', rover)} rover's "
            f"{camera.get('full_name', camera.get('name', 'unknown camera'))} "
            f"on sol {photo.get('sol')} (Earth date {photo.get('earth_date')}). "
            f"Rover status: {rover_info.get('status', 'unknown')}."
        )
        results.append(
            {
                "source": "Mars Rover Photos",
                "category": "mars_exploration",
                "title": f"{rover_info.get('name', rover)} — Sol {photo.get('sol')} ({camera.get('name', '')})",
                "text": summary_text,
                "url": photo.get("img_src", ""),
                "date": photo.get("earth_date", ""),
                "media_type": "image",
            }
        )
    return results


# --------------------------------------------------------------------------- #
# 1. api.nasa.gov — NeoWs (Near-Earth Object Web Service)
# --------------------------------------------------------------------------- #
def fetch_neows(api_key: str, days: int = 3) -> List[Dict[str, Any]]:
    """
    Fetch near-Earth asteroid data for the next `days` days (max 7 per NeoWs limits).

    Returns
    -------
    List[Dict[str, Any]]
        One record per asteroid, summarizing hazard status, diameter, and
        closest approach details.
    """
    start_date = datetime.now(timezone.utc).date()
    end_date = start_date + timedelta(days=min(max(days - 1, 0), 6))

    params = {
        "api_key": api_key,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }
    response = _request_with_retries(NASA_NEOWS_URL, params=params)
    if response is None:
        return []

    try:
        payload = response.json()
    except ValueError:
        logger.error("NeoWs response was not valid JSON")
        return []

    near_earth_objects = payload.get("near_earth_objects", {})
    results: List[Dict[str, Any]] = []
    for date_str, objects in near_earth_objects.items():
        for obj in objects:
            est_diameter = obj.get("estimated_diameter", {}).get("meters", {})
            close_approach = (
                obj.get("close_approach_data", [{}])[0]
                if obj.get("close_approach_data")
                else {}
            )
            summary_text = (
                f"Asteroid {obj.get('name', 'Unknown')} "
                f"is {'POTENTIALLY HAZARDOUS' if obj.get('is_potentially_hazardous_asteroid') else 'not classified as hazardous'}. "
                f"Estimated diameter: {est_diameter.get('estimated_diameter_min', 0):.1f}–"
                f"{est_diameter.get('estimated_diameter_max', 0):.1f} meters. "
                f"Close approach on {close_approach.get('close_approach_date_full', date_str)} "
                f"at a relative velocity of "
                f"{close_approach.get('relative_velocity', {}).get('kilometers_per_hour', 'unknown')} km/h, "
                f"miss distance of "
                f"{close_approach.get('miss_distance', {}).get('kilometers', 'unknown')} km."
            )
            results.append(
                {
                    "source": "NeoWs",
                    "category": "near_earth_objects",
                    "title": f"Asteroid {obj.get('name', 'Unknown')} ({date_str})",
                    "text": summary_text,
                    "url": obj.get("nasa_jpl_url", ""),
                    "date": date_str,
                    "media_type": "text",
                }
            )
    return results


# --------------------------------------------------------------------------- #
# 2. data.nasa.gov — Open Data dataset metadata
# --------------------------------------------------------------------------- #
def fetch_data_nasa_gov_datasets(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch dataset metadata records from data.nasa.gov's Socrata-backed open
    data API.

    Notes
    -----
    data.nasa.gov exposes many individual Socrata "resource" endpoints; this
    function targets one general dataset resource as a representative example
    and is intentionally defensive about schema differences between datasets.

    Returns
    -------
    List[Dict[str, Any]]
        Dataset metadata records normalized into the standard record shape.
    """
    params = {"$limit": limit}
    response = _request_with_retries(DATA_NASA_GOV_URL, params=params)
    if response is None:
        return []

    try:
        payload = response.json()
    except ValueError:
        logger.error("data.nasa.gov response was not valid JSON")
        return []

    if not isinstance(payload, list):
        logger.warning("data.nasa.gov returned an unexpected payload shape")
        return []

    results: List[Dict[str, Any]] = []
    for item in payload:
        # Socrata resources vary in schema; fall back gracefully across
        # common field name variants.
        title = item.get("title") or item.get("name") or "Untitled dataset"
        description = (
            item.get("description")
            or item.get("summary")
            or "No description available."
        )
        results.append(
            {
                "source": "data.nasa.gov",
                "category": "open_dataset",
                "title": str(title),
                "text": str(description),
                "url": item.get("landing_page", item.get("url", "")),
                "date": item.get("issued", item.get("modified", "")),
                "media_type": "metadata",
            }
        )
    return results


# --------------------------------------------------------------------------- #
# 3. earthdata.nasa.gov — CMR collection metadata
# --------------------------------------------------------------------------- #
def fetch_earthdata_collections(
    keyword: str = "climate", page_size: int = 10
) -> List[Dict[str, Any]]:
    """
    Fetch Earth science / climate dataset collection metadata from NASA's
    Common Metadata Repository (CMR) — the public search backend for
    earthdata.nasa.gov.

    Parameters
    ----------
    keyword : str
        Free-text keyword to filter collections (e.g. "climate", "drought").
    page_size : int
        Number of collections to retrieve.

    Returns
    -------
    List[Dict[str, Any]]
        One record per matching collection, with title/summary/link.
    """
    params = {"keyword": keyword, "page_size": page_size}
    headers = {"Accept": "application/json"}
    response = _request_with_retries(EARTHDATA_CMR_URL, params=params, headers=headers)
    if response is None:
        return []

    try:
        payload = response.json()
    except ValueError:
        logger.error("EarthData CMR response was not valid JSON")
        return []

    entries = payload.get("feed", {}).get("entry", [])
    results: List[Dict[str, Any]] = []
    for entry in entries:
        links = entry.get("links", [])
        primary_link = links[0].get("href", "") if links else ""
        results.append(
            {
                "source": "EarthData",
                "category": "earth_science_collection",
                "title": entry.get("title", "Untitled collection"),
                "text": entry.get("summary", "No summary available."),
                "url": primary_link,
                "date": entry.get("time_start", entry.get("updated", "")),
                "media_type": "metadata",
            }
        )
    return results


# --------------------------------------------------------------------------- #
# 4. nasa.gov/news — RSS feed (primary) + HTML scrape (fallback)
# --------------------------------------------------------------------------- #
def fetch_nasa_news_rss(max_items: int = 10) -> List[Dict[str, Any]]:
    """
    Parse NASA's official news RSS feed via `feedparser`.

    Returns
    -------
    List[Dict[str, Any]]
        One record per news item (title, summary, link, published date).
        Returns an empty list (never raises) if the feed is unreachable or
        malformed — callers should treat that as "try the HTML fallback".
    """
    try:
        feed = feedparser.parse(NASA_NEWS_RSS_URL)
    except Exception as exc:  # feedparser rarely raises, but be defensive
        logger.error("feedparser failed on NASA news RSS: %s", exc)
        return []

    if getattr(feed, "bozo", 0) and not feed.entries:
        logger.warning("NASA news RSS feed appears malformed or empty")
        return []

    results: List[Dict[str, Any]] = []
    for entry in feed.entries[:max_items]:
        # Strip any embedded HTML tags from the summary using BeautifulSoup
        # for a clean plain-text chunk suitable for embedding.
        raw_summary = getattr(entry, "summary", "")
        clean_summary = BeautifulSoup(raw_summary, "html.parser").get_text(
            separator=" ", strip=True
        )
        results.append(
            {
                "source": "NASA News",
                "category": "news_rss",
                "title": getattr(entry, "title", "Untitled news item"),
                "text": clean_summary,
                "url": getattr(entry, "link", ""),
                "date": getattr(entry, "published", ""),
                "media_type": "text",
            }
        )
    return results


def fetch_nasa_news_html(max_items: int = 10) -> List[Dict[str, Any]]:
    """
    Fallback scraper for NASA news using `BeautifulSoup`, used when the RSS
    feed is unavailable or returns no entries.

    Notes
    -----
    HTML scraping is inherently fragile to site redesigns. This function is
    written defensively: it looks for common article/heading patterns and
    returns whatever it can find rather than raising on a missed selector.

    Returns
    -------
    List[Dict[str, Any]]
        Scraped news headline + snippet records.
    """
    headers = {"User-Agent": "Mozilla/5.0 (NASA-Knowledge-Assistant/1.0)"}
    response = _request_with_retries(NASA_NEWS_HTML_URL, headers=headers)
    if response is None:
        return []

    try:
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception as exc:
        logger.error("BeautifulSoup failed to parse NASA news HTML: %s", exc)
        return []

    results: List[Dict[str, Any]] = []
    # Generic strategy: look for anchor tags whose text looks like a headline
    # (reasonably long, not a nav link). Real deployments should tailor this
    # selector to the current site markup.
    candidates = soup.find_all("a", href=True)
    seen_titles = set()
    for tag in candidates:
        title = tag.get_text(strip=True)
        href = tag["href"]
        if not title or len(title) < 25 or title in seen_titles:
            continue
        if not href.startswith("http"):
            href = f"https://www.nasa.gov{href}"
        seen_titles.add(title)
        results.append(
            {
                "source": "NASA News",
                "category": "news_html_scrape",
                "title": title,
                "text": title,  # snippet unavailable via this generic selector
                "url": href,
                "date": datetime.now(timezone.utc).isoformat(),
                "media_type": "text",
            }
        )
        if len(results) >= max_items:
            break
    return results


def fetch_nasa_news(max_items: int = 10) -> List[Dict[str, Any]]:
    """
    Public entry point for NASA news: try RSS first, fall back to HTML scrape.

    Returns
    -------
    List[Dict[str, Any]]
        News records from whichever strategy succeeded first.
    """
    rss_results = fetch_nasa_news_rss(max_items=max_items)
    if rss_results:
        return rss_results
    logger.info("RSS feed returned no results — falling back to HTML scrape")
    return fetch_nasa_news_html(max_items=max_items)


# --------------------------------------------------------------------------- #
# Aggregate convenience function  (concurrent fetching)
# --------------------------------------------------------------------------- #
def fetch_all_sources(
    api_key: str,
    apod_days: int = 5,
    mars_sol: int = 1000,
    neows_days: int = 3,
    dataset_limit: int = 10,
    earthdata_keyword: str = "climate",
    news_items: int = 10,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch from all six NASA sources concurrently using a thread pool.

    Each source runs in its own thread so network I/O for all six endpoints
    overlaps instead of running sequentially. Failures in one source are
    isolated — the others still return data. Wall-clock fetch time drops
    from ~sum(source_latencies) to ~max(source_latency).

    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        Mapping of source name -> list of raw records, in a deterministic order.
    """
    tasks: Dict[str, Any] = {
        "APOD":             lambda: fetch_apod(api_key=api_key, days=apod_days),
        "Mars Rover Photos": lambda: fetch_mars_rover_photos(api_key=api_key, sol=mars_sol),
        "NeoWs":            lambda: fetch_neows(api_key=api_key, days=neows_days),
        "data.nasa.gov":    lambda: fetch_data_nasa_gov_datasets(limit=dataset_limit),
        "EarthData":        lambda: fetch_earthdata_collections(keyword=earthdata_keyword),
        "NASA News":        lambda: fetch_nasa_news(max_items=news_items),
    }

    results: Dict[str, List[Dict[str, Any]]] = {}

    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        future_to_name = {
            executor.submit(fn): name for name, fn in tasks.items()
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                logger.error("Unexpected error fetching '%s': %s", name, exc)
                results[name] = []

    # Return in a stable, predictable order regardless of completion order.
    return {name: results.get(name, []) for name in tasks}
