#!/usr/bin/env python3
# file: ip_visit_stats.py

import requests
from collections import defaultdict
from datetime import datetime, timedelta
import pytz
from rich.console import Console
from rich.table import Table
from rich.pretty import Pretty
from tomlkit.items import Array

API_URL = "https://api.impulsepdr.online/"
TIMEZONE = pytz.timezone("Europe/Moscow")

# Filters
EXCLUDE_BROWSERS = ["Googlebot", "AdsBot-Google"]
EXCLUDE_IPS = ["146.120.163.81"]

# Cache for IP geolocation
ip_geo_cache = {}

def get_geo_ip(ip):
    if ip in ip_geo_cache:
        return ip_geo_cache[ip]
    try:
        response = requests.get(f'http://ip-api.com/json/{ip}', timeout=3)
        data = response.json()
        city = data.get('city', 'Unknown')
        country = data.get('country', 'Unknown')
    except Exception:
        city, country = 'Unknown', 'Unknown'
    ip_geo_cache[ip] = (city, country)
    return city, country


def fetch_data():
    response = requests.get(API_URL)
    response.raise_for_status()
    return response.json()


def parse_visits(visits):
    stats = defaultdict(list)
    now = datetime.now(tz=TIMEZONE)
    for visit in visits:
        if visit["browser"] in EXCLUDE_BROWSERS or visit["ip"] in EXCLUDE_IPS:
            continue
        raw_time = datetime.fromisoformat(visit["time"])
        time = (raw_time + timedelta(hours=3)).astimezone(TIMEZONE)
        delta = now - time
        visit["relative_time"] = format_time_delta(delta)
        visit["time"] = time  # overwrite with adjusted time
        stats[visit["ip"]].append(visit)
    return stats


def format_time_delta(delta: timedelta) -> str:
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds} seconds ago"
    elif seconds < 3600:
        return f"{seconds // 60} minutes ago"
    elif seconds < 86400:
        return f"{seconds // 3600} hours ago"
    else:
        return f"{seconds // 86400} days ago"


def display_stats(ip_stats):
    console = Console()
    table = Table(title="IP Visit Statistics")

    table.add_column("IP Address", style="cyan", no_wrap=True)
    table.add_column("Location", style="yellow")
    table.add_column("Visit Count", style="magenta")
    table.add_column("Last Visit", style="green")

    for ip, visits in sorted(ip_stats.items(), key=lambda x: -len(x[1])):
        last_visit = max(visits, key=lambda v: v["time"])
        city, country = get_geo_ip(ip)
        location = f"{city}, {country}"
        if country == "Ukraine":
            for v in sorted(visits, key=lambda v: v["time"], reverse=True):
                if ("WebView" not in str(v['browser'])):
                    table.add_row(
                    ip,
                    location,
                    str(len(visits)),
                    last_visit["relative_time"]
                )

    console.print(table)
    # console.print("\nDetailed visits per IP:")
    #
    # for ip, visits in ip_stats.items():
    #     city, country = get_geo_ip(ip)
    #     location = f"{city}, {country}"
    #     console.print(f"\n[bold]{ip}[/bold] ({len(visits)} visits) [{location}]:")
    #     for v in sorted(visits, key=lambda v: v["time"], reverse=True):
    #         console.print(
    #             f"  - {v['relative_time']} | {v['device']} | {v['browser']} | {v['os']}"
    #         )


def main():
    data = fetch_data()
    visits: Array = data.get("visits", [])
    ip_stats = parse_visits(visits)
    display_stats(ip_stats)


if __name__ == "__main__":
    main()
