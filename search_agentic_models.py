#!/usr/bin/env python3
"""
Search OpenRouter for free agentic text→text models.
Lists up to 10 models with latest uptime and context size.

Usage:
  python search_agentic_models.py                    # top 10 with live uptime
  python search_agentic_models.py --limit 5         # top 5
  python search_agentic_models.json                  # JSON output
  python search_agentic_models.py --no-monitor      # skip live uptime fetch
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://openrouter.ai/api/v1/models"

# Keywords that suggest agentic capabilities
AGENTIC_KEYWORDS = [
    "agent", "agentic", "tool use", "tool-use", "tool-call",
    "function calling", "function-calling", "autonomous",
    "workflow", "structured output", "json mode",
    "function call", "multi-step", "reasoning",
    "code generation", "automated",
]

# Uptime emoji thresholds: (min_pct, emoji)
UPTIME_THRESHOLDS = [
    (99.9, "🟢"),
    (99.0, "🟡"),
    (95.0, "🟠"),
]

STATUS_MAP = {0: "active", 1: "degraded", 2: "down"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AgenticModel:
    model_id: str = ""
    name: str = ""
    provider: str = "N/A"
    quantization: str = ""
    status: str = "unknown"
    context_length: int = 0
    max_output: int = 0
    uptime_5m: Optional[float] = None
    uptime_30m: Optional[float] = None
    uptime_1d: Optional[float] = None


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def fetch_json(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_model_list() -> dict:
    return fetch_json(BASE_URL)


def fetch_endpoints(canonical_slug: str) -> Optional[dict]:
    try:
        return fetch_json(f"{BASE_URL}/{canonical_slug}/endpoints", timeout=15)
    except (urllib.error.HTTPError, Exception):
        return None


def resolve_canonical_slug(model_id: str, model_list: dict) -> Optional[str]:
    for model in model_list.get("data", []):
        if model["id"] == model_id:
            return model.get("canonical_slug")
    return None


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def is_free(model: dict) -> bool:
    pricing = model.get("pricing", {})
    try:
        return float(pricing.get("prompt", "")) == 0.0 and \
               float(pricing.get("completion", "")) == 0.0
    except (ValueError, TypeError):
        return False


def is_text_to_text(model: dict) -> bool:
    return model.get("architecture", {}).get("modality") == "text->text"


def is_agentic(model: dict) -> bool:
    text = f"{model.get('name', '')} {model.get('description', '')}".lower()
    return any(kw in text for kw in AGENTIC_KEYWORDS)


# ---------------------------------------------------------------------------
# Uptime helpers
# ---------------------------------------------------------------------------

def uptime_indicator(pct: Optional[float]) -> str:
    if pct is None:
        return "⚪"
    for threshold, emoji in UPTIME_THRESHOLDS:
        if pct >= threshold:
            return emoji
    return "🔴"


def parse_best_endpoint(endpoints_data: Optional[dict]) -> dict:
    """Extract best endpoint uptime data."""
    if not endpoints_data:
        return {}
    endpoints = endpoints_data.get("data", {}).get("endpoints", [])
    if not endpoints:
        return {}
    best = max(endpoints, key=lambda e: e.get("uptime_last_1d", 0) or 0)
    return {
        "status": STATUS_MAP.get(best.get("status", -1), "unknown"),
        "provider": best.get("provider_name", "N/A"),
        "quantization": best.get("quantization", ""),
        "context_length": best.get("context_length"),
        "max_output": best.get("max_completion_tokens"),
        "uptime_5m": best.get("uptime_last_5m"),
        "uptime_30m": best.get("uptime_last_30m"),
        "uptime_1d": best.get("uptime_last_1d"),
    }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def search_agentic_models(limit: int = 10, monitor: bool = True) -> list:
    """
    Search OpenRouter for free agentic text→text models.
    Returns list of AgenticModel sorted by context length (descending).
    """
    print("Fetching model list from OpenRouter...")
    try:
        full_list = fetch_model_list()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    all_models = full_list.get("data", [])
    print(f"  Total models available: {len(all_models)}")

    # Filter: free + text→text + agentic
    filtered = [m for m in all_models
                if is_text_to_text(m) and is_free(m) and is_agentic(m)]
    print(f"  Free + agentic + text→text: {len(filtered)}")

    if not filtered:
        print("No matching models found.")
        return []

    # Sort by context length descending, take top N
    filtered.sort(key=lambda m: m.get("context_length", 0), reverse=True)
    candidates = filtered[:limit]

    # Fetch live endpoint data in parallel
    results = []
    if monitor:
        print(f"\nFetching live endpoint data for {len(candidates)} models...\n")
        with ThreadPoolExecutor(max_workers=8) as pool:
            future_to_model = {}
            for raw in candidates:
                top = raw.get("top_provider", {})
                slug = resolve_canonical_slug(raw["id"], full_list)
                if slug:
                    future_to_model[pool.submit(fetch_endpoints, slug)] = (raw, top)
                else:
                    results.append(make_model(raw, top, {}))

            for future in as_completed(future_to_model):
                raw, top = future_to_model[future]
                try:
                    ep_data = future.result()
                except Exception:
                    ep_data = None
                results.append(make_model(raw, top, parse_best_endpoint(ep_data)))
    else:
        for raw in candidates:
            top = raw.get("top_provider", {})
            results.append(make_model(raw, top, {}))

    # Sort by context length descending
    results.sort(key=lambda m: m.context_length, reverse=True)
    return results


def make_model(raw: dict, top: dict, ep: dict) -> AgenticModel:
    ctx = ep.get("context_length") or raw.get("context_length", 0) or 0
    return AgenticModel(
        model_id=raw["id"],
        name=raw.get("name", raw["id"]),
        provider=ep.get("provider", "N/A"),
        quantization=ep.get("quantization", ""),
        status=ep.get("status", "no endpoints"),
        context_length=ctx,
        max_output=ep.get("max_output") or top.get("max_completion_tokens") or 0,
        uptime_5m=ep.get("uptime_5m"),
        uptime_30m=ep.get("uptime_30m"),
        uptime_1d=ep.get("uptime_1d"),
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_table(models: list) -> None:
    """Print a compact ranked table."""
    hdr = f"{'#':<3} {'Model':<42} {'Context':>11} {'Uptime 5m':>11} {'Uptime 1d':>11} {'Status':<10} {'Provider':<15}"
    print(f"\n{'=' * len(hdr)}")
    print(f"  Free Agentic Text→Text Models on OpenRouter")
    print(f"{'=' * len(hdr)}")
    print(hdr)
    print(f"{'─' * len(hdr)}")

    for i, m in enumerate(models, 1):
        name = m.name[:40] + ".." if len(m.name) > 42 else m.name
        u5 = f"{uptime_indicator(m.uptime_5m)} {m.uptime_5m:.1f}%" if m.uptime_5m else "⚪  N/A"
        u1 = f"{uptime_indicator(m.uptime_1d)} {m.uptime_1d:.1f}%" if m.uptime_1d else "⚪  N/A"
        ctx = f"{m.context_length:,}" if m.context_length else "N/A"
        provider = m.provider[:13] + ".." if len(m.provider) > 15 else m.provider

        print(f"  {i:<3} {name:<42} {ctx:>11} {u5:>11} {u1:>11} {m.status:<10} {provider:<15}")

    print(f"{'=' * len(hdr)}")
    print("  Legend: 🟢 >=99.9% 🟡 >=99% 🟠 >=95% 🔴 <95%")


def display_json(models: list) -> None:
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(models),
        "models": [asdict(m) for m in models],
    }
    print(json.dumps(data, indent=2, default=str))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Search OpenRouter for free agentic text→text models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python search_agentic_models.py                    # top 10 with live uptime
  python search_agentic_models.py --limit 5         # top 5
  python search_agentic_models.py --json            # JSON output
  python search_agentic_models.py --no-monitor      # skip live uptime
  python search_agentic_models.py --json-file out.json
""",
    )
    parser.add_argument("--limit", type=int, default=10,
                        help="Max models to list (default: 10)")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--json-file", type=str, default=None,
                        help="Export JSON to file")
    parser.add_argument("--no-monitor", action="store_true",
                        help="Skip fetching live uptime data (faster)")
    args = parser.parse_args()

    models = search_agentic_models(
        limit=args.limit,
        monitor=not args.no_monitor,
    )

    if not models:
        sys.exit(0)

    if args.json:
        display_json(models)
    else:
        display_table(models)

    if args.json_file:
        data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(models),
            "models": [asdict(m) for m in models],
        }
        with open(args.json_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\nExported to {args.json_file}")


if __name__ == "__main__":
    main()
