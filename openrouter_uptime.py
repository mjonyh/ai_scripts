#!/usr/bin/env python3
"""
OpenRouter Model Uptime Monitor & Discovery

Modes:
  1. Monitor specific models (default):
       python openrouter_uptime.py [model_id ...]

  2. Discover free agentic text→text models:
       python openrouter_uptime.py --search

  3. Discover and monitor in one shot:
       python openrouter_uptime.py --search --monitor
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

# =============================================================================
# Constants
# =============================================================================

BASE_URL = "https://openrouter.ai/api/v1/models"

DEFAULT_MODELS = [
    "openrouter/owl-alpha",
    "inclusionai/ring-2.6-1t:free",
]

AGENTIC_KEYWORDS = [
    "agent", "agentic", "tool use", "tool-use", "tool-call",
    "function calling", "function-calling", "autonomous",
    "workflow", "structured output", "json mode",
    "function call", "multi-step", "reasoning",
    "code generation", "automated",
]

UPTIME_THRESHOLDS = [
    (99.9, "🟢"),
    (99.0, "🟡"),
    (95.0, "🟠"),
]

STATUS_MAP = {0: "active", 1: "degraded", 2: "down"}

BOX_H = "─"
BOX_L = "╭"
BOX_R = "╯"
BOX_PIPE = "│"


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class UptimeInfo:
    """Consolidated uptime information for a model endpoint."""
    provider: str = "N/A"
    tag: str = ""
    status: str = "unknown"
    uptime_5m: Optional[float] = None
    uptime_30m: Optional[float] = None
    uptime_1d: Optional[float] = None
    latency: Optional[float] = None
    throughput: Optional[float] = None
    context_length: Optional[int] = None
    max_tokens: Optional[int] = None
    quantization: str = ""


@dataclass
class ModelInfo:
    """Model metadata from OpenRouter."""
    model_id: str
    name: str = ""
    context_length: int = 0
    max_completion_tokens: int = 0
    modality: str = ""
    created: int = 0
    canonical_slug: Optional[str] = None
    architecture: dict = field(default_factory=dict)
    description: str = ""


# =============================================================================
# API helpers
# =============================================================================

def fetch_json(url: str, timeout: int = 30) -> dict:
    """Fetch JSON from a URL with error handling."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_model_list() -> dict:
    """Fetch all available models from OpenRouter."""
    return fetch_json(BASE_URL)


def fetch_endpoints(canonical_slug: str) -> Optional[dict]:
    """Fetch endpoint details (including uptime) for a model."""
    try:
        return fetch_json(f"{BASE_URL}/{canonical_slug}/endpoints", timeout=15)
    except (urllib.error.HTTPError, Exception):
        return None


def resolve_canonical_slug(model_id: str, model_list: dict) -> Optional[str]:
    """Find the canonical slug for a given model ID."""
    for model in model_list.get("data", []):
        if model["id"] == model_id:
            return model.get("canonical_slug")
    return None


# =============================================================================
# Filter helpers
# =============================================================================

def is_free(model: dict) -> bool:
    """Return True if the model has zero-cost prompt and completion."""
    pricing = model.get("pricing", {})
    try:
        return float(pricing.get("prompt", "")) == 0.0 and \
               float(pricing.get("completion", "")) == 0.0
    except (ValueError, TypeError):
        return False


def is_text_to_text(model: dict) -> bool:
    """Return True if the model modality is text→text."""
    return model.get("architecture", {}).get("modality") == "text->text"


def is_agentic(model: dict) -> bool:
    """Heuristic: check if model description suggests agentic capabilities."""
    text = f"{model.get('name', '')} {model.get('description', '')}".lower()
    return any(kw in text for kw in AGENTIC_KEYWORDS)


# =============================================================================
# Uptime helpers
# =============================================================================

def get_uptime_indicator(pct: Optional[float]) -> str:
    """Return a colored circle emoji based on uptime percentage."""
    if pct is None:
        return "⚪"
    for threshold, emoji in UPTIME_THRESHOLDS:
        if pct >= threshold:
            return emoji
    return "🔴"


def parse_uptime(endpoints_data: dict) -> Optional[UptimeInfo]:
    """Return UptimeInfo for the best endpoint, or None if unavailable."""
    endpoints = endpoints_data.get("data", {}).get("endpoints", [])
    if not endpoints:
        return None

    best = max(endpoints, key=lambda e: e.get("uptime_last_1d", 0) or 0)
    return UptimeInfo(
        provider=best.get("provider_name", "N/A"),
        tag=best.get("tag", ""),
        status=STATUS_MAP.get(best.get("status", -1), "unknown"),
        uptime_5m=best.get("uptime_last_5m"),
        uptime_30m=best.get("uptime_last_30m"),
        uptime_1d=best.get("uptime_last_1d"),
        latency=best.get("latency_last_30m"),
        throughput=best.get("throughput_last_30m"),
        context_length=best.get("context_length"),
        max_tokens=best.get("max_completion_tokens"),
        quantization=best.get("quantization", ""),
    )


def parse_model(raw: dict, model_list: dict) -> ModelInfo:
    """Parse a raw model dict into a ModelInfo."""
    tp = raw.get("top_provider", {})
    return ModelInfo(
        model_id=raw["id"],
        name=raw.get("name", raw["id"]),
        context_length=raw.get("context_length", 0),
        max_completion_tokens=tp.get("max_completion_tokens", 0),
        modality=raw.get("architecture", {}).get("modality", ""),
        created=raw.get("created", 0),
        canonical_slug=resolve_canonical_slug(raw["id"], model_list),
        architecture=raw.get("architecture", {}),
        description=raw.get("description", ""),
    )


# =============================================================================
# Display helpers
# =============================================================================

def fmt_uptime(label: str, value: Optional[float]) -> str:
    """Format a single uptime row with indicator emoji."""
    if value is not None:
        return f"  {BOX_PIPE}  {label:<13s} {get_uptime_indicator(value)}  : {value:.2f}%"
    return f"  {BOX_PIPE}  {label:<13s} ⚪  : N/A"


def fmt_opt(label: str, value, fmt_str: str = "{}") -> str:
    """Format an optional field, returning empty string if value is falsy."""
    if value:
        return f"  {BOX_PIPE}  {label:<14s}: {fmt_str.format(value)}"
    return ""


def display_uptime_box(model_id: str, endpoint_data: dict) -> None:
    """Pretty-print uptime info for a model using box-drawing characters."""
    endpoints = endpoint_data.get("data", {}).get("endpoints", [])
    if not endpoints:
        print(f"\n  {model_id}\n    Status  : ⚪ No active endpoints")
        return

    for ep in endpoints:
        name = ep.get("model_name", model_id)
        status_text = STATUS_MAP.get(ep.get("status", -1), f"unknown ({ep.get('status')})")
        uptime_5m = ep.get("uptime_last_5m")
        uptime_30m = ep.get("uptime_last_30m")
        uptime_1d = ep.get("uptime_last_1d")
        latency = ep.get("latency_last_30m")
        throughput = ep.get("throughput_last_30m")
        ctx_len = ep.get("context_length")
        max_tok = ep.get("max_completion_tokens")
        quant = ep.get("quantization", "")
        provider = ep.get("provider_name", "Unknown")
        tag = ep.get("tag", "")

        width = 50
        print(f"\n  {BOX_L}{BOX_H}─ {name} ({model_id})")
        print(f"  {BOX_PIPE}  {'Provider':<14s}: {provider} ({tag})")
        print(f"  {BOX_PIPE}  {'Status':<14s}: {status_text}")
        if ctx_len:
            print(f"  {BOX_PIPE}  {'Context Len':<14s}: {ctx_len:,} tokens")
        if max_tok:
            print(f"  {BOX_PIPE}  {'Max Output':<14s}: {max_tok:,} tokens")
        print(f"  {BOX_PIPE}  {'Quantization':<14s}: {quant}")
        print(f"  {BOX_PIPE}")
        print(fmt_uptime("Uptime (5m)", uptime_5m))
        print(fmt_uptime("Uptime (30m)", uptime_30m))
        print(fmt_uptime("Uptime (1d)", uptime_1d))
        if latency is not None:
            print(f"  {BOX_PIPE}  {'Latency (30m)':<14s}: {latency:.0f}ms")
        if throughput is not None:
            print(f"  {BOX_PIPE}  {'Throughput':<14s}: {throughput:.0f} tok/s")
        print(f"  {BOX_R}{BOX_H * width}")


def display_model_summary(model: ModelInfo, uptime: Optional[UptimeInfo], index: int) -> None:
    """Pretty-print a single model in summary format."""
    print(f"\n  {index}. {model.name}")
    print(f"     ID         : {model.model_id}")
    print(f"     Modality   : {model.modality}")
    print(f"     Status     : {uptime.status if uptime else 'no endpoints'}")
    print(f"     Context    : {model.context_length:,} tokens")
    if model.max_completion_tokens:
        print(f"     Max Output : {model.max_completion_tokens:,} tokens")
    if uptime:
        print(f"     Provider   : {uptime.provider}")
        print(f"     Quant      : {uptime.quantization}")
        print(f"     Uptime 5m  {get_uptime_indicator(uptime.uptime_5m)} : "
              f"{uptime.uptime_5m:.2f}%" if uptime.uptime_5m is not None else "     Uptime 5m  ⚪ : N/A")
        print(f"     Uptime 30m    : "
              f"{uptime.uptime_30m:.2f}%" if uptime.uptime_30m is not None else "     Uptime 30m    : N/A")
        print(f"     Uptime 1d     : "
              f"{uptime.uptime_1d:.2f}%" if uptime.uptime_1d is not None else "     Uptime 1d     : N/A")


def print_section(title: str, width: int = 60) -> None:
    """Print a section header."""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print("=" * width)


def print_legend() -> None:
    """Print the uptime emoji legend."""
    print("\n  Legend: 🟢 >=99.9%  🟡 >=99%  🟠 >=95%  🔴 <95%")


# =============================================================================
# Core logic
# =============================================================================

def discover_agentic_models(limit: int = 10, monitor: bool = True) -> list:
    """
    Discover free, agentic, text→text models on OpenRouter.
    Optionally fetch live uptime for each.
    Returns list of (ModelInfo, Optional[UptimeInfo]) tuples.
    """
    print("Fetching full model list from OpenRouter...")
    try:
        full_list = fetch_model_list()
    except Exception as e:
        print(f"Error fetching model list: {e}")
        sys.exit(1)

    all_models = full_list.get("data", [])
    print(f"  Total models available: {len(all_models)}")

    # Filter: free + text→text + agentic
    filtered = [m for m in all_models
                if is_text_to_text(m) and is_free(m) and is_agentic(m)]
    print(f"  Free + text→text + agentic matches: {len(filtered)}")

    # Sort by recency, limit
    filtered.sort(key=lambda m: m.get("created", 0), reverse=True)
    candidates = filtered[:limit]

    # Fetch uptime in parallel
    if monitor:
        print(f"\nFetching live endpoint data for {len(candidates)} models...\n")
        results: list = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            future_to_model = {}
            for raw in candidates:
                mi = parse_model(raw, full_list)
                if mi.canonical_slug:
                    future_to_model[pool.submit(fetch_endpoints, mi.canonical_slug)] = mi
                else:
                    results.append((mi, None))

            for future in as_completed(future_to_model):
                mi = future_to_model[future]
                try:
                    ep_data = future.result()
                except Exception as e:
                    print(f"  Error fetching endpoints for {mi.model_id}: {e}")
                    results.append((mi, None))
                    continue
                results.append((mi, parse_uptime(ep_data) if ep_data else None))

        return results
    else:
        return [(parse_model(raw, full_list), None) for raw in candidates]


def monitor_specific_models(model_ids: list) -> None:
    """Monitor user-specified models and display their uptime."""
    print("Fetching model list...")
    try:
        model_list = fetch_model_list()
    except Exception as e:
        print(f"Error fetching model list: {e}")
        sys.exit(1)

    print(f"  Found {len(model_list.get('data', []))} models on OpenRouter.\n")

    for model_id in model_ids:
        slug = resolve_canonical_slug(model_id, model_list)
        if not slug:
            print(f"  ⚠  Model '{model_id}' not found in registry.")
            continue
        try:
            endpoint_data = fetch_endpoints(slug)
            if endpoint_data is None:
                print(f"  ⚠  No endpoint data for '{model_id}'.")
                continue
            display_uptime_box(model_id, endpoint_data)
        except Exception as e:
            print(f"  ❌ Error fetching data for '{model_id}': {e}")


# =============================================================================
# Entry point
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="OpenRouter Model Uptime Monitor & Discovery"
    )
    parser.add_argument(
        "models", nargs="*",
        help="Specific model IDs to monitor (e.g. openrouter/owl-alpha)"
    )
    parser.add_argument(
        "--search", action="store_true",
        help="Discover free agentic text→text models"
    )
    parser.add_argument(
        "--monitor", action="store_true",
        help="Also fetch live uptime during discovery (slower)"
    )
    parser.add_argument(
        "--limit", type=int, default=10,
        help="Max models to show in discovery (default: 10)"
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print_section("OpenRouter Model Uptime Monitor & Discovery")
    print(f"  Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    if args.search:
        results = discover_agentic_models(limit=args.limit, monitor=args.monitor)
        print_section(f"Top {min(args.limit, len(results))} Free Agentic Text→Text Models")
        for i, (model, uptime) in enumerate(results, 1):
            display_model_summary(model, uptime, i)
        print_section("")
        print_legend()
        print("=" * 60)

    elif args.models:
        monitor_specific_models(args.models)
        print_section("")
        print_legend()
        print("=" * 60)

    else:
        print("\nNo mode specified. Use --search to discover models or pass model IDs.\n")
        print("Examples:")
        print("  python openrouter_uptime.py --search --monitor")
        print("  python openrouter_uptime.py openrouter/owl-alpha")
        print("  python openrouter_uptime.py --search --limit 5")


if __name__ == "__main__":
    main()
