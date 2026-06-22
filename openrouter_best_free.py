#!/usr/bin/env python3
"""
OpenRouter Best Free Model Finder

Finds and ranks the best free text→text models on OpenRouter for different
use cases using a composite scoring system.

Profiles:
  agentic   — Best all-around agentic model (default)
                Weights: uptime 40%, context 25%, output 20%, size 15%
  research  — Best for academic research: reading papers, literature review,
                summarization, reasoning over long documents.
                Weights: context 35%, uptime 30%, size 20%, output 15%
                Bonus: prefers models with "reasoning", "thinking", "math",
                "science" in name/description.
  education — Best for generating educational materials: lecture notes, syllabi,
                assessments, explanations. Needs large output and reliability.
                Weights: output 35%, uptime 30%, context 20%, size 15%
                Bonus: prefers models with "instruct", "chat", "teacher",
                "explain" in name/description.

Usage:
  python openrouter_best_free.py                          # default agentic profile
  python openrouter_best_free.py --profile research       # research profile
  python openrouter_best_free.py --profile education      # education profile
  python openrouter_best_free.py --profile research --json
  python openrouter_best_free.py --uptime-weight 0.5      # custom weights
"""

import argparse
import json
import math
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

# =============================================================================
# Constants
# =============================================================================

BASE_URL = "https://openrouter.ai/api/v1/models"

AGENTIC_KEYWORDS = [
    "agent", "agentic", "tool use", "tool-use", "tool-call",
    "function calling", "function-calling", "autonomous",
    "workflow", "structured output", "json mode",
    "function call", "multi-step", "reasoning",
    "code generation", "automated",
]

RESEARCH_KEYWORDS = [
    "reasoning", "thinking", "math", "science", "research",
    "analytical", "logic", "theorem", "proof", "stem",
    "deep", "instruct",
]

EDUCATION_KEYWORDS = [
    "instruct", "chat", "teacher", "explain", "tutor",
    "education", "curriculum", "lesson", "pedagogy",
    "student", "learn", "guide",
]

UPTIME_THRESHOLDS = [
    (99.9, "🟢"),
    (99.0, "🟡"),
    (95.0, "🟠"),
]

STATUS_MAP = {0: "active", 1: "degraded", 2: "down"}

# Profile definitions: (uptime, context, output, size)
PROFILES = {
    "agentic": {
        "weights": (0.40, 0.25, 0.20, 0.15),
        "label": "Agentic (general purpose)",
        "bonus_keywords": [],
        "bonus_score": 0,
    },
    "research": {
        "weights": (0.30, 0.35, 0.15, 0.20),
        "label": "Research (papers, reasoning, long context)",
        "bonus_keywords": RESEARCH_KEYWORDS,
        "bonus_score": 5.0,
    },
    "education": {
        "weights": (0.30, 0.20, 0.35, 0.15),
        "label": "Education (lectures, materials, large output)",
        "bonus_keywords": EDUCATION_KEYWORDS,
        "bonus_score": 5.0,
    },
}


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class ModelResult:
    """Complete scoring result for a single model."""
    rank: int = 0
    model_id: str = ""
    name: str = ""
    status: str = "unknown"
    provider: str = "N/A"
    quantization: str = ""
    uptime_5m: Optional[float] = None
    uptime_30m: Optional[float] = None
    uptime_1d: Optional[float] = None
    uptime_avg: float = 0.0
    context_length: int = 0
    max_output: int = 0
    param_count: float = 0.0
    bonus: float = 0.0
    uptime_score: float = 0.0
    context_score: float = 0.0
    output_score: float = 0.0
    size_score: float = 0.0
    total_score: float = 0.0


@dataclass
class Weights:
    uptime: float = 0.40
    context: float = 0.25
    output: float = 0.20
    size: float = 0.15


# =============================================================================
# API helpers
# =============================================================================

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


# =============================================================================
# Filter helpers
# =============================================================================

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


def keyword_bonus(model: dict, keywords: list) -> float:
    """Return a bonus score if model name/description matches keywords."""
    if not keywords:
        return 0.0
    text = f"{model.get('name', '')} {model.get('description', '')}".lower()
    matches = sum(1 for kw in keywords if kw in text)
    return float(matches)


# =============================================================================
# Parsing helpers
# =============================================================================

def extract_param_count(model_id: str) -> float:
    """Extract parameter count in billions from model ID string."""
    for pat in [r'(\d+\.?\d*)\s*b(?:illion)?\b', r'(\d+\.?\d*)\s*m(?:illion)?\b']:
        m = re.search(pat, model_id, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if 'm' in pat:
                val /= 1000
            return val
    return 0.0


def parse_uptime(endpoints_data: Optional[dict]) -> tuple:
    """Return (status, provider, quant, uptime_5m, uptime_30m, uptime_1d)."""
    if not endpoints_data:
        return "no endpoints", "N/A", "", None, None, None
    endpoints = endpoints_data.get("data", {}).get("endpoints", [])
    if not endpoints:
        return "no endpoints", "N/A", "", None, None, None
    best = max(endpoints, key=lambda e: e.get("uptime_last_1d", 0) or 0)
    return (
        STATUS_MAP.get(best.get("status", -1), "unknown"),
        best.get("provider_name", "N/A"),
        best.get("quantization", ""),
        best.get("uptime_last_5m"),
        best.get("uptime_last_30m"),
        best.get("uptime_last_1d"),
    )


# =============================================================================
# Scoring
# =============================================================================

def _log_normalize(value: float, max_value: float) -> float:
    """Log-scale normalization to 0-100."""
    if value <= 0 or max_value <= 0:
        return 0.0
    return (math.log1p(value) / math.log1p(max_value)) * 100.0


def score_models(models: list, weights: Weights, bonus_keywords: list,
                 bonus_score: float) -> list:
    """Score and rank all models. Returns sorted list (best first)."""
    if not models:
        return []

    max_ctx = max((m.context_length for m in models), default=1)
    max_out = max((m.max_output for m in models), default=1)
    max_params = max((m.param_count for m in models), default=1)

    for m in models:
        # Uptime
        uptimes = [u for u in [m.uptime_5m, m.uptime_30m, m.uptime_1d] if u is not None]
        if uptimes:
            m.uptime_avg = sum(uptimes) / len(uptimes)
            m.uptime_score = m.uptime_avg * (len(uptimes) / 3.0)
        else:
            m.uptime_avg = 0.0
            m.uptime_score = 0.0

        m.context_score = _log_normalize(m.context_length, max_ctx)
        m.output_score = _log_normalize(m.max_output, max_out)
        m.size_score = _log_normalize(m.param_count, max_params)

        # Profile keyword bonus
        m.bonus = m.bonus * bonus_score if m.bonus > 0 else 0.0

        m.total_score = (
            weights.uptime * m.uptime_score +
            weights.context * m.context_score +
            weights.output * m.output_score +
            weights.size * m.size_score +
            m.bonus
        )

    models.sort(key=lambda m: m.total_score, reverse=True)
    for i, m in enumerate(models, 1):
        m.rank = i
    return models


# =============================================================================
# Discovery
# =============================================================================

def discover_and_score(weights: Weights, bonus_keywords: list,
                       bonus_score: float) -> list:
    """Full pipeline: fetch, filter, enrich, score, rank."""
    print("Fetching model list from OpenRouter...")
    try:
        full_list = fetch_model_list()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    all_models = full_list.get("data", [])
    print(f"  Total models: {len(all_models)}")

    filtered = [m for m in all_models
                if is_text_to_text(m) and is_free(m) and is_agentic(m)]
    print(f"  Free + agentic + text→text: {len(filtered)}")

    if not filtered:
        print("No matching models found.")
        return []

    print("Fetching endpoint data...\n")
    results = []

    def process_model(raw: dict) -> ModelResult:
        top = raw.get("top_provider", {})
        mi = ModelResult(
            model_id=raw["id"],
            name=raw.get("name", raw["id"]),
            context_length=raw.get("context_length", 0) or 0,
            max_output=top.get("max_completion_tokens") or 0,
            param_count=extract_param_count(raw["id"]),
            bonus=keyword_bonus(raw, bonus_keywords),
        )
        slug = resolve_canonical_slug(raw["id"], full_list)
        ep_data = fetch_endpoints(slug) if slug else None
        status, provider, quant, u5, u30, u1 = parse_uptime(ep_data)
        mi.status = status
        mi.provider = provider
        mi.quantization = quant
        mi.uptime_5m = u5
        mi.uptime_30m = u30
        mi.uptime_1d = u1
        return mi

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(process_model, m) for m in filtered]
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"  Warning: {e}", file=sys.stderr)

    return score_models(results, weights, bonus_keywords, bonus_score)


# =============================================================================
# Display
# =============================================================================

def uptime_indicator(pct: Optional[float]) -> str:
    if pct is None:
        return "⚪"
    for threshold, emoji in UPTIME_THRESHOLDS:
        if pct >= threshold:
            return emoji
    return "🔴"


def fmt_pct(v: Optional[float]) -> str:
    return f"{v:.2f}%" if v is not None else "N/A"


def fmt_int(v: int) -> str:
    return f"{v:,}" if v > 0 else "N/A"


def display_table(results: list, profile_label: str) -> None:
    """Print a ranked table of all models."""
    print(f"\n{'=' * 115}")
    print(f"  Profile: {profile_label}")
    print(f"  {'Rank':<5}{'Model':<42}{'Status':<10}{'Uptime 5m':<12}"
          f"{'Uptime 1d':<12}{'Context':<12}{'Max Out':<10}{'Params':<8}{'Score':<8}")
    print(f"{'─' * 115}")

    for m in results:
        short_name = m.name[:40] + ".." if len(m.name) > 42 else m.name
        ind = uptime_indicator(m.uptime_5m)
        bonus_tag = " ★" if m.bonus > 0 else ""
        print(f"  {m.rank:<5}{short_name:<42}{m.status:<10}"
              f"{ind} {fmt_pct(m.uptime_5m):<8}{fmt_pct(m.uptime_1d):<12}"
              f"{fmt_int(m.context_length):<12}{fmt_int(m.max_output):<10}"
              f"{m.param_count:.0f}B{'':<5}{m.total_score:.1f}{bonus_tag}")

    print(f"{'=' * 115}")
    print("  ★ = profile keyword bonus applied")


def display_top_pick(results: list, profile_label: str, weights: Weights) -> None:
    """Print detailed recommendation for the top model."""
    if not results:
        return

    top = results[0]
    print(f"\n{'━' * 60}")
    print(f"  🏆  TOP PICK [{profile_label}]: {top.name}")
    print(f"{'━' * 60}")
    print(f"  Model ID    : {top.model_id}")
    print(f"  Status      : {top.status}")
    print(f"  Provider    : {top.provider}")
    print(f"  Quant       : {top.quantization}")
    print(f"  Parameters  : {top.param_count:.0f}B")
    print(f"  Context     : {fmt_int(top.context_length)} tokens")
    print(f"  Max Output  : {fmt_int(top.max_output)} tokens")
    print(f"  Uptime 5m   : {uptime_indicator(top.uptime_5m)} {fmt_pct(top.uptime_5m)}")
    print(f"  Uptime 30m  : {uptime_indicator(top.uptime_30m)} {fmt_pct(top.uptime_30m)}")
    print(f"  Uptime 1d   : {uptime_indicator(top.uptime_1d)} {fmt_pct(top.uptime_1d)}")
    print(f"  Avg Uptime  : {top.uptime_avg:.2f}%")
    print(f"")
    print(f"  Score Breakdown:")
    print(f"    Uptime  ({weights.uptime:.0%}) : {top.uptime_score:.1f}")
    print(f"    Context ({weights.context:.0%}) : {top.context_score:.1f}")
    print(f"    Output  ({weights.output:.0%}) : {top.output_score:.1f}")
    print(f"    Size    ({weights.size:.0%}) : {top.size_score:.1f}")
    if top.bonus > 0:
        print(f"    Profile bonus  : +{top.bonus:.1f}")
    print(f"    ─────────────────")
    print(f"    TOTAL          : {top.total_score:.1f}")
    print(f"{'━' * 60}")

    if len(results) >= 2:
        second = results[1]
        diff = top.total_score - second.total_score
        print(f"\n  Runner-up: {second.name} (score: {second.total_score:.1f}, -{diff:.1f})")

    best_ctx = max(results, key=lambda m: m.context_length)
    best_uptime = max(results, key=lambda m: m.uptime_avg)
    best_output = max(results, key=lambda m: m.max_output)

    print(f"\n  Also notable:")
    if best_ctx.model_id != top.model_id:
        print(f"    Largest context : {best_ctx.name} ({fmt_int(best_ctx.context_length)})")
    if best_uptime.model_id != top.model_id:
        print(f"    Best uptime     : {best_uptime.name} ({best_uptime.uptime_avg:.2f}%)")
    if best_output.model_id != top.model_id:
        print(f"    Largest output  : {best_output.name} ({fmt_int(best_output.max_output)})")


def display_profile_info() -> None:
    """Print available profiles and their weight distributions."""
    print("\n  Available profiles:")
    for name, cfg in PROFILES.items():
        w = cfg["weights"]
        print(f"    {name:<12s} — {cfg['label']}")
        print(f"                uptime={w[0]:.0%}  context={w[1]:.0%}  "
              f"output={w[2]:.0%}  size={w[3]:.0%}", end="")
        if cfg["bonus_keywords"]:
            print(f"  bonus=+{cfg['bonus_score']:.0f} for keywords")
        else:
            print()
    print()


def export_json(results: list, filepath: str) -> None:
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(results),
        "models": [asdict(m) for m in results],
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\n  Exported to {filepath}")


# =============================================================================
# Entry point
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find the best free AI model on OpenRouter for your use case",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Profiles:
  agentic    Best all-around agentic model (default)
  research   Best for academic research, papers, reasoning
  education  Best for lectures, materials, large output generation

Examples:
  python openrouter_best_free.py --profile research
  python openrouter_best_free.py --profile education --json
  python openrouter_best_free.py --list-profiles
        """,
    )
    parser.add_argument(
        "--profile", type=str, default="agentic",
        choices=list(PROFILES.keys()),
        help="Scoring profile (default: agentic)"
    )
    parser.add_argument(
        "--list-profiles", action="store_true",
        help="List available profiles and exit"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON to stdout"
    )
    parser.add_argument(
        "--json-file", type=str, default=None,
        help="Export results as JSON to file"
    )
    parser.add_argument(
        "--uptime-weight", type=float, default=None,
        help="Override uptime weight"
    )
    parser.add_argument(
        "--context-weight", type=float, default=None,
        help="Override context weight"
    )
    parser.add_argument(
        "--output-weight", type=float, default=None,
        help="Override output weight"
    )
    parser.add_argument(
        "--size-weight", type=float, default=None,
        help="Override size weight"
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_profiles:
        display_profile_info()
        sys.exit(0)

    # Build weights from profile + overrides
    profile_cfg = PROFILES[args.profile]
    w = profile_cfg["weights"]
    weights = Weights(
        uptime=args.uptime_weight if args.uptime_weight is not None else w[0],
        context=args.context_weight if args.context_weight is not None else w[1],
        output=args.output_weight if args.output_weight is not None else w[2],
        size=args.size_weight if args.size_weight is not None else w[3],
    )

    # Normalize if custom weights don't sum to 1.0
    total_w = weights.uptime + weights.context + weights.output + weights.size
    if abs(total_w - 1.0) > 0.01:
        print(f"Weights sum to {total_w:.2f}, normalizing to 1.0")
        weights.uptime /= total_w
        weights.context /= total_w
        weights.output /= total_w
        weights.size /= total_w

    bonus_keywords = profile_cfg["bonus_keywords"]
    bonus_score = profile_cfg["bonus_score"]
    profile_label = profile_cfg["label"]

    print("=" * 60)
    print("  OpenRouter Best Free Model Finder")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Profile : {profile_label}")
    print(f"  Weights : uptime={weights.uptime:.0%}  context={weights.context:.0%}  "
          f"output={weights.output:.0%}  size={weights.size:.0%}")
    if bonus_keywords:
        print(f"  Bonus   : +{bonus_score:.0f} pts for keyword match")
    print("=" * 60)

    results = discover_and_score(weights, bonus_keywords, bonus_score)

    if not results:
        print("No models found.")
        sys.exit(0)

    if args.json:
        data = {
            "profile": args.profile,
            "profile_label": profile_label,
            "weights": asdict(weights),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(results),
            "models": [asdict(m) for m in results],
        }
        print(json.dumps(data, indent=2, default=str))
    else:
        display_table(results, profile_label)
        display_top_pick(results, profile_label, weights)

    if args.json_file:
        export_json(results, args.json_file)


if __name__ == "__main__":
    main()
