# ai_scripts

Utility scripts for monitoring and discovering AI models on [OpenRouter](https://openrouter.ai).

## Scripts

### `openrouter_uptime.py`

Monitor uptime and endpoint health for specific OpenRouter models, or discover
free agentic text→text models with live endpoint data.

```bash
# Monitor specific models
python openrouter_uptime.py openrouter/owl-alpha

# Discover free agentic models
python openrouter_uptime.py --search

# Discover with live uptime data
python openrouter_uptime.py --search --monitor

# Limit results
python openrouter_uptime.py --search --monitor --limit 5
```

**Output:** Model name, status, context length, max output, provider,
quantization, and uptime across 5m / 30m / 1d windows with color indicators
(🟢 >=99.9%, 🟡 >=99%, 🟠 >=95%, 🔴 <95%).

### `openrouter_best_free.py`

Find the best free agentic model on OpenRouter for a specific use case using
composite scoring. Supports three built-in profiles with different weight
distributions and keyword bonuses.

```bash
# Default agentic profile
python openrouter_best_free.py

# Research profile (weights context + reasoning)
python openrouter_best_free.py --profile research

# Education profile (weights output + instruction-following)
python openrouter_best_free.py --profile education

# JSON output
python openrouter_best_free.py --profile research --json

# Export to file
python openrouter_best_free.py --profile education --json-file results.json

# List available profiles
python openrouter_best_free.py --list-profiles

# Override individual weights
python openrouter_best_free.py --profile research --context-weight 0.5
```

#### Profiles

| Profile | Uptime | Context | Output | Size | Bonus keywords |
|---------|--------|---------|--------|------|----------------|
| `agentic` | 40% | 25% | 20% | 15% | none |
| `research` | 30% | 35% | 15% | 20% | +5 for "reasoning", "thinking", "math", "science", "instruct" |
| `education` | 30% | 20% | 35% | 15% | +5 for "instruct", "teacher", "explain", "tutor", "learn" |

#### Scoring

Each model is scored 0–100 on four dimensions (log-scaled normalization):

- **Uptime** — average of 5m / 30m / 1d windows, penalized for missing data
- **Context** — context length in tokens
- **Output** — max completion tokens
- **Size** — parameter count extracted from model ID

Profile keyword bonuses add +5 per matching keyword in model name/description.
Weights are auto-normalized if they don't sum to 1.0.

#### Output

- Ranked table with all models, uptime indicators, and scores
- Detailed top-pick card with full score breakdown
- Runner-up comparison
- Honorable mentions (largest context, best uptime, largest output)

## Requirements

- Python 3.10+
- No external dependencies (uses only stdlib: `urllib`, `json`, `argparse`,
  `concurrent.futures`, `dataclasses`, `math`, `re`)

## Project structure

```
ai_scripts/
  README.md
  pyproject.toml
  main.py                    # entry point (if applicable)
  openrouter_uptime.py       # uptime monitor & discovery
  openrouter_best_free.py    # best model finder with profiles
```
