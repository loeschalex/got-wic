# got-wic

Strategy simulators and analysis tools for **Game of Thrones: Winter is Coming**.

## Features

- **Game model** — data classes for objectives, dragons, treasures, and army allocations across match phases
- **Battle simulator** — minute-by-minute simulation of two players' allocations, producing score breakdowns and timelines
- **Opponent generator** — heuristic opponent strategies for testing allocations against
- **Grid search optimizer** — finds the best allocation against a generated opponent by sweeping army distributions

## Quickstart

```bash
uv sync
```

### Python API

```python
from got_wic import default_config, optimize

result = optimize(default_config(), total_armies=1000, step_pct=25)
print(f"Best score: {result.score_a} vs {result.score_b}")
print(result.allocation)
```

### Interactive notebook

```bash
uv run marimo edit notebooks/sow-7th-ann-se.py
```

## Project structure

```
src/got_wic/
  model.py      # GameConfig, Objective, Dragon, Allocation
  simulate.py   # simulate() — runs a full match
  opponent.py   # generate_opponent() — heuristic opponent
  optimize.py   # optimize() — grid search over allocations
notebooks/
  sow-7th-ann-se.py   # Marimo notebook with interactive optimizer UI
tests/
  test_model.py, test_simulate.py, test_opponent.py,
  test_optimize.py, test_integration.py
```

## Development

```bash
uv run pytest
uv run ruff check .
```

Requires Python 3.13+.
