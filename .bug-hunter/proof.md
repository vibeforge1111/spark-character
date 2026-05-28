# Bug Hunter Proof

## Before

```python
EM_DASH = "—"
p1 = 0.0 if EM_DASH in text else 1.0
```

## After

```python
from .output_sanitizer import EM_DASH_FAMILY
p1 = 0.0 if any(ch in text for ch in EM_DASH_FAMILY) else 1.0
```

## Why

P1 only caught the true em-dash (U+2014). output_sanitizer treats 5 dash-family chars as violations. En-dashes silently passed P1 scoring but were rewritten at delivery — evolution never observed the blind spot.

## Evidence

| Field | Value |
|---|---|
| PR | [19](https://github.com/vibeforge1111/spark-character/pull/19) |
| Repo | vibeforge1111/spark-character |
| Severity | high |
| Files changed | `src/spark_character/scoring.py` |
| Branch | `fix/score-persona-p1-em-dash-family` |
| Validated | pass (0 errors, 0 warnings) |