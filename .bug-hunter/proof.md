# Bug Hunter Proof

## Before

```python
if cleaned.upper().startswith(PASS_TOKEN) and len(cleaned) <= 8:
```

## After

```python
first_line = cleaned.split("\n", 1)[0].strip().upper()
if first_line == PASS_TOKEN:
```

## Why

The len check cap of 8 chars means any critic reply starting with PASS followed by a newline or commentary is treated as a rewrite. The full PASS+... string is delivered to the user instead of the original draft.

## Evidence

| Field | Value |
|---|---|
| PR | [17](https://github.com/vibeforge1111/spark-character/pull/17) |
| Repo | vibeforge1111/spark-character |
| Severity | high |
| Files changed | `src/spark_character/critic.py` |
| Branch | `fix/critic-interpret-pass-trailing-content` |
| Validated | pass (0 errors, 0 warnings) |