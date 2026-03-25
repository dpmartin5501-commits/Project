# Dependency Update Plan

**Date:** 2026-03-25
**Python:** 3.12.3
**Baseline:** 44/44 tests passing

---

## Current State

| Package | Declared Floor | Installed | Latest on PyPI | CVEs |
|---------|---------------|-----------|---------------|------|
| ccxt | >=4.0.0 | 4.5.45 | 4.5.45 | None |
| pandas | >=2.0.0 | 3.0.1 | 3.0.1 | None |
| numpy | >=1.24.0 | 2.4.3 | 2.4.3 | None |
| ta | >=0.11.0 | 0.11.0 | 0.11.0 | None |
| requests | >=2.31.0 | 2.32.5 | 2.32.5 | None |
| beautifulsoup4 | >=4.12.0 | 4.14.3 | 4.14.3 | None |
| rich | >=13.0.0 | 14.3.3 | 14.3.3 | None |
| click | >=8.1.0 | 8.3.1 | 8.3.1 | None |
| aiohttp | >=3.9.0 | 3.13.3 | 3.13.3 | None |

**All direct dependencies are already at the latest PyPI versions.** No outdated packages to upgrade.

### Vulnerability Scan (`pip-audit`)

No CVEs were found in direct project dependencies. Vulnerabilities exist only in system-level packages unrelated to this project (ansible, cryptography, jinja2, pip, setuptools, wheel).

---

## Issues Found

### 1. `aiohttp` is declared but never imported

`aiohttp` appears in `requirements.txt` and `setup.py` but is not imported anywhere in the codebase. The project uses only synchronous ccxt APIs (`fetch_ohlcv`). Since ccxt lists `aiohttp` as its own dependency, it is always installed transitively — the explicit declaration is redundant.

**Risk:** None.

### 2. Version floors are too loose

The declared minimums allow installing versions with known incompatibilities:

- `numpy>=1.24.0` admits numpy 1.x, but the project runs and is tested on numpy 2.x. NumPy 2.0 removed many aliases (`np.float_`, `np.int_`, etc.) and changed type-promotion rules. Installing numpy 1.x in a fresh environment could silently work or silently break depending on which `ta` version resolves.
- `pandas>=2.0.0` admits early pandas 2.0 releases, but the project runs on pandas 3.0.1 which enforces Copy-on-Write and changes string dtype inference. A fresh install could land on pandas 2.0.0 or 3.0.1 — very different runtime behavior.
- `ccxt>=4.0.0` is an extremely fast-moving package (multiple releases per week). A floor of 4.0.0 is ~1 year old and exchange APIs may have changed.

**Risk of tightening:** Low. We are codifying what is already tested and working.

### 3. `pytest` is not declared as a dependency

`pytest` is used in `tests/test_backtester.py` but not listed in `requirements.txt`, `setup.py`, or any dev-requirements file.

**Risk:** None.

### 4. `ta` library is low-maintenance

The `ta` library's latest release (0.11.0) was November 2023. The last repository push was July 2024. There are 151 open issues. The library depends on `pandas` and `numpy` but does not declare upper bounds — future pandas/numpy releases could break it without warning.

The project uses 12 indicator classes from `ta.momentum`, `ta.trend`, and `ta.volatility`. There is no newer version to upgrade to; 0.11.0 is the latest.

**Risk:** Medium-term maintenance risk. No action needed now, but consider monitoring for a maintained fork (e.g., `pandas-ta`, `ta-lib`) if `ta` stops working with future pandas/numpy versions.

### 5. Deprecation warnings (2920 per test run)

All tests produce `DeprecationWarning: Bitwise inversion '~' on bool is deprecated` originating from `pandas/core/internals/blocks.py`. This is upstream pandas internal code, not project code. It is expected to be fixed in a future pandas release.

**Risk:** None for now. These warnings will disappear in a future pandas patch.

---

## Proposed Update Plan

### Tier 1 — Safe, no-risk changes

These changes codify what is already true and tested. No behavior change.

| Change | Files | Rationale |
|--------|-------|-----------|
| Remove `aiohttp` from direct deps | `requirements.txt`, `setup.py` | Unused; pulled in transitively by ccxt |
| Raise `numpy` floor to `>=2.0.0` | `requirements.txt`, `setup.py` | Project is tested on 2.x; numpy 1.x has removed APIs |
| Raise `pandas` floor to `>=2.2.0` | `requirements.txt`, `setup.py` | 2.2 is the earliest 2.x with Copy-on-Write warnings; project runs on 3.0.1 |
| Raise `ccxt` floor to `>=4.4.0` | `requirements.txt`, `setup.py` | Reduce exposure to stale exchange API changes |
| Raise `requests` floor to `>=2.32.0` | `requirements.txt`, `setup.py` | 2.32 is the current stable branch |
| Raise `beautifulsoup4` floor to `>=4.13.0` | `requirements.txt`, `setup.py` | Align with current major |
| Raise `rich` floor to `>=13.7.0` | `requirements.txt`, `setup.py` | Align with current major |
| Raise `click` floor to `>=8.1.7` | `requirements.txt`, `setup.py` | 8.1.7 is the last 8.1.x bugfix |
| Add `pytest>=8.0.0` to a new `requirements-dev.txt` | New file | Declare the test dependency |

### Tier 2 — Low-risk, recommended

| Change | Files | Rationale | Breaking risk |
|--------|-------|-----------|---------------|
| Pin upper bounds on `ta<1.0` | `requirements.txt`, `setup.py` | Protect against future API changes from an unmaintained library | None — 0.11.0 is the only version |
| Add `python_requires=">=3.10,<3.14"` | `setup.py` | Already declares `>=3.10`; add ceiling to prevent untested runtime | Low |

### Tier 3 — Monitor only (no action now)

| Item | Notes |
|------|-------|
| `ta` library maintenance | Watch for breakage with future pandas/numpy. Consider `pandas-ta` or `ta-lib` as alternatives if the library goes fully unmaintained. |
| pandas 4.x | When released, will likely require migration. Current code is pandas-3.x compatible. |
| numpy 3.x | Not yet released. Monitor for changes to `np.random` legacy API. |

---

## What This Plan Does NOT Do

- **No package upgrades** — all packages are already at their latest versions.
- **No code changes** — the plan only adjusts dependency metadata.
- **No upper-bound pins on fast-moving packages** (ccxt, pandas, numpy) — upper pins on these would create maintenance burden and block security patches.

---

## Verification

After applying Tier 1 changes, run:

```bash
python3 -m pytest tests/ -v
```

Expected result: 44/44 tests passing (same as baseline).
