# Contributing to DAG Brain

DAG Brain is created and maintained by [SlightlyHappy](https://github.com/SlightlyHappy). Contributions are welcome, from software developers and from epidemiologists, economists, psychologists, statisticians and other researchers. This page explains how contributions are made and reviewed.

## How contributions work

- **Every change goes through a pull request.** Nobody pushes to `main` directly. Each PR needs an approving review from the maintainer, as listed in [`.github/CODEOWNERS`](.github/CODEOWNERS), and passing tests before it can be merged.
- **The maintainer decides what is merged** and sets the direction of the project. For anything bigger than a small fix, open an issue first, so we can agree on the approach before you spend time on it.
- **Sign the Contributor License Agreement.** Before your first PR can be merged you must agree to the [CLA](CLA.md), by ticking the box in the PR description or through the CLA bot if it comments on your PR. You keep the copyright to your work; the CLA lets the project keep a single steward and evolve its licensing.
- **License and name.** The code is licensed under the [GNU AGPL v3](LICENSE) with the additional terms in [NOTICE](NOTICE). The name "DAG Brain" and the logo are not covered by the license (see [TRADEMARKS.md](TRADEMARKS.md)). Do not remove copyright headers or the attribution in the About dialog.

## Workflow

```bash
# 1. Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/DAG_Brain.git
cd DAG_Brain
git remote add upstream https://github.com/SlightlyHappy/DAG_Brain.git

# 2. Create a branch for your change
git checkout -b fix/short-description

# 3. Install and run the tests
pip install -r requirements-dev.txt
pytest

# 4. Push to your fork and open a pull request against SlightlyHappy/DAG_Brain:main
```

Keep pull requests focused: one change per PR, with tests. New source files start with the same copyright header as the existing ones:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 SlightlyHappy. DAG Brain: https://github.com/SlightlyHappy/DAG_Brain
```

## What a PR must include

1. **Tests.** Any change to `backend/causal_core.py`, `backend/causal_engine.py`, `backend/data_analysis.py`, `backend/validation.py` or `backend/dag_eval.py` needs tests in `tests/`. Graph algorithms should be checked against an independent reference (see `tests/reference.py`) or against simulated data with a known answer.
2. **No regression in the validation study.** If you touch the engine or the estimators, run `python scripts/validation_report.py --quick` and confirm that the agreement rates stay at 100%.
3. **Documentation.** Update the README or ARCHITECTURE if behaviour changes.

## Design rules

1. **Deterministic maths first.** Graph properties (d-separation, adjustment sets, cycles, instruments) are computed in `backend/causal_core.py`, never by a language model.
2. **The AI only proposes.** AI output must go through `backend/validation.py` as proposals the researcher accepts or rejects. Retrieved text is untrusted input and must pass through `backend/untrusted.py`.
3. **Honest claims.** Label results by what establishes them: structural validity, identification under the graph's assumptions, or scientific support. Never present an association as a causal effect.
4. **Local first.** Every feature must work self-hosted, with local models and with `OFFLINE_MODE` on. Outbound requests go through `backend/net.py`.
5. **Legal sources only.** Full-text discovery uses open-access and institutional routes.

## Good first contributions

- Published reference DAGs for the extraction benchmark (`benchmarks/cases/`), each with its citation.
- Front-door identification and estimation.
- Additional citation styles.
- Grouped-variable views and pathway filters on the canvas.
- Spreadsheet import of variables and arrows.

## Reporting security issues

Do not open a public issue. Use GitHub's private vulnerability reporting on the repository (Security tab, "Report a vulnerability").
