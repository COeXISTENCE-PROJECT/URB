# Contributing to URB

Thank you for your interest in contributing to URB.

This repository accepts:
- **Benchmark coverage extensions**, which add new experimental components,
- **Leaderboard contributions**, new results to be ranked in our leaderboard,
- **Study contributions**, focused tests of research hypotheses using URB, and
- **Software changes**, which improve the framework itself.

Please read the guidelines below before opening a pull request.

---

## Types of contributions

### 1. Benchmark extensions
These contributions extend the set of experiments that can be conducted with URB.

Examples include:
- new experiment scripts,
- new baseline models,
- new algorithm hyperparameterizations,
- new environment configurations,
- new task configurations, or
- new networks.

If your contribution falls into this category, please follow the repository structure described below.

### 2. Leaderboard contributions
These contributions include new results placed under `results/` with a unique experiment identifier for the leaderboard.

These can be obtained from existing methods and parameterizations, or with a new addition of a method or a parameterization. The addition of new tasks may be subject to further considerations to keep the contributor efforts more focused.

### 3. Study contributions
Study contributions test a focused research hypothesis and are placed under `studies/`. Copy [`studies/_template`](studies/_template/) and follow the full [`studies/README.md`](studies/README.md) guidelines.

A submitted study should include its proposal, compact experiment evidence, final report, references, and at least one completed independent review. Independent reviews can be made by LLM agents (which we provide a prompt template for). Keep the template filenames and standard fields unchanged, report the full range of relevant results, and limit conclusions to the tested setting. A `COVER.png` image is optional.

### 4. Software changes
These contributions modify or improve the software itself rather than adding benchmark components.

Examples include:
- bug fixes,
- refactoring,
- documentation improvements,
- CI/CD updates,
- registry or dashboard changes,
- tooling or infrastructure improvements, and
- core framework development.

These contributions do **not** need to follow the benchmark-extension structure below, but the pull request should clearly explain what part of the software is affected and why.

---

## Repository structure for benchmark extensions

### Adding new scripts
Users can add new experiment scripts for testing different algorithms, implementations, and training pipelines.

New scripts should:
- be placed in `scripts/`,
- follow the recommended structure from [`scripts/base_script.py`](scripts/base_script.py), 
- be paired with at least 5 promising hyperparameterization files in [`config/algo_config/<alg_name>`](config/algo_config/), and
- include enough information in a header docstring or a dedicated markdown in [`scripts/docs/<alg-name>`](scripts/docs/) for others to understand how they are intended to be used.

### Adding new baselines

#### Decentralized (per-agent) baseline models
New decentralized baseline models should extend [`baseline_models/BaseLearningModel`](baseline_models/base.py). This type of methods often introduce naive agent-level decision making mechanisms.

#### Centralized methods
Methods that cannot be reasonably implemented in a per-agent baseline form can be implemented as scripts using [`scripts/base_script.py`](scripts/base_script.py).

### Adding new scenarios and hyperparameterizations
Experiment configurations can be extended by adding:
- algorithm hyperparameterizations in [`config/algo_config/`](config/algo_config/),
- environment settings in [`config/env_config/`](config/env_config/), and
- new tasks in [`config/task_config/`](config/task_config/).

Please place new files in the appropriate directory.

---

## Branching

Create your contribution branch from `dev`.

A clear branch naming convention is recommended, for example:
- `feat/123-short-description`
- `fix/123-short-description`
- `docs/123-short-description`
- `refactor/123-short-description`

If your contribution is linked to an issue, include the issue number in the branch name when possible.

---

## Pull requests

When opening a pull request, please:

- clearly describe what the PR changes,
- indicate whether it is a **benchmark extension**, **leaderboard contribution**, **study contribution**, **software change**, or any combination that applies,
- explain why the change is needed,
- keep the scope focused, and
- update documentation if the change affects how the repository is used.

For benchmark extensions, please also state:
- what was added,
- where it was added, and
- how it is expected to be used.

For study contributions, please also state:
- the hypothesis and tested setting,
- the conclusion and main evidence, and
- who or what performed the independent review.

For software changes, please state:
- what part of the software was changed, and
- whether the change affects existing behavior.

New study and result directories are checked automatically against the website indexer. Added or modified experiment scripts are tested individually; maintainers can also run the complete script suite through the manual **Test all scripts** workflow.

---

## General expectations

Before submitting, please make sure that:
- your changes are placed in the appropriate location,
- the code is reasonably clear and consistent with the style of the repository,
- outdated or experimental code is not left behind unnecessarily,
- related documentation is updated when needed, and
- the PR description is sufficient for review.

---

## Questions

If you are unsure whether your contribution should be implemented as a benchmark extension or as a software change, describe your intent clearly in the pull request and follow the closest matching structure.
