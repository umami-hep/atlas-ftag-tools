# Developer Guidelines

## 1  Quick local setup

```bash
# clone and create a dev environment
git clone https://github.com/umami-hep/atlas-ftag-tools.git
cd atlas-ftag-tools
python -m venv .venv # or: conda create -n ftag python=3.11
source .venv/bin/activate

# install the library *and* dev extras
python -m pip install -e ".[dev]"

# install pre-commit hooks (formatting, linting, typing)
pre-commit install
pre-commit run --all-files
```

> **Why use `-e`?**
> The *editable* install means you can `import ftag` from any directory while
> editing the source in place.

## 2  Branching & PR policy

| Branch | Purpose | Protection rules |
|--------|---------|------------------|
| **`main`** | always deployable; releases are tagged from here | ✓ CI must pass<br/>✓ linear-history |
| **feature/**`xyz` | work in progress, short-lived | none |
| **`release/*`** | back-ports / hot-fixes after a tag | same rules as *main* |

1. Open a Pull Request early; mark it **Draft** if it’s not ready.
2. At least **one review** from the *atlas-ftag-tools dev-team* is required.
3. Squash-merge once *green*; avoid merge commits.

## 3  Code style & static checks

| Tool | Checked in CI | Fix command |
|------|---------------|-------------|
| **black** | formatting | `black .` |
| **ruff** (`pyproject.toml`) | lint + simple refactors | `ruff check . --fix` |
| **mypy** | type safety | `mypy ftag` |
| **isort** (via ruff) | import order | auto-fixed by *ruff* |
| **pre-commit** | umbrella runner | `pre-commit run --all-files` |

### How the hook chain works

```
pre-commit ➜ ruff ➜ black ➜ mypy
```

If a stage fails the commit is rejected. Run the hook manually to fix issues
before pushing:

```bash
pre-commit run --all-files
```

## 4  Testing strategy

* **Unit tests** live in `ftag/tests/` and follow the *pytest* conventions.
* CI executes the suite on 3 Python versions (3.9, 3.11, 3.12) and uploads a coverage report.

```bash
# local one-liner
pytest ftag/tests/
```

Coverage **must stay ≥ 90 %**. If you add logic, add matching tests; if you
change public behaviour, update the doc-strings *and* the examples.

## 5  Communication channels

* **GitHub Issues** – bug reports & feature requests
* **Mattermost `Umami / puma / upp`** – quick questions, coordination

When in doubt, open an Issue and tag the maintainer team.
Happy coding — and may your $b$-tagging be ever accurate! 🎉
