
# AI WhiteShorts — GP Refactor (v3.3)

This branch moves the project to a best-practice Python layout:

```
/src/white_shorts/
  __init__.py
  featurize.py
  io.py
  models/
  cli/
.github/workflows/ci.yml
pyproject.toml
setup.cfg
```

## How to use this branch locally

```bash
git checkout -b GP_Refactor
pip install -e .
```

Then adapt existing scripts to import from `white_shorts` (examples to follow in PR).
