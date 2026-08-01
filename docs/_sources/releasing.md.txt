# Releasing Mandol

This guide describes the manual release process for Mandol `0.1.0`. PyPI
publication is intentionally not performed by GitHub Actions.

## 1. Confirm the release commit

```bash
cd /home/zyh/code/Mandol-main
git status -sb
git pull --ff-only origin main
git log -1 --oneline
```

Confirm that the `main` CI and documentation deployment for the release commit
have succeeded.

## 2. Confirm that the version is unused

```bash
git fetch --tags origin
test -z "$(git tag -l v0.1.0)"
! curl -fsS https://pypi.org/pypi/mandol/0.1.0/json >/dev/null
```

Stop if either command indicates that the tag or PyPI version exists.

## 3. Create and push the annotated tag

```bash
git tag -a v0.1.0 -m "Mandol v0.1.0"
git push origin v0.1.0
```

Do not use `--force`. Pushing the tag runs the release-build workflow, which
builds and validates distributions but does not publish to PyPI.

## 4. Verify the tag target

```bash
git rev-list -n 1 v0.1.0
git rev-parse HEAD
git ls-remote --tags origin refs/tags/v0.1.0
```

The local tag commit and current release commit must match. For an annotated
tag, the remote output is the tag object; use
`git ls-remote --tags origin 'refs/tags/v0.1.0^{}'` to inspect its peeled commit.

## 5. Verify tag-pinned README assets

```bash
curl -fL --retry 3 \
  -o /tmp/mandol-overview-v0.1.0.png \
  https://raw.githubusercontent.com/AgentCombo/Mandol/v0.1.0/README.assets/Mandol-overview-v2.png

file /tmp/mandol-overview-v0.1.0.png

curl -fL https://github.com/AgentCombo/Mandol/blob/v0.1.0/README.md \
  >/dev/null
curl -fL https://github.com/AgentCombo/Mandol/blob/v0.1.0/README_CN.md \
  >/dev/null
```

The downloaded Overview asset must be reported as a PNG before uploading to
PyPI.

## 6. Rebuild from a clean tag checkout

Do not upload distributions left over from release preparation.

```bash
release_dir="$(mktemp -d)"

git clone \
  --branch v0.1.0 \
  --depth 1 \
  https://github.com/AgentCombo/Mandol.git \
  "$release_dir/Mandol"

cd "$release_dir/Mandol"
python3.12 -m venv .release-venv
source .release-venv/bin/activate
python -m pip install --upgrade pip build twine
python -m build
python -m twine check --strict dist/*
```

## 7. Inspect the final files

```bash
ls -lh dist/
sha256sum dist/*
test "$(find dist -maxdepth 1 -type f | wc -l)" -eq 2
```

Only these archives should be present:

```text
mandol-0.1.0-py3-none-any.whl
mandol-0.1.0.tar.gz
```

## 8. Upload interactively to PyPI

```bash
python -m twine upload dist/*
```

Use `__token__` as the username and enter the PyPI API token only at Twine's
password prompt. Do not place the token in the command, shell history,
`.pypirc`, `.env`, shell startup files, Git configuration, or project files.

## 9. Verify the PyPI page

```bash
curl -fsS https://pypi.org/pypi/mandol/0.1.0/json \
  | python -m json.tool \
  >/tmp/mandol-0.1.0-pypi.json
```

Open <https://pypi.org/project/mandol/0.1.0/> and verify the title, Overview
image, English and Chinese links, project links, installation commands,
Python requirement, license, classifiers, and BibTeX.

## 10. Verify a clean installation from PyPI

```bash
verify_dir="$(mktemp -d)"
python3.12 -m venv "$verify_dir/venv"
"$verify_dir/venv/bin/python" -m pip install --upgrade pip
"$verify_dir/venv/bin/python" -m pip install --no-cache-dir mandol==0.1.0
"$verify_dir/venv/bin/python" -m pip check

"$verify_dir/venv/bin/python" - <<'PY'
import importlib.metadata

import mandol
from mandol import MemoryUnit, SemanticGraph, SemanticMap

assert importlib.metadata.version("mandol") == "0.1.0"
assert mandol.__version__ == "0.1.0"
print("PyPI release verification passed")
PY

"$verify_dir/venv/bin/python" -m pip install \
  --dry-run \
  --no-cache-dir \
  mandol
```

Confirm that the unpinned candidate is `0.1.0`, not an older release.

## 11. Create the GitHub Release

Only after PyPI and clean-install verification succeed:

```bash
gh release create v0.1.0 \
  -R AgentCombo/Mandol \
  --verify-tag \
  --title "Mandol v0.1.0" \
  --notes-file release-notes/v0.1.0.md
```

The tag workflow already retains checked wheel and sdist artifacts in GitHub
Actions. Do not attach a separately rebuilt variant of the same version.

## 12. Final checks

Confirm the `main` CI, tag release-build workflow, Pages deployment, GitHub
Release, PyPI page, default `pip install mandol` resolution, README badges, and
website installation instructions.
