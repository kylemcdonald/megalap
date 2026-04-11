# Publishing `megalap`

## Local validation

From the repository root:

```bash
python -m pip install -U build twine pytest
python -m build
python -m twine check dist/*
python -m pip install --force-reinstall dist/*.whl
python -m pytest -q
```

## Continuous integration

The repository includes:

- [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for editable-install tests and distribution checks
- [`.github/workflows/release.yml`](.github/workflows/release.yml) for tagged multi-platform releases

`release.yml` builds:

- one source distribution on Linux
- wheels on Linux, macOS Intel, macOS Apple Silicon, and Windows

## One-time PyPI setup

Before `release.yml` can publish to PyPI:

1. Create the `megalap` project on PyPI.
2. Configure PyPI trusted publishing for GitHub repository `kylemcdonald/megalap`.
3. Set the trusted publisher workflow file to `.github/workflows/release.yml`.
4. Add a GitHub `pypi` environment if you want environment protection on releases.

## Release steps

1. Update `version` in `pyproject.toml`.
2. Commit and push to `main`.
3. Wait for `.github/workflows/ci.yml` to pass.
4. Create and push a tag like `v0.1.0`.
5. Confirm the release workflow uploads the sdist and wheels, then publishes them to PyPI.

## Optional dry run

If you want to verify the package manually before a real release:

```bash
python -m pip install -U pytest
python -m build
python -m pip install --force-reinstall dist/*.tar.gz
python -m pytest -q
python -m pip install --force-reinstall dist/*.whl
python -m pytest -q
```

## Notes

- The repository README keeps the hero image. PyPI uses `README_PYPI.md` as the package long description so the package page does not depend on local image assets.
- The release workflow uses `cibuildwheel` so PyPI receives Linux, macOS, and Windows wheels instead of a single local platform wheel.
