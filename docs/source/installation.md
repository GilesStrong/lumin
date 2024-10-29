### From PyPI

The main package can be installed via:
`pip install lumin`

Full functionality requires an additional package as described below.

### For development

Check out the repo locally:

```bash
git clone git@github.com:GilesStrong/lumin.git
cd lumin
```

For development usage, we use [`poetry`](https://python-poetry.org/docs/#installing-with-the-official-installer) to handle dependency installation.
Poetry can be installed via, e.g.

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry self update
```

and ensuring that `poetry` is available in your `$PATH`

Lumin requires `python >= 3.10`. This can be installed via e.g. [`pyenv`](https://github.com/pyenv/pyenv):

```bash
curl https://pyenv.run | bash
pyenv update
pyenv install 3.10
pyenv local 3.10
```

Install the dependencies:

```bash
poetry install
poetry self add poetry-plugin-export
poetry config warnings.export false
poetry run pre-commit install
```

### Optional requirements

- sparse: enables loading on COO sparse-format tensors, install via e.g. `pip install sparse`
