Due to some strict version requirements on packages, it is recommended to install LUMIN in its own Python environment, e.g `conda create -n lumin python=3.6`

## From PyPI

The main package can be installed via:
`pip install lumin`

Full functionality requires two additional packages as described below.

## From source

```
git clone git@github.com:GilesStrong/lumin.git
cd lumin
pip install .
```

Optionally, run pip install with `-e` flag for development installation.

### Optional requirements

- sparse: enables loading on COO sparse-format tensors, install via e.g. `pip install sparse`