# Table of contents

- [Contributing to LUMIN](#contributing-to-lumin)
- [Developing LUMIN](#developing-lumin)

## Contributing to LUMIN

If you are interested in contributing to LUMIN,
    - Search for your issue here: https://github.com/GilesStrong/lumin/issues
    - Pick an issue and comment on the task that you want to work on this feature.
    - If you need more context on a particular issue, please ask and discuss.

Once you finish implementing a feature or bug-fix, please send a Pull Request to
https://github.com/GilesStrong/lumin

## Developing LUMIN

To develop LUMIN on your machine, here are some tips:

1. Uninstall all existing LUMIN installs:

```bash
conda uninstall lumin
pip uninstall lumin
```

2. Clone a copy of LUMIN from source:

```bash
git clone https://github.com/GilesStrong/lumin
cd lumin
```

3. Perform the modifications in source code and rebuild, test the changes locally.

```bash
python setup.py install
```
(after the changes)

Create a Pull Request referencing the issue.
