# Build instructions

1. Run: sphinx-apidoc -l -o source/ ../lumin ../lumin/*abs_* ../lumin/utils/mod_ver.py -f
1. Change toctree max depth to 1 in source files
1. Ensure TOC in source/index.rst is up to date
1. Replace contents of source/description.md with relevant sections from ../README.md
1. Run: make html
1. Open _build/html/index.html and check
1. Push changes and check build readthedocs build progress
