# Build instructions

1. Run: sphinx-apidoc -l -o source/ ../lumin ../lumin/*abs_* ../lumin/utils/mod_ver.py -f
1. Change toctree max depth to 1 in source files
1. Run: make html
