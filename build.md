# Pypi build instructions
1. Run: python setup.py sdist bdist_wheel
1. Run. twine upload --repository-url https://test.pypi.org/legacy/ dist/*
1. Test: pip install -i https://test.pypi.org/simple/ lumin
1. Release: twine upload dist/*