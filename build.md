# Release Steps

1. Check scheduled depreciations
1. Run examples and fix errors
1. Check an example on Google Colab
1. Update readme and copy relevant information to docs/source/description.md
1. Follow instructions in docs/build.md
1. Run: python setup.py sdist bdist_wheel
1. Run. twine upload --repository-url https://test.pypi.org/legacy/ dist/*
1. Test: pip install -i https://test.pypi.org/simple/ --no-deps --upgrade lumin
1. Release: twine upload dist/*
1. Update Colab example links to upcoming version
1. Make github release
1. git pull
1. Increment lumin.version.py
1. Add build to readthedocs and check docs
1. Update authorship on Zenodo
