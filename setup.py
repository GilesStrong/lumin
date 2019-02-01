#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pip

try:  from setuptools import setup, find_packages
except ImportError: from distutils.core import setup, find_packages
    
try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except ImportError:
    with open("README.md", "r") as fin: long_description = fin.read()

with open('requirements.txt') as f: requirements = f.read().strip().split('\n')


exec(open('lumin/version.py').read())

setup(
    name="lumin",
    version=__version__,

    author="Giles Strong",
    author_email="giles.strong@outlook.com",

    description="LUMIN Unifies Many Improvements for Networks: A PyTorch wrapper to make deep learning more accessable to scientists",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GilesStrong/lumin",
    keywords='lumin, deep learning, machine learning, physics, science, statistics',

    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.6',
    dependency_links=[
        'https://github.com/GilesStrong/random-forest-importances/tarball/master#egg=rfpimp-1.3.1+git.0edef31'
    ],
    install_requires=requirements,

    license="Apache Software License 2.0",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        'Programming Language :: Python :: 3.7',
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
        'Intended Audience :: Scientists',
        'Natural Language :: English',
        'Development Status :: 3 - Alpha',
    ],
)
