#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:  from setuptools import setup, find_packages
except ImportError: from distutils.core import setup, find_packages
    
with open("README.md", "r") as fin: long_description = fin.read()

with open('requirements.txt') as f: requirements = f.read().strip().split('\n')


exec(open('lumin/version.py').read())

setup(
    name="lumin",
    version=__version__,  # noqa

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
    install_requires=requirements,

    license="Apache Software License 2.0",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        'Programming Language :: Python :: 3.7',
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X ",
        "Operating System :: POSIX :: Linux",
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Development Status :: 4 - Beta',
    ],
)
