#!/usr/bin/env python
# encoding: utf-8

from setuptools import setup, find_packages
import platform


setup(
    name='OUUWakeModels',
    version='0.0.0',
    description='collection of wake models for optimization under uncertainty',
    install_requires=['florisse'],
    dependency_links=['https://github.com/WISDEM/FLORISSE/tarball/FLORIScosine#egg=florisse'],
    zip_safe=False
)
