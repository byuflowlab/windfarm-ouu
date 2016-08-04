#!/usr/bin/env python
# encoding: utf-8

from setuptools import setup, find_packages
import platform


setup(
    	name='OUUWakeModels',
    	version='0.0.0',
    	description='collection of wake models for optimization under uncertainty',
    	install_requires=['florisse', 'akima>=1.0', 'wakeexchange', 'jensen3d', 'gaussianwake'],
    	dependency_links=['https://github.com/WISDEM/FLORISSE/tarball/develop#egg=florisse', 'https://github.com/andrewning/akima/tarball/master#egg=akima-1.0.0', 'https://github.com/byuflowlab/wake-exchange/tarball/master#egg=wakeexchange', 'https://github.com/byuflowlab/Jensen3D/tarball/module#egg=jensen3d', 'https://github.com/byuflowlab/gaussian-wake/tarball/master#egg=gaussianwake'],
	zip_safe=False
)
