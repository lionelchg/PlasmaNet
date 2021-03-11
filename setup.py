#!/usr/bin/env python

"""
The setup script for pip. Allows for `pip install -e .` installation.
"""

from setuptools import setup, find_packages

requirements = ['numpy', 'matplotlib', 'torch', 'h5py', 'PyYAML', 'torchvision']
setup_requirements = []
tests_requirements = ['pytest']

setup(
    author='G. Bogopolsky, L. Cheng, E. Ajuria',
    author_email='bogopolsky@cerfacs.fr',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8'
    ],
    description='PlasmaNet: Solving the electrostatic Poisson equation for plasma simulations',
    install_requires=requirements,
    license='GNU General Public License v3',
    long_description='\n\n',
    include_package_data=True,
    keywords='plasma poisson deep learning pytorch',
    name='PlasmaNet',
    packages=find_packages(include=['PlasmaNet']),
    setup_requires=setup_requirements,

    test_suite='tests',
    tests_require=tests_requirements,
    url='https://nitrox.cerfacs.fr/cfd-apps/plasmanet',
    version='0.1',
    zip_safe=False,
)
