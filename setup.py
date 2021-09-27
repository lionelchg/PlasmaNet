#!/usr/bin/env python

"""
The setup script for pip. Allows for `pip install -e .` installation.
"""

from setuptools import setup, find_packages

requirements = ['numpy', 'matplotlib', 'torch', 'h5py', 'PyYAML', 'torchvision']
setup_requirements = []
tests_requirements = ['pytest']

setup(
    author='L. Cheng, E. Ajuria, G. Bogopolsky',
    author_email='cheng@cerfacs.fr',
    classifiers=[
        'Development Status :: 3 - Alpha',
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
    version='1.0',
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'euler=PlasmaNet.cfdsolver.euler.euler:main',
            'plasma_euler=PlasmaNet.cfdsolver.euler.plasma:main',
            'scalar=PlasmaNet.cfdsolver.scalar.scalar:main',
            'streamer=PlasmaNet.cfdsolver.scalar.streamer:main',
            'run_cases=PlasmaNet.cfdsolver.cases:main',
            'show_network=PlasmaNet.nnet.show_network:main',
            'train_network=PlasmaNet.nnet.trainer.train_network:main',
            'train_networks=PlasmaNet.nnet.trainer.train_networks:main',
            'train_pproc=PlasmaNet.nnet.pproc:main',
            'eval_datasets=PlasmaNet.poissonsolver.eval_datasets:main',
            'compute_rf=PlasmaNet.nnet.compute_rf:compute_RF_2D',
        ],
    },
)
