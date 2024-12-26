"""
Setup.py file

Sources:
- https://docs.python.org/3.11/distutils/setupscript.html
- https://stackoverflow.com/questions/57089252/create-python-development-environment-virtualenv-using-setup-py
"""

import os
import pathlib
from setuptools import setup

parent_dir_path = str(pathlib.Path(__file__).parent.resolve())
with open(os.path.join(parent_dir_path, 'requirements.txt')) as f:
    requirements = f.readlines()

setup(
    name='FLightcase',
    version='0.1.0',
    author='Stijn Denissen',
    author_email='stijn.denissen@vub.be',
    description='FLightcase toolbox for Federated Learning',
    url='https://github.com/AIMS-VUB/FLightcase/',
    license='GPLv3',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'FLightcase = cli:cli',
        ]
    }
)
