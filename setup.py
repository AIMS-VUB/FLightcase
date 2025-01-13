"""
Setup.py file

Sources:
- https://docs.python.org/3.11/distutils/setupscript.html
- https://stackoverflow.com/questions/57089252/create-python-development-environment-virtualenv-using-setup-py
"""

from setuptools import setup

setup(
    name='FLightcase',
    version='0.1.0',
    author='Stijn Denissen',
    author_email='stijn.denissen@vub.be',
    description='FLightcase toolbox for Federated Learning',
    url='https://github.com/AIMS-VUB/FLightcase/',
    license='CC BY-NC-ND 4.0',
    install_requires=[
        'torch==2.5.1',
        'pandas==2.2.3',
        'monai==1.4.0',
        'scikit-learn==1.6.0',
        'tqdm==4.67.1',
        'nibabel==5.3.2',
        'paramiko==3.5.0',
        'scp==0.15.0',
        'matplotlib==3.10.0',
        'scipy==1.14.1',
        'numpy==1.26.4',  # 2.2.0 causes conflict with monai==1.4.0
        'click==8.1.8',
        'twine==6.0.1'
    ],
    entry_points={
        'console_scripts': [
            'FLightcase = cli:cli',
        ]
    },
    packages=[]
)
