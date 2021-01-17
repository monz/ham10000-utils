from setuptools import find_packages, setup

setup(
    name='ham10000_utils',
    packages=find_packages(),
    version='0.2.0',
    description='Util package for HAM10000 dataset',
    author='Markus Monz',
    install_requires=[
        'pandas >= 1.2.0',
    ],
    license='MIT',
)
