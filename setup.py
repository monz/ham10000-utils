from setuptools import find_packages, setup

setup(
    name='ham10000_utils',
    packages=find_packages(),
    version='0.1.0',
    description='Util package for HAM10000 dataset',
    author='Markus Monz',
    install_requires = [
        'pandas >= 1.1.4',
    ],
    extras_require={
        "tf": ["tensorflow >= 2.3.1"],
        "tf_gpu": ["tensorflow-gpu>=2.3.1"],
    },
    license='MIT',
)
