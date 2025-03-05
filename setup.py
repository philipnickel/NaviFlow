from setuptools import setup, find_packages

setup(
    name="naviflow",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        # any other dependencies
    ],
)