# setup.py
from setuptools import setup, find_packages

setup(
    name="nnetnav",  # Name of your package
    version="0.1",
    packages=find_packages(where="src"),  # Looks for packages inside `src`
    package_dir={"": "src"},  # Specifies `src` as the root for the packages
    install_requires=[],  # List dependencies here if needed
)

