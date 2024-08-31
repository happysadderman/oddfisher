from setuptools import setup, find_packages

setup(
    name="oddfisher",
    version="0.1.1",
    description="A simple Python package for fisher exact test with hypothesized odds ratio",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)