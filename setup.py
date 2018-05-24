import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="pyalgebra",
    version="0.0.1",
    author="Tim Becker",
    author_email="tjbecker@cmu.edu",
    description="Toy Computer algebra package in pure python3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tim-becker/pyalgebra",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
