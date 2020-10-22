import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lazygcnt", 
    version="0.1.0",
    author="Morteza Ramezani",
    author_email="morteza@cse.psu.edu",
    description="LazyGCN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MortezaRamezani/lazygcn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
