import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GSOM",
    version="0.0.1",
    author="Rodrigo de Sales da Silva Adeu",
    author_email="",
    description="A simple python implementation for GSOM algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rodrigosales/GSOM",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)