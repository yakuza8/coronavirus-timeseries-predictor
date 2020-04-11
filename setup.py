from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["beautifulsoup4>=4.9.0", "aiohttp>=3.6.2", "PyYAML>=3.12"]

setup(
    name="coronavirus-predictor",
    version="1.0.0",
    author="Berat Cankar",
    author_email="berat.cankar@gmail.com",
    description="Coronavirus Prediction with RNN and Data Crawler",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/yakuza8/coronavirus-timeseries-predictor/",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
    ],
)