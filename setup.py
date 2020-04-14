from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["wheel==0.34.2", "keras==2.3.1", "tensorflow==2.1.0", "pandas==1.0.3", "matplotlib==3.2.1",
                "beautifulsoup4>=4.9.0", "aiohttp>=3.6.2", "PyYAML==5.1"]

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
    python_requires='>=3.6',
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
    ],
)
