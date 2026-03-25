from setuptools import setup, find_packages

setup(
    name="backtester",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "ccxt>=4.4.0",
        "pandas>=2.2.0",
        "numpy>=2.0.0",
        "ta>=0.11.0,<1.0",
        "requests>=2.32.0",
        "beautifulsoup4>=4.13.0",
        "rich>=13.7.0",
        "click>=8.1.7",
    ],
    entry_points={
        "console_scripts": [
            "backtester=backtester.cli:main",
        ],
    },
    python_requires=">=3.10,<3.14",
)
