from setuptools import setup, find_packages

setup(
    name="backtester",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "ccxt>=4.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "ta>=0.11.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "rich>=13.0.0",
        "click>=8.1.0",
        "aiohttp>=3.9.0",
        "flask>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "backtester=backtester.cli:main",
        ],
    },
    python_requires=">=3.10",
)
