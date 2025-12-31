"""
Setup script for Mauritius Land Cover Classification System.

Install in development mode:
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="mauritius-landcover",
    version="1.0.0",
    author="PhD Research",
    author_email="your.email@university.edu",
    description="Land cover classification and change detection for Mauritius using Sentinel-2 imagery",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mauritius-landcover",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "segmentation-models-pytorch>=0.3.3",
        "rasterio>=1.3.0",
        "geopandas>=0.13.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "flask>=2.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "albumentations>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
        ],
        "gee": [
            "earthengine-api>=0.1.350",
            "geemap>=0.22.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "landcover-train=models.train:main",
            "landcover-web=web.app:main",
        ],
    },
)
