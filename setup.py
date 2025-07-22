"""
Setup script for Meta-Predictor Package
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Meta-Predictor Package for metabolite prediction using transformer models"

# Read requirements
def read_requirements():
    """Read requirements from requirements_pip.txt"""
    requirements = []
    
    # Skip requirements file - use minimal dependencies only
    # The original requirements have dependency conflicts, so we'll use fallback only
    pass
        
    # Use fallback requirements if file parsing fails or file not found
    if not requirements:
        requirements = [
            # Core dependencies only - OpenNMT-py should be installed separately
            'rdkit',                # Molecular structure handling
            'numpy>=1.21.0',        # Numerical operations
            'pandas>=1.3.0',        # Data manipulation
            'tqdm>=4.60.0'          # Progress bars
        ]
    return requirements

setup(
    name="meta-predictor",
    version="1.0.0",
    author="Meta-Predictor Team",
    author_email="your.email@example.com",
    description="A Python package for metabolite prediction using transformer models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/meta-predictor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
        "web": [
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
        ],
        "viz": [
            "matplotlib>=3.0",
            "pillow>=8.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "meta-predict=meta_predictor.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="metabolite prediction chemistry transformer neural machine translation",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/meta-predictor/issues",
        "Source": "https://github.com/yourusername/meta-predictor",
        "Documentation": "https://meta-predictor.readthedocs.io/",
    },
)