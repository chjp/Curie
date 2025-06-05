"""
Setup script for Curie MVP
"""

from setuptools import setup, find_packages

setup(
    name="curie-mvp",
    version="0.1.0",
    description="Minimum Viable Product of Curie: A Research Experimentation Agent",
    author="Curie Team",
    packages=find_packages(),
    package_dir={"": "."},
    include_package_data=True,
    install_requires=[
        "langchain>=0.1.0",
        "openai>=1.0.0",
        "anthropic>=0.5.0",
        "numpy>=1.22.0",
        "pandas>=1.4.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
