from setuptools import setup, find_packages
import os
setup(
    name="StructPot-CLR",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "ase>=3.22.0",
        "e3nn>=0.5.0",
        "lmdb>=1.4.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "transformers>=4.30.0",
    ],
    python_requires=">=3.8",
    author="Haoyu Wan",
    description="WHY PAPER",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
)
