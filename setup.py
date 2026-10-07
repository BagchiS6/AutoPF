from setuptools import setup, find_packages

setup(
    name="autopf",
    version="0.4.0",
    description="High-throughput Phase Field simulations automation using MOOSE and MatEnsemble",
    author="Soumendu Bagchi",
    author_email="bagchis@ornl.gov",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24",
        # matensemble should be installed separately
    ],
    extras_require={
        "dev": ["pytest>=7"],
        "tiled": ["tiled[client]"],
        "production": [
            "tiled[client]",
            "torch==2.6.0",
            "gpytorch==1.15.2",
            "botorch==0.17.1",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
    ],
)
