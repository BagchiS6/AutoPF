from setuptools import setup, find_packages

setup(
    name="autopf",
    version="0.1.0",
    description="High-throughput Phase Field simulations automation using MOOSE and MatEnsemble",
    author="Soumendu Bagchi",
    author_email="bagchis@ornl.gov",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        # matensemble should be installed separately
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
    ],
)
