from setuptools import setup, find_packages

setup(
    name="earningsnlp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "nltk>=3.6.0",
        "tmtoolkit>=0.10.0",
        "streamlit>=1.10.0",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.7",
    author="Your Name",
    description="A package for analyzing earnings announcements using NLP and ML",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)