#!/usr/bin/env python3
"""
Setup script for tech-news-summarizer package.
This will automatically download required NLTK data after installation.
"""

from setuptools import setup, find_packages

# Post-install hook removed - no longer needed

setup(
    name="tech-news-summarizer",
    version="0.1.0",
    description="A tool for scraping, cleaning, and summarizing tech news articles",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "pandas<2.2.0",
        "python-dateutil>=2.8",
        "tqdm>=4.66",
        "newspaper3k>=0.2.8",
        "feedparser>=6.0.11",
        "beautifulsoup4>=4.12.2",
        "requests>=2.31.0",
        "lxml==4.9.3",
        # "nltk==3.9.1",  # Removed - using custom sentence tokenizer instead
        "unidecode>=1.3.7",
        "regex>=2023.10.3",
        "scikit-learn>=1.4",
        "spacy==3.8.7",
        "matplotlib==3.10.5",
        "seaborn==0.13.2",
        "PyYAML>=6.0.2",
        "pytest",
        "transformers>=4.42",
        "tokenizers>=0.15",
        "sentencepiece>=0.2",
        "rouge-score>=0.1.2",
        "absl-py>=1.4.0",
        "evaluate",
        "sumy",
        "torch",
    ],
    # No post-install hooks needed
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
