"""
GEKO: Gradient-Efficient Knowledge Optimization

Install with: pip install geko
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="gekolib",
    version="0.3.1",
    author="Syed Abdur Rehman",
    author_email="ra2157218@gmail.com",
    description="Gradient-Efficient Knowledge Optimization - Smart training for any LLM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Abd0r/GEKO",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "geko-app = geko.app:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "isort>=5.0",
        ],
        "transformers": [
            "transformers>=4.20.0",
        ],
        "peft": [
            "peft>=0.6.0",
        ],
        "bnb": [
            "bitsandbytes>=0.41.0",
        ],
        "rich": [
            "rich>=10.0.0",
        ],
        "app": [
            "gradio>=4.0.0",
            "plotly>=5.0.0",
            "transformers>=4.20.0",
            "datasets>=2.0.0",
        ],
        "all": [
            "transformers>=4.20.0",
            "peft>=0.6.0",
            "bitsandbytes>=0.41.0",
            "rich>=10.0.0",
            "gradio>=4.0.0",
            "plotly>=5.0.0",
            "datasets>=2.0.0",
        ],
    },
    keywords=[
        "machine learning",
        "deep learning",
        "nlp",
        "llm",
        "training",
        "optimization",
        "curriculum learning",
        "sample selection",
    ],
)
