from setuptools import setup, find_packages

setup(
    name="meta-cognitive-learning-system",
    version="1.0.0",
    author="mwasifanwar",
    description="Advanced AI system that monitors and improves its own learning process through meta-cognition and self-reflection",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "pytest>=7.3.0",
        "torchvision>=0.15.0"
    ],
    python_requires=">=3.8",
)