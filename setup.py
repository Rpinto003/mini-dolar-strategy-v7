from setuptools import setup, find_packages

setup(
    name="mini_dolar_strategy",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas>=1.5.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'loguru>=0.6.0',
        'pyyaml>=6.0.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.12.0',
        'pytest>=7.0.0'
    ],
)