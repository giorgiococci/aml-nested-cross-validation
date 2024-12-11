from setuptools import setup, find_packages

setup(
    name="cross_validation_common",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "azure-ai-ml",
        "azure-identity",
        "azureml-fsspec"
    ],
    author="Giorgio Cocci",
    author_email="your.email@example.com",
    description="A library for common cross-validation utilities.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cross-validation-common",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)