import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="auto_modelling",
    version="1.2.0",
    author="Tang Han",
    author_email="aloofness54@gmail.com",
    description="A light package for automatic model tuning and stacking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TangHan54/auto-modelling.git",
    packages=setuptools.find_packages(),
    install_requires=[
        'sklearn',
        'ipython',
        'numpy',
        'pandas',
        'setuptools',
        'xgboost'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)