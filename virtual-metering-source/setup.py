from setuptools import find_packages, setup

REQUIRED_PACKAGES = ["lightgbm==2.2.0", "pandas==0.23.4"]

setup(
    name="my_model",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES,
)
