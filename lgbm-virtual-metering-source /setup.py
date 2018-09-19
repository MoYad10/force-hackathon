from setuptools import find_packages, setup

REQUIRED_PACKAGES = ["scikit-learn==0.19.2", "pandas==0.23.4", "lightgbm==2.1.0"]

setup(
    name="LGBM-Force",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES,
)
