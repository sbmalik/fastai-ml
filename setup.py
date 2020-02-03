from setuptools import setup

def readme():
    with open('README.md') as f:
        README = f.read()
    return README


setup(
    name="fastai_ml",
    version="1.0.0",
    description="A Python package to get helper functions for ML while using latest fastai.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    author="Subtain Tariq Malik",
    author_email="m3si6ti@gmail.com",
    license="MIT",
    packages=["fastai_ml"],
    include_package_data=True,
    install_requires=["fastai", "pandas", "pandas_summary"],
    # url="https://github.com/nikhilkumarsingh/weather-reporter"
)