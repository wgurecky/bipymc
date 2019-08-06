from setuptools import setup, find_packages
setup(
    name = "bipymc",
    version = "0.1",
    packages = find_packages(),
    install_requires = ['numpy>=1.7.0', 'scipy>=0.12.0', 'scipydirect'],
    package_data = { '': ['*.txt'] },
    author = 'William Gurecky',
    license = "BSD3",
    author_email = "william.gurecky@gmail.com",
)
