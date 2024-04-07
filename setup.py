from setuptools import setup, find_packages

setup(
    name='lpotato',
    version='1.0.0',
    url='https://github.com/mypackage.git',
    author='Neal Fultz',
    author_email='nfultz@gmail.com',
    description='Small scoring module for lpotato payout model',
    py_modules=['lpotato'],
    install_requires=['pandas', 'xgboost', 'scikit-learn'],
)

