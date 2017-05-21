from distutils.core import setup

setup(
    name='MyML',
    version='1.0.0',
    author='J. Nistal Hurle',
    author_email='j.nistalhurle@gmail.com',
    packages=['my_ml', 'tests'],
    scripts=[],
    url='http://pypi.python.org/pypi/MyML/',
    license='LICENSE.txt',
    description='This module implements simple machine-learning algorithms built from scratch.',
    long_description=open('README.md').read(),
)
