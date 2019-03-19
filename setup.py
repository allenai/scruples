"""setup.py file for packaging ``socialnorms``."""

from setuptools import setup


with open('readme.md', 'r') as readme_file:
    readme = readme_file.read()

with open('requirements.txt', 'r') as requirements_file:
    requirements = requirements_file.readlines()


setup(
    name='socialnorms',
    version='0.0.0',
    description='A benchmark for detecting social norm violations in stories.',
    long_description=readme,
    url='https://github.com/allenai/socialnorms',
    author='Nicholas Lourie',
    author_email='nicholasl@allenai.org',
    keywords='social norms artificial intelligence ai machine learning'
             ' ml',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    license='Apache',
    packages=['socialnorms'],
    install_requires=requirements,
    python_requires='>= 3.7',
    zip_safe=False
)
