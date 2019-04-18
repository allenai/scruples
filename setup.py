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
    package_dir={'': 'src'},
    install_requires=[
        'Click >= 7.0',
        'apex >= 0.1',
        'attrs >= 19.1.0',
        'ftfy >= 5.5.1',
        'numpy >= 1.16.2',
        'pytorch-pretrained-bert >= 0.6.1',
        'regex >= 2018.1.10',
        'scikit-learn >= 0.20.3',
        'scikit-optimize >= 0.5.2',
        'scipy >= 1.2.1',
        'spacy >= 2.1.3',
        'tensorboard >= 1.13.1',
        'tensorboardX >= 1.6',
        'tensorflow-estimator >= 1.13.0',
        'tensorflow >= 1.13.1',
        'torch >= 1.0.1.post2',
        'torchvision >= 0.2.2.post3',
        'xgboost >= 0.82'
    ],
    setup_requires=[
        'pytest-runner'
    ],
    tests_require=[
        'pytest'
    ],
    include_package_data=True,
    python_requires='>= 3.7',
    zip_safe=False)
