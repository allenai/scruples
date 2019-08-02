"""setup.py file for packaging ``scruples``."""

from setuptools import setup


with open('readme.md', 'r') as readme_file:
    readme = readme_file.read()

with open('requirements.txt', 'r') as requirements_file:
    requirements = requirements_file.readlines()


setup(
    name='scruples',
    version='0.0.0',
    description='A benchmark for detecting social norm violations in stories.',
    long_description=readme,
    url='https://github.com/allenai/scruples',
    author='Nicholas Lourie',
    author_email='nicholasl@allenai.org',
    keywords='scruples social norms artificial intelligence ai'
             ' machine learning ml',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    license='Apache',
    packages=['scruples'],
    package_dir={'': 'src'},
    scripts=['bin/scruples'],
    install_requires=[
        'Click >= 7.0',
        'Pattern >= 3.6',
        'apex >= 0.1',
        'attrs >= 19.1.0',
        'autograd >= 1.2',
        'dill >= 0.2.9',
        'ftfy >= 5.5.1',
        'matplotlib >= 3.0.3',
        'networkx >= 2.3',
        'numpy >= 1.16.2',
        'pandas >= 0.24.2',
        'pytorch-pretrained-bert == 0.6.1',
        'regex >= 2018.1.10',
        'scikit-learn >= 0.20.3, < 0.21',
        'scikit-optimize >= 0.5.2',
        'scipy >= 1.2.1, < 1.3',
        'seaborn >= 0.9.0',
        'spacy >= 2.1.3',
        'tensorboard >= 1.13.1',
        'tensorboardX >= 1.6',
        'tensorflow >= 1.13.1',
        'tensorflow-estimator >= 1.13.0',
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
