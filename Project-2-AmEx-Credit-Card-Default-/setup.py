from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='This is a classification project to predict American Express clients default probability based on their profile',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Camilo Durango',
    license='MIT',
    install_requires=[
        'pandas>=1.0.0',
        'seaborn>=0.11.0',
        'matplotlib>=3.3.0',
        'click>=7.0',
        'python-dotenv>=0.15.0',
        'scikit-learn>=1.2.2',
        'xgboost=1.7.6',
        'numpy>=1.24.3'
    ],
    extras_require={
        'dev': [
            'pytest>=6.0'
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    url='https://github.com/CamiloDS16/capstone_project-amex-credit-default-',
    keywords='classification, american express, credit default, machine learning, data science',
)

