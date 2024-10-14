from setuptools import setup, find_packages

setup(
    name='EasyTorch',
    version='0.1.0',
    author='Jaylon Nelson-Sellers',
    author_email='jaylonnelsonsellers@gmail.com',
    description='A wrapper of DL models with Sci-Kit Learn ',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    install_requires=[
        'astropy==6.0.1',
        'keras==3.2.1',
        'matplotlib==3.8.4',
        'numpy==1.26.4',
        'pandas==2.2.2',
        'scikit_learn==1.4.1.post1',
        'scipy==1.13.1',
        'setuptools==69.1.1',
        'skorch==0.15.0',
        'ta==0.11.0',
        'torch==2.2.0+cu121',
        'torch==2.2.2+cu118',
        'yfinance==0.2.37',
    ],
)
