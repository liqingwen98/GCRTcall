from setuptools import setup, find_packages

setup(
    name='GCRTcall',
    packages = find_packages(),
    version='1.0',
    description='GCRTcall: a Transformer based basecaller for nanopore RNA sequencing',
    author='Qingwen Li',
    author_email='arwinleecn@gmail.com',
    url='https://github.com/liqingwen98/BaseNet.git',
    install_requires=[
        'funasr==0.7.0',
        'torch',
    ],
    keywords=['nanopore sequencing', 'basecalling', 'transformer', 'end-to-end'],
    python_requires='>=3.8'
)