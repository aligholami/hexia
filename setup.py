from distutils.core import setup
import os.path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "hexia"))

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='hexia',
    version='0.0.1',
    description='Mid-level PyTorch Based Framework for Visual Question Answering',
    author='Ali Gholami',
    author_email='hexpheus@gmail.com',
    url='https://hexiadocs.readthedocs.io',
    install_requires=reqs
)