#!/usr/bin/env python

import os
import sys
from setuptools import setup, find_packages

if sys.argv[-1] == 'install':
    setup(name='ittk',
          version='1.0.1',
          description='Information Theory Toolkit',
          author='Maxwell Rebo',
          author_email='maxwell.b.rebo@gmail.com',
          url='https://github.com/MaxwellRebo/ittk',
          packages=find_packages()
          )
