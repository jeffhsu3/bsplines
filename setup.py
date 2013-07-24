import os, sys, glob

name = 'bsplines'
version = '0.1'

from distutils.core import setup
from distutils.extension import Extension

metadata = {'name': name,
            'vestion': version,
            'description': 'bsplines for python',
            'packages': ['bspline'],
           }



if __name__ == '__main__':
    dist = setup(**metadata)

