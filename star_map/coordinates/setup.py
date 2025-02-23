from distutils.core import setup
from distutils.extension import Extension
import numpy
import os

cython_environment_variable = 'USE_CYTHON'
USE_CYTHON = True#bool(int(os.environ.get(cython_environment_variable, None)))

include_gsl_dir = '/usr/local/include/gsl'
lib_gsl_dir = '/usr/local/lib'
gsl_libs = ['gsl','gslcblas']
inc_dirs = [numpy.get_include(), include_gsl_dir]

suffix = '.pyx' if USE_CYTHON else '.c'

extensions = [
        Extension(
                'coordinates.orbital_parameters',
                [
                    'coordinates.orbital_parameters' + suffix,
                    'orbital_parameters.c'],
                libraries=gsl_libs,
                include_dirs=inc_dirs,
                library_dirs=[lib_gsl_dir])
              ]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)
    print('Using Cython')

setup(ext_modules=extensions, zip_safe=False)
        
