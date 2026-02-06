"""
File: compileC.py
Author: Yaofeng Chen, Xuanchen Zhang
Organization: Tsinghua University

Description:

    Backend code to compile the C codes for Lanczos-based implementations.

    Run using this command in command line:

        python compileC.py build_ext --inplace

License:
MIT License

"""

from setuptools import setup, Extension
import numpy

# zeroize.c
module = Extension('zeroize',
                   sources=['zeroize.c'],
                   include_dirs=[numpy.get_include()],  # Add include directory of NumPy
                   extra_compile_args=['-fopenmp'],
                   extra_link_args=['-fopenmp'])
setup(name='zeroize',
      version='1.0',
      description='This is a package with a C extension for zero-izing all input arrays.',
      ext_modules=[module])

# getSz.c
module = Extension('getSz',
                   sources=['getSz.c'],
                   include_dirs=[numpy.get_include()],  # Add include directory of NumPy
                   extra_compile_args=['-fopenmp'],
                   extra_link_args=['-fopenmp'])
setup(name='getSz',
      version='1.0',
      description='This is a package with a C extension for getting Sz.',
      ext_modules=[module])

# subLanczos.c
module = Extension('subLanczos',
                   sources=['subLanczos.c'],
                   include_dirs=[numpy.get_include()],  # Add include directory of NumPy
                   extra_compile_args=['-fopenmp'],
                   extra_link_args=['-fopenmp'])
setup(name='subLanczos',
      version='1.0',
      description='This is a package with a C extension for applying a sub-part of a Lanczos loop.',
      ext_modules=[module])

# subLanczosJ0.c
module = Extension('subLanczosJ0',
                   sources=['subLanczosJ0.c'],
                   include_dirs=[numpy.get_include()],  # Add include directory of NumPy
                   extra_compile_args=['-fopenmp'],
                   extra_link_args=['-fopenmp'])
setup(name='subLanczosJ0',
      version='1.0',
      description='This is a package with a C extension for applying a sub-part of a Lanczos loop with J0.',
      ext_modules=[module])

# getPred.c
module = Extension('getPred',
                   sources=['getPred.c'],
                   include_dirs=[numpy.get_include()],  # Add include directory of NumPy
                   extra_compile_args=['-fopenmp'],
                   extra_link_args=['-fopenmp'])
setup(name='getPred',
      version='1.0',
      description='This is a package with a C extension for getting predictions.',
      ext_modules=[module])