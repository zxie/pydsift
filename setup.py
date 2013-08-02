from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    Extension(
        "c_pydsift",
        ["src/c_pydsift.pyx"],
        language="c",
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
  name = 'pydsift',
  version='0.0.1',
  cmdclass = {'build_ext': build_ext},
  ext_package = 'pydsift',
  ext_modules = ext_modules,
  packages= ['pydsift'],
)
