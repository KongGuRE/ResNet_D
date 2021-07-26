from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

sourcefiles = ['CythonPyTorchResNet.pyx']

setup(
    ext_modules = cythonize(sourcefiles)
)