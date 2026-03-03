from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "P_matrix._utils",
        ["P_matrix/_utils.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "P_matrix.Pmatrix",
        ["P_matrix/Pmatrix.py"],
        include_dirs=[numpy.get_include()],
    ),
]

setup(
    name="P_matrix",
    packages=["P_matrix"],
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)