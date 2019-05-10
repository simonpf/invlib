import os
import pathlib
import shutil

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig

class CMakeExtension(Extension):

    def __init__(self, name, sources = []):
        super().__init__(name, sources = sources)

class build_ext(build_ext_orig):

    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents = True, exist_ok = True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        print("EXTDIR: ", extdir)
        extdir.mkdir(parents = True, exist_ok = True)

        # example of cmake args
        config = 'Debug' if self.debug else 'Release'

        cmake_args = [
            '-DCMAKE_BUILD_TYPE=' + config,
            '-DPYTHON_OUTPUT_DIR=' + os.path.join(str(cwd), ".")
        ]

        # example of build args
        build_args = [
            '--config', config,
            '--', '-j4'
        ]

        os.chdir(str(build_temp))
        self.spawn(['cmake', str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.', '--target', 'invlib_cpu_float'] + build_args)
            self.spawn(['cmake', '--build', '.', '--target', 'invlib_cpu_double'] + build_args)
            self.spawn(['cmake', '--build', '.', '--target', 'invlib_mpi_float'] + build_args)
            self.spawn(['cmake', '--build', '.', '--target', 'invlib_mpi_double'] + build_args)
        os.chdir(str(cwd))

import glob

invlib_files = glob.glob("src/**/*.h", recursive = True)
invlib_files += glob.glob("src/**/*.cpp", recursive = True)

setup(
    name        = 'invlib',
    version     = '0.1.0',
    packages    = ['invlib', 'invlib.api'],
    ext_modules = [CMakeExtension('invlib', sources = invlib_files)],
    cmdclass    = {'build_ext': build_ext},

    author       = "Simon Pfreundschuh",
    author_email = "simon.pfreundschuh@chalmers.se",
    description  = "Maximum a posteriori estimators for Bayesian inverse " \
    " problems.",
    long_description = "This package is an interface to the invlib library "
    " which provides efficient maximum a posteriori (MAP) solvers for "\
    " Bayesian inverse problems in remote sensing, also known as the optimal "\
    "estimation method (OEM) or XDVAR.",
    license          = "PSF",
    install_requires = ["numpy", "scipy"],
    keywords         = "Bayesian, MAP, OEM, remote sensing, inverse problems",
    url              = "https://github.com/simonpf/invlib",
    project_urls = {
        "Source Code": "https://github.com/simonpf/invlib",
    },
    classifiers = ['Development Status :: 3 - Alpha',
                   'Intended Audience :: Science/Research',
                   'Topic :: Scientific/Engineering :: Atmospheric Science',
                   'License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 3.8'],
    python_requires='>=3'
)
