import os
import pathlib
import shutil

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig

class CMakeExtension(Extension):

    def __init__(self, name):
        super().__init__(name, sources=[])

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
            '-DPYTHON_OUTPUT_DIR=' + os.path.join(str(cwd), "invlib")
        ]

        # example of build args
        build_args = [
            '--config', config,
            '--', '-j4'
        ]

        os.chdir(str(build_temp))
        self.spawn(['cmake', str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.', '--target', 'invlib'] + build_args)
        os.chdir(str(cwd))

setup(
    name='invlib',
    version='0.1',
    packages=['invlib'],
    ext_modules=[CMakeExtension('invlib/invlib')],
    cmdclass={'build_ext': build_ext},

    # metadata to display on PyPI
    author="Simon Pfreundschuh",
    author_email="simon.pfreundschuh@chalmers.se",
    description="Python interface for the invlib library.",
    license="PSF",
    keywords="OEM",
    url="https://github.com/simonpf/invlib",
    project_urls={
        "Source Code": "https://github.com/simonpf/invlib",
    }
)
