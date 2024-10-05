#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.install import install
from setuptools.command.develop import develop
import subprocess
import shutil
import os

class Build(build_py):
    def run(self):
        super(Build, self).run()
        self.execute_build()

    def execute_build(self):
        subprocess.check_call(['bash', './build/build.sh'])
        build_lib = self.build_lib
        src_dir = os.path.join(os.path.dirname(__file__), 'src')

        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.endswith('.so'):
                    source_file = os.path.join(root, file)
                    dest_file = os.path.join(build_lib, 'cuda_ops', os.path.relpath(source_file, src_dir))
                    self.copy_file(source_file, dest_file)

class Install(install):
    def run(self):
        self.reinitialize_command('build_py', inplace=1)
        self.run_command('build_py')
        install.run(self)


class Develop(develop):
    def install_for_development(self):
        self.reinitialize_command('build_py', inplace=1)
        self.run_command('build_py')
        super(Develop, self).install_for_development()
        self.copy_so_files()

    def copy_so_files(self):
        build_command = self.get_finalized_command('build')
        build_lib = build_command.build_lib

        root_dir = os.path.dirname(__file__)

        for root, _, files in os.walk(build_lib):
            for file in files:
                if file.endswith('.so'):
                    source_file = os.path.join(root, file)
                    relative_path = os.path.relpath(root, build_lib)
                    dest_dir = os.path.join(root_dir, relative_path)
                    if not os.path.exists(dest_dir):
                        os.makedirs(dest_dir)
                    print('shutil.copy2()', source_file, dest_dir)
                    shutil.copy2(source_file, dest_dir)


setup(
    name='cuda_ops',
    version='0.1',
    packages=find_packages(),
    cmdclass={
        'build_py': Build,
        'install': Install,
        'develop': Develop,
    },
    install_requires=[
        'numpy',
    ],
    requires=[
        'numpy'
    ],
    extras_require={
        'test': [
            'pytest',
        ]
    },
    package_data={
        'cuda_ops': ['*.so'],
        'test': ['*']
    }
)
