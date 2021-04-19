import glob
import os
import pathlib
from distutils.command.build import build
from distutils.command.install import install as DistutilsInstall
from subprocess import call

from setuptools import setup, find_packages, Extension

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()


class MyInstall(DistutilsInstall):

    def do_pre_install_stuff(self):
        cmd = [
            'make'
            # 'OUT=' + build_path,
            # 'V=' + str(self.verbose),
        ]

        # try:
        #     cmd.append(' -j')
        # except NotImplementedError:
        #     print('Unable to determine number of CPUs. Using single threaded make.')

        # options = [
        #     'DEBUG=n',
        #     'ENABLE_SDL=n',
        # ]
        # cmd.extend(options)
        # targets = ['python']
        # cmd.extend(targets)
        call(cmd, cwd=os.path.join(HERE, 'culane'))

    def run(self):
        DistutilsInstall.run(self)
        self.do_pre_install_stuff()
        target_file = os.path.join(HERE, 'culane', 'evaluate')
        self.move_file(target_file, self.install_lib)


setup(
    name='Haomo Eval Scripts',
    # some version number you may wish to add - increment this after every update
    version='0.1a0',
    long_description=README,
    long_description_content_type="text/markdown",
    author="Jian QIN",
    install_requires=[],
    python_requires='>=3',
    scripts=['bin/hm_eval_lane'],
    ext_modules=[
        Extension('culane',
                  glob.glob(os.path.join(HERE, 'culane/src', '*.cpp')),
                  libraries=['rt'],
                  include_dirs=[os.path.join(HERE, 'culane/include'), os.path.join(HERE, 'culane/getopt')])
    ],
    cmdclass={
        'install': MyInstall,
    },
    packages=find_packages(),  # include/exclude arguments take * as wildcard, . for any sub-package names
)
