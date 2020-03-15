# Copyright 2020 Ankur Gupta
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#

from setuptools import setup

PACKAGE_NAME = 'flicker'

# Read-in the version
# See 3 in https://packaging.python.org/guides/single-sourcing-package-version/
version_file = './{}/version.py'.format(PACKAGE_NAME)
version = {}
try:
    # Python 2
    execfile(version_file, version)
except NameError:
    # Python 3
    exec(open(version_file).read(), version)

# Read-in the README.md
with open('README.md', 'r') as f:
    readme = f.readlines()
readme = ''.join(readme)

setup(name=PACKAGE_NAME,
      version=version['__version__'],
      url='https://github.com/ankur-gupta/flicker',
      author='Ankur Gupta',
      author_email='ankur@perfectlyrandom.org',
      description=('Provides FlickerDataFrame, a wrapper over '
                   'Pyspark DataFrame to provide a pandas-like API'),
      long_description=readme,
      long_description_content_type="text/markdown",
      keywords='pyspark, pandas',
      packages=[PACKAGE_NAME,
                '{}.tests'.format(PACKAGE_NAME)],
      include_package_data=True,
      install_requires=['six', 'pandas', 'numpy', 'pyspark'],
      setup_requires=['pytest-runner'],
      # pytest-cov needed for coverage only
      tests_require=['pytest', 'pytest-cov'],
      zip_safe=True)
