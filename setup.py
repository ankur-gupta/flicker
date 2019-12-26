from setuptools import setup

PACKAGE_NAME = 'flicker'

# Read-in the version
version_file = './{}/version.py'.format(PACKAGE_NAME)
try:
    # Python 2
    execfile(version_file)
except NameError:
    # Python 3
    exec(open(version_file).read())

# Read-in the README.md
with open('README.md', 'r') as f:
    readme = f.readlines()
readme = ''.join(readme)

setup(name=PACKAGE_NAME,
      version=__version__,
      url='https://github.com/ankur-gupta/flicker',
      author='Ankur Gupta',
      author_email='ankur@perfectlyrandom.org',
      description=('Provides FlickerDataFrame, a wrapper over '
                   'Pyspark DataFrame to provide a pandas-like API'),
      long_description=readme,
      keywords='pyspark, pandas',
      packages=[PACKAGE_NAME],
      include_package_data=True,
      install_requires=['six', 'pandas', 'numpy', 'pyspark'],
      setup_requires=['pytest-runner'],
      # pytest-cov needed for coverage only
      tests_require=['pytest', 'pytest-cov'],
      zip_safe=True)
