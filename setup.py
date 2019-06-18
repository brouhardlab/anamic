from setuptools import setup
from setuptools import find_packages


setup(name="anamic",
      version='0.2.2',
      author='Hadrien Mary',
      author_email='hadrien.mary@gmail.com',
      url='https://github.com/hadim/anamic/',
      description='Simulate, fit and analyze microtubules.',
      include_package_data=True,
      packages=find_packages(),
      classifiers=['Development Status :: 5 - Production/Stable',
                   'Natural Language :: English',
                   'License :: OSI Approved :: BSD License',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3'])
