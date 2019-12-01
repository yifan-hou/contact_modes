from setuptools import setup, find_packages

setup(name='contact_modes',
      version='1.0',
      description='Contact Mode Enumeration',
      url='http://github.com/ehuang3/contact_modes',
      author='Eric Huang',
      author_email='erich1@andrew.cmu.edu',
      license='',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      scripts=[],
      install_requires=[],
      zip_safe=False)
