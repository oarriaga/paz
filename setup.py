from setuptools import setup
from setuptools import find_packages

setup(name='Paz',
      version='0.1',
      description='Perception for Autonomous Systems',
      author='Octavio Arriaga',
      author_email='octavio.arriaga@dfki.de',
      license='MIT',
      install_requires=[
          'opencv-python',
          'tensorflow'],
      packages=find_packages()
      )
