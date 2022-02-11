from setuptools import setup
from setuptools import find_packages
import paz

setup(name='pypaz',
      version='0.1.7',
      description='Perception for Autonomous Systems',
      author='Octavio Arriaga',
      author_email='octavio.arriaga@dfki.de',
      license='MIT',
      install_requires=['opencv-python', 'tensorflow', 'numpy'],
      packages=find_packages())
