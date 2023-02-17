from setuptools import setup
from setuptools import find_packages
import paz


if __name__ == "__main__":
    with open("README.md", "r") as f:
        long_description = f.read()
    setup(name='pypaz',
          version=paz.__version__,
          description='Perception for Autonomous Systems',
          long_description=long_description,
          author='Octavio Arriaga',
          author_email='octavio.arriaga@dfki.de',
          url='https://github.com/oarriaga/paz/',
          license='MIT',
          install_requires=['opencv-python', 'tensorflow', 'numpy'],
          packages=find_packages())
