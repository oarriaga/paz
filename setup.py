from setuptools import setup
from setuptools import find_packages
import paz


if __name__ == "__main__":

    setup(name='pypaz',
          version=paz.__version__,
          description='Perception for Autonomous Systems',
          long_description='Perception for Autonomous Systems',
          author='Octavio Arriaga',
          author_email='octavio.arriaga@dfki.de',
          url='https://github.com/oarriaga/paz/',
          license='MIT',
          classifiers=[
              'Intended Audience :: Science/Research',
              'Topic :: Scientific/Engineering :: Artificial Intelligence',
              'Topic :: Scientific/Engineering :: Image Recognition',
              'Programming Language :: Python :: 3',
              'License :: OSI Approved :: MIT License'
          ],
          install_requires=['opencv-python', 'tensorflow', 'numpy'],
          packages=find_packages())
