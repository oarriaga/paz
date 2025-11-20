from setuptools import setup, find_packages

setup(
    name="paz",
    version="0.3.0",
    packages=find_packages(),
    description="Perception for Autonomous Systems (JAX branch)",
    author="Octavio Arriaga",
    # JAX requires manual installation to chose between CPU and GPU.
    install_requires=["numpy", "opencv-python"],
)
