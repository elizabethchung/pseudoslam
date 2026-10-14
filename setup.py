from setuptools import setup, find_packages

setup(name='pseudoslam',
      version='0.0.1',
      description='Learning-based obstacle detection from 2D LiDAR, built on the PseudoSLAM simulator',
      packages=find_packages(include=["pseudoslam", "pseudoslam.*"]),
      install_requires=['torch', 'numpy', 'scipy', 'pyyaml', 'matplotlib', 'opencv-python', 'gym']
      )
