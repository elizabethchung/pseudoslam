from setuptools import setup, find_packages

setup(name='pseudoslam',
      version='0.0.1',
      description='A reinforcement learning environment for robot autonomous exploration',
      packages=find_packages(include=["pseudoslam", "pseudoslam.*"]),
      install_requires=['gym', 'numpy', 'pyyaml', 'matplotlib', 'opencv-python']
      )
