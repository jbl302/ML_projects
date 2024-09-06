from setuptools import find_packages , setup
from typing import List
def get_requirements(filePath:str) -> List[str]:
  '''This function gets the requirements from the 
  requirements.txt file'''
  requirements = []
  with open(filePath,'r') as file:
    requirements = file.readlines()
    requirements = [req.replace('\n','')for req in requirements] # To remove \n
    
    if '-e .' in requirements:
      requirements.remove('-e .')
  return requirements
  
setup(
  name='MlProject',
  version='0.0.1',
  author='Jai',
  author_email='jaibhagatnet@gmail.com',
  packages = find_packages(),
  install_requires = get_requirements('requirements.txt')
)