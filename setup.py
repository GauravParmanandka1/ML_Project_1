from setuptools import find_packages,setup
##from typing import list # In Python 3.11 'list' is a built-in type and importing it from typing is incorrect
## Metadata Information about the project

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->list[str]:
    '''
    this function will return the list of required libraries from the file provided
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    print(requirements)
    return requirements
    
setup(
name='GitHub_Projects_Repo',
version='0.0.1',
author='Gaurav Parmanandka',
author_email='gaurav.parmanandka@gmail.com',
packages=find_packages(), ##searches __init__.py file in all directories and try to build that as a package
##install_requires=['pandas','numpy','matplotlib','seaborn','scikit-learn','jupyter','jupyterlab']
install_requires=get_requirements('requirements.txt') ## To fetch the library requirements from the requirements.txt file
)