from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow==1.15.0','pandas==0.23.1','setuptools==38.7.0','numpy==1.14.1','Keras==2.1.4','scikit_learn==0.19.1','h5py']

setup(
    name='regressor',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.',
    author='Divyam Madaan',
    author_email='divyam3897@gmail.com',
    license='MIT',
    zip_safe=False
)
