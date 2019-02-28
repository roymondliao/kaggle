'''Cloud ML Engine package configuration.'''
from setuptools import setup
from setuptools import find_packages

REQUIRED_PACKAGES = ['h5py', 'lightgbm', 'scikit-optimize', 'xgboost', 'pandas==0.22.0', 'catboost']
# 'scikit-learn>=0.19.1', 'keras'

setup(name='kaggle',
      version='1.0',
      packages=find_packages(),
      include_package_data=True,
      description='Kaggle Competitions',
      author='Yuyu',
      author_email='yuyuliao@onead.com.tw',
      license='MIT',
      install_requires=REQUIRED_PACKAGES,
      zip_safe=False)
