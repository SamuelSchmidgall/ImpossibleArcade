from setuptools import setup

setup(name='ImpossibleArcade',
      version='0.0.1',
      description='Simple Arcade with AI that are (impossible) to beat [CS321 Semester Project]',
      url='http://github.com/SamuelSchmidgall/ImpossibleArcade',
      author='Samuel Schmidgall, ',
      author_email='sschmidg@gmu.edu',
      license='MIT',
      packages=['impossible_arcade'],
      install_requires=[
          "numpy",
          "pygame", 'torch'
      ],
      zip_safe=False)
