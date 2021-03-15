from pkg_resources import DistributionNotFound, get_distribution
from distutils.core import setup


def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None

install_deps = [
    'numpy',
    'regex',
    'tqdm',
    'gym'

]
tf_ver = '2.0.0a'
if get_dist('tensorflow>='+tf_ver) is None and get_dist('tensorflow_gpu>='+tf_ver) is None:
    install_deps.append('tensorflow>='+tf_ver)

setup(
  name = 'introtodeeplearning',         # How you named your package folder (MyLib)
  packages = ['introtodeeplearning'],   # Chose the same as "name"
  version = '0.1.2',      # Start with a small number and increase it with every change you make
  author = 'Alexander Amini, Ava Soleimany',                   # Type in your name
  keywords = ['deep learning', 'neural networks', 'tensorflow', 'introduction'],   # Keywords that define your package best
  install_requires=install_deps,
  classifiers=[
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support',
    'Programming Language :: Python :: 3.6',
  ],
  package_data={
      'introtodeeplearning': ['bin/*', 'data/*', 'data/faces/DF/*', 'data/faces/DM/*', 'data/faces/LF/*', 'data/faces/LM/*'],
   },

)
