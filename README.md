This repository contains the software exercises and final project for the ThoughtBridge 2020 Summer Deep Learning Online Program.

## Setup
Using the command line terminal, install the following dependencies: [conda 3.x](https://docs.conda.io/en/latest/miniconda.html)

## Installation of dependencies
This code is tested and runs on Ubuntu 18.04. All dependencies exist on other operating systems but their installation may differ from the instructions provided here. Most of the dependencies for the software exercises are captured in a conda environment. To use the exercise toolbox, first create the conda environment and install the package.
```
cd thoughtbridge
conda env create -f environment.yml
conda activate introtodeeplearning
pip install -e .
```

Now install some final dependencies:
```
sudo add-apt-repository multiverse
sudo apt-get install abcmidi timidity ubuntu-restricted-extras
```

## Usage and running the labs
Enter the environment by running `conda activate introtodeeplearning`.

Once inside the environment, the package can be directly imported and used in a Python shell/script: 
```
>>> import introtodeeplearning as mdl
```
This is taken care of in each of the iPython notebook labs. 

The 2020 exercises can be run in the Jupyter notebook environment. After you have followed the above instructions to install the conda environment and have activated the conda environment, run `jupyter notebook` to open up a browser connection to jupyter and open up the lab notebook you want to complete.

Go through the notebooks and fill in the `#TODO` cells to get the code to compile for yourself!

On this Github repo, navigate to the exercise folder you want to run (`1_IntroDeepLearning`, `2_RecurrentNN`, etc.) and open the appropriate python notebook (\*.ipynb) using `jupyter notebook` from the command line, after activating the environment.

Once you are done working, you can exit out of the environment using `conda deactivate`.

## Authorship
All code in this repository is created by Alexander Amini and Ava Soleimany.
