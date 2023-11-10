# protein_scaffold_software
This repo contains my work to combine two existing repos that have done some work with the protein scaffolding project into a piece of executable software.

## Setup
* Install miniconda from https://docs.conda.io/projects/miniconda/en/latest/index.html (version during development was 23.9.0). You'll run into a screen that gives you some options, check the ones that are reccomended. Once you get this fully setup, go into the miniconda CLI use the command ```conda --version``` to confirm you have miniconda setup.
* Setup a python environment that you will install your dependencies in the miniconda CLI:
  - Create your env using the command: ```conda create -n tfenv python=3.9``` (tfenv can be whatever name you deem fit for your env)
  - Activate your new env using the command: ```conda activate tfenv```
  - Install pip using the command: ```conda install pip``` Then upgrade it using the command ```python -m pip install --upgrade pip```
  - Install cudatoolkit and cuDNN using the command: ```conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0```
  - Install tensorflow by using the command: ```pip install "tensorflow<2.11"```
  - Install other libraries in order using the commands: ```pip install matplotlib```, ```pip install PyMuPDF```, ```pip install reportlab```, ```pip install -U scikit-learn```

## Running the Application
* Once this is setup all you have to do is run ```python backend.py``` in the tfevn environment.
