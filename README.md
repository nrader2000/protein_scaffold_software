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
  - Install other libraries in order using the commands: ```pip install matplotlib```, ```pip install PyMuPDF```, ```pip install reportlab```, ```pip install -U scikit-learn```, ```pip install PyQt6```

## Running the Application
* Once this is setup all you have to do is run ```python software-runner.py``` in the tfenv environment that you've done all the previous setup in.

## How the Software *currently* Works
* This software simply launches the two major files from the repos this software is suppoed to work with. To launch either of the files, all you have to do is click on one of the buttons named after the file you would like to run.

## Future Planned Work
* Add some labels to show how long each file should take to complete ✅
* Get a properly working codebase for autoencoder.py (currently runs into a division by 0 error)
* Attempt to have the ```software-runner``` script perform the main setup of the python development so users do not have to.
* Attempt to have the ```software-runner``` script become it's own executable so there is no need to need to interact with any python CLI.
* Attempt to have the ```software-runner``` script report to the user with an alert when the file is finished with execution so the user knows to look inside the results folder ✅. 
