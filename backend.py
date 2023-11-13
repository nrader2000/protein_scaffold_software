import sys
import os
from subprocess import call

from PyQt6.QtGui import QGuiApplication
from PyQt6.QtQml import QQmlApplicationEngine
from PyQt6.QtQuick import QQuickWindow

# Executes the autoencoder.py python script from the QinggeLab-ISBRA_2023 repo by Richard 
def execute_autoencoder():
   try:
      print("Launching autoencoder.py")
      call(["python","autoencoder.py"])
   except FileNotFoundError:
      print(f"Error: The file does not exist.")
# Executes the main.py python script from the protein_scaffold_filler repo by Jordan 
def execute_main():
   try:
      print("Launching main.py")
      call(["python","main.py"])
   except FileNotFoundError:
      print(f"Error: The file does not exist.")
# Gives the user the opetion to run either file and then executes whichever choice
def main():
   # Setup of Environment Variable
   os.environ["PYTHONHASHSEED"] = str(1)
   if os.environ.get('PYTHONHASHSEED') == '1':
      # Menu for Python Files
      file = int(input('Please select a Python File the Run: \n1-autoencoder.py \n2-main.py\n'))
      if file == 1:
         execute_autoencoder()
      elif file == 2: 
        execute_main()
      else: print("Invalid Input")
   else: print("Environment Variable Not Set")

# Run main()
if __name__ == '__main__':
    main()