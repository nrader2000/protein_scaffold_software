# IMPORTANT IMPORTS
import os
from subprocess import call

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

# Software Window
class MainWindow(QMainWindow):
   def __init__(self):
      # Init of our Main Window
      super().__init__()
      self.setWindowTitle("Protein Scaffolding")
      width = 800
      self.setFixedWidth(width)
      self.setStyleSheet("background-color: rgb(16, 126, 222); color: white; ")
      
      # Window content
      self.label = QLabel("Please select a file to run") # Label with instruction
      self.label.setFont(QFont('Arial', 24))
      
      AEButton = QPushButton("autoencoder.py") # First button with autoencoder.py option
      AEButton.setFont(QFont('Arial', 24))
      AEButton.setStyleSheet("""
                             QPushButton{
                              background-color: #d9b100; 
                              border-style: outset; 
                              border-width: 2px; 
                              border-radius: 15px; 
                              border-color: black; 
                              padding: 10px;
                             }
                             QPushButton::hover{
                              background-color: #ffdd47
                             }""")
      AEButton.setCheckable(True)
      AEButton.clicked.connect(self.execute_autoencoder)

      MButton = QPushButton("main.py") # Second button with main.py option
      MButton.setFont(QFont('Arial', 24))
      MButton.setStyleSheet("""
                             QPushButton{
                              background-color: #d9b100; 
                              border-style: outset; 
                              border-width: 2px; 
                              border-radius: 15px; 
                              border-color: black; 
                              padding: 10px;
                             }
                             QPushButton::hover{
                              background-color: #ffdd47
                             }""")      
      MButton.setCheckable(True)
      MButton.clicked.connect(self.execute_main)

      layout = QVBoxLayout() # Layout with all of our widgets
      layout.setSpacing(20)
      layout.setContentsMargins(100,100,100,100)
      layout.addWidget(self.label,0,Qt.AlignmentFlag.AlignCenter)
      layout.addWidget(AEButton,0,Qt.AlignmentFlag.AlignCenter)
      layout.addWidget(MButton,0,Qt.AlignmentFlag.AlignCenter)
      

      container = QWidget() # Container that utlizes layout
      container.setLayout(layout)

      self.setCentralWidget(container) # Set the central widget to be the container  

   # Executes the autoencoder.py python script from the QinggeLab-ISBRA_2023 repo by Richard 
   def execute_autoencoder(self):
      if os.environ.get('PYTHONHASHSEED') == '1': # Checks if required environmental variable is set
         try: # Attemps to launch autoencoder.py via call function
            print("Launching autoencoder.py")
            call(["python","autoencoder.py"])
         except FileNotFoundError:
            print(f"Error: The file does not exist.")
      else: print("Environment Variable Not Set")

   # Executes the main.py python script from the protein_scaffold_filler repo by Jordan 
   def execute_main(self):
      if os.environ.get('PYTHONHASHSEED') == '1':
         try:
            print("Launching main.py")
            call(["python","main.py"])
         except FileNotFoundError:
            print(f"Error: The file does not exist.")
      else: print("Environment Variable Not Set")

# Sets up the environment variable beforehand and runs the main window for the software
def main():
   
   # Setup of Environment Variable
   os.environ["PYTHONHASHSEED"] = str(1)

   # Window Setup
   app = QApplication([])
   window = MainWindow()
   window.show()
   app.exec()

# Run main()
if __name__ == '__main__':
    main()