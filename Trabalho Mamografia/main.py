from frontend import front
import os

# Current working dir
# JPEG and PNG types
types = [('Images', '*.jpeg *.png *.jpg')]

gui = front.GUI(os.getcwd())
gui.drawGUI()