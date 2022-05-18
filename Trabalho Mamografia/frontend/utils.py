from skimage import io
from matplotlib import pyplot as plt

# Showing image to the user
def showImage(image_path):
    io.imshow(image_path)
    plt.show()
    
def read_file(filepath):
    with open(filepath) as f:
        text = f.read()
    
    return text