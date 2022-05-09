import os
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt

from skimage.feature import graycomatrix, graycoprops
from skimage import io
from sklearn import svm
from skimage.measure import shannon_entropy
from sklearn.model_selection import train_test_split

from PIL  import Image, ImageEnhance

IMAGE_DIR = "./"
TRAIN_SIZE = 75

'''
Returns two sets containing the training and testing sets, respectively
@param train_size: an int from 0 to 100 indicating how much of the available
data will be used for training
'''
def split_train_test(train_size:int, contrast_factor, sharpness_factor, color_factor, brightness_factor):
    train = []
    test  = []
    train_answers = []
    test_answers  = []
    '''
    contrast_factor   =  4
    sharpness_factor  =  2
    color_factor      =  1
    brightness_factor =  2
    '''
    supported_extensions = ['.jpg', '.jpeg', '.png']

    density_classes = ["1", "2", "3", "4"]
    for density_class in density_classes:
        path_images = [file for file in os.listdir(f"{IMAGE_DIR}/{density_class}") if file.endswith(tuple(supported_extensions))]
        tmp_train, tmp_test = train_test_split(path_images, train_size=train_size, shuffle=True)

        for i in tmp_train:

            image       =  Image.open(f"{IMAGE_DIR}/{density_class}/{i}")
            
            enhancer_sharpness = ImageEnhance.Sharpness(image)
            image = enhancer_sharpness.enhance(sharpness_factor)
            enhancer_contrast = ImageEnhance.Contrast(image)
            image = enhancer_contrast.enhance(contrast_factor)
            enhancer_color = ImageEnhance.Color(image)
            image= enhancer_color.enhance(color_factor)
            enhancer_brightness = ImageEnhance.Brightness(image)
            image= enhancer_color.enhance(brightness_factor)

            image = np.asarray(image.quantize(32))
            glcm        =  graycomatrix(image, [1, 2, 4, 8, 16], [0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8], levels=32)
            energy        =  graycoprops(glcm, 'energy')
            homogeneity   =  graycoprops(glcm, 'homogeneity')
            asm           =  graycoprops(glcm, 'ASM')
            dissimilarity =  graycoprops(glcm, 'dissimilarity')
            correlation   =  graycoprops(glcm, 'correlation')
            contrast      =  graycoprops(glcm, 'contrast')
            entropy       =  shannon_entropy(glcm, base=2)

            tmp = np.concatenate((energy, homogeneity, entropy, correlation, dissimilarity, asm, contrast), axis=None)
            train.append(tmp)
            train_answers.append(f"{density_class}")
        
        for i in tmp_test:
            image       =  Image.open(f"{IMAGE_DIR}/{density_class}/{i}")
            
            enhancer_sharpness = ImageEnhance.Sharpness(image)
            image = enhancer_sharpness.enhance(sharpness_factor)
            enhancer_contrast = ImageEnhance.Contrast(image)
            image = enhancer_contrast.enhance(contrast_factor)
            enhancer_color = ImageEnhance.Color(image)
            image= enhancer_color.enhance(color_factor)
            enhancer_brightness = ImageEnhance.Brightness(image)
            image= enhancer_color.enhance(brightness_factor)

            image = np.asarray(image.quantize(32))

            glcm        =  graycomatrix(image, [1, 2, 4, 8, 16], [0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8], levels=32)
            energy        =  graycoprops(glcm, 'energy')
            homogeneity   =  graycoprops(glcm, 'homogeneity')
            asm           =  graycoprops(glcm, 'ASM')
            dissimilarity =  graycoprops(glcm, 'dissimilarity')
            correlation   =  graycoprops(glcm, 'correlation')
            contrast      =  graycoprops(glcm, 'contrast')
            entropy       =  shannon_entropy(glcm, base=2)

            tmp = np.concatenate((energy, homogeneity, entropy, correlation, dissimilarity, asm, contrast), axis=None)

            test.append(tmp)
            test_answers.append(f"{density_class}")

    return train, test, train_answers, test_answers

valid_contrast = []
valid_sharpness= []
valid_color    = []
valid_bright   = []

'''
for i in range(100, 500, 25):
    valid_contrast.append(i/100)
    valid_sharpness.append(i/100)
    valid_color.append(i/100)
    valid_bright.append(i/100)

for i in valid_contrast:
    for j in valid_sharpness:
        for k in valid_color:
            for v in valid_bright:

                print(f"contrast = {i} -- sharp = {j} -- color = {k} -- bright = {v}\n")
'''
train, test, train_answers, test_answers = split_train_test(TRAIN_SIZE, 1, 1, 3, 3)

clf = svm.SVC(kernel="linear")
clf.fit(train, train_answers)

prediction = clf.predict(test)

accuracy = sklearn.metrics.accuracy_score(test_answers, prediction)
confusion_matrix = sklearn.metrics.confusion_matrix(test_answers, prediction)

print(f"Accuracy = {accuracy}")
print(f"Matriz de confusão = {confusion_matrix}")
   
plt.matshow(confusion_matrix, fignum="int")

for (i, j), z in np.ndenumerate(confusion_matrix):
    plt.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

plt.text(0.5, -1, f"Accuracy = {accuracy}", ha='center', va='center', fontsize=16,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

plt.show()