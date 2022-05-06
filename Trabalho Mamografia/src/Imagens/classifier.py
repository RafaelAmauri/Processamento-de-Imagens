import os
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt

from skimage.feature import graycomatrix, graycoprops
from skimage import io
from sklearn import svm
from skimage.measure import shannon_entropy
from sklearn.model_selection import train_test_split

from PIL  import Image
import PIL

IMAGE_DIR = "./Imagens"
TRAIN_SIZE = 75

'''
Returns two sets containing the training and testing sets, respectively
@param train_size: an int from 0 to 100 indicating how much of the available
data will be used for training
'''
def split_train_test(train_size:int):
    train = []
    test  = []
    train_answers = []
    test_answers  = []

    supported_extensions = ['.jpg', '.jpeg', '.png']

    density_classes = ["1", "2", "3", "4"]#os.listdir(IMAGE_DIR)
    for density_class in density_classes:
        path_images = [file for file in os.listdir(f"{IMAGE_DIR}/{density_class}") if file.endswith(tuple(supported_extensions))]
        tmp_train, tmp_test = train_test_split(path_images, train_size=train_size, shuffle=True)

        for i in tmp_train:

            image       =  np.asarray(Image.open(f"{IMAGE_DIR}/{density_class}/{i}").quantize(32))
            
            glcm        =  graycomatrix(image, [1, 2, 4, 8, 16], [0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8])
            energy      =  graycoprops(glcm, 'energy')
            homogeneity =  graycoprops(glcm, 'homogeneity')
            asm         =  graycoprops(glcm, 'ASM')
            entropy     =  shannon_entropy(glcm, base=2)

            tmp = np.concatenate((energy, homogeneity, entropy), axis=None)

            train.append(tmp)
            train_answers.append(f"{density_class}")
        
        for i in tmp_test:
            image       =  np.asarray(Image.open(f"{IMAGE_DIR}/{density_class}/{i}").quantize(32))

            glcm        =  graycomatrix(image, [1, 2, 4, 8, 16], [0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8])
            energy      =  graycoprops(glcm, 'energy')
            homogeneity =  graycoprops(glcm, 'homogeneity')
            asm         =  graycoprops(glcm, 'ASM')
            entropy     =  shannon_entropy(glcm, base=2)
            
            tmp = np.concatenate((energy, homogeneity, entropy), axis=None)

            test.append(tmp)
            test_answers.append(f"{density_class}")

    return train, test, train_answers, test_answers

train, test, train_answers, test_answers = split_train_test(TRAIN_SIZE)

clf = svm.SVC(kernel="linear")
clf.fit(train, train_answers)

prediction = clf.predict(test)

print(sklearn.metrics.accuracy_score(test_answers, prediction))