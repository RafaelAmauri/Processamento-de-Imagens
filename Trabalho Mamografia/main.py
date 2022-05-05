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

    density_classes = ["1", "2", "3", "4"]#os.listdir(IMAGE_DIR)
    for density_class in density_classes:
        pictures = [file for file in os.listdir(f"{IMAGE_DIR}/{density_class}") if file.endswith('.png')]
        tmp_train, tmp_test = train_test_split(pictures, train_size=train_size, shuffle=True)

        for i in tmp_train:
            # Nao vamos mais importar a img assim, porque precisa mudar o numero de cinzas na imagem
            #image       =  io.imread(f"{IMAGE_DIR}/{density_class}/{i}")

            image        =  np.asarray(Image.open(f"{IMAGE_DIR}/{density_class}/{i}").quantize(32))

            glcm        =  graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
            energy      =  graycoprops(glcm, 'energy')[0,0]
            homogeneity =  graycoprops(glcm, 'homogeneity')[0,0]
            #corr        =  graycoprops(glcm, 'correlation')[0,0]
            entropy     =  shannon_entropy(image, base=2)
            

            train.append([energy, homogeneity, entropy, corr])
            train_answers.append(f"{density_class}")
        
        for i in tmp_test:
            image       =  io.imread(f"{IMAGE_DIR}/{density_class}/{i}")
            glcm        =  graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
            energy      =  graycoprops(glcm, 'energy')[0,0]
            homogeneity =  graycoprops(glcm, 'homogeneity')[0,0]
            #corr        =  graycoprops(glcm, 'correlation')[0,0]
            entropy     =  shannon_entropy(image, base=2)

            test.append([energy, homogeneity, entropy, corr])
            test_answers.append(f"{density_class}")

    return train, test, train_answers, test_answers

train, test, train_answers, test_answers = split_train_test(TRAIN_SIZE)

'''
Vale a pena investigar se o modelo que é ruim
'''
clf = svm.SVC()
clf.fit(train, train_answers)

prediction = clf.predict(test)

print(sklearn.metrics.precision_score(test_answers, prediction, average='micro'))
