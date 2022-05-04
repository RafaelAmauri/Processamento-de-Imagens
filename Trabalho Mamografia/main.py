from skimage.feature import graycomatrix, graycoprops
from skimage import io
from sklearn import svm

import os
import numpy as np
import matplotlib.pyplot as plt

'''
image = io.imread("Imagens/4/p_g_left_cc(12).png")

GLCM = graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])

print(graycoprops(GLCM, 'energy'))
'''

arquivos_1 = "./Imagens/1"

filelist = [file for file in os.listdir(arquivos_1) if file.endswith('.png')]

features = []
respostas = []

for image in filelist:
    image = io.imread(f"{arquivos_1}/{image}")
    glcm = graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
    energy = graycoprops(glcm, 'energy')[0,0]
    homo   = graycoprops(glcm, 'homogeneity')[0,0]

    features.append([energy * homo])
    respostas.append("1")

arquivos_2 = "./Imagens/2"

filelist = [file for file in os.listdir(arquivos_2) if file.endswith('.png')]

for image in filelist:
    image = io.imread(f"{arquivos_2}/{image}")
    glcm = graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
    energy = graycoprops(glcm, 'energy')[0,0]
    homo   = graycoprops(glcm, 'homogeneity')[0,0]

    features.append([energy * homo])
    respostas.append("2")

#print(features)

clf = svm.SVC()
clf.fit(features, respostas)

image_random = io.imread("random_1.png")
glm = graycomatrix(image_random, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
energy = graycoprops(glm, 'energy')[0,0]
homo   = graycoprops(glm, 'homogeneity')[0,0]

#print(clf.predict(np.array(energy).reshape(-1, 1)))
print(clf.predict(np.array(energy * homo).reshape(-1, 1)))