import os, time

import sklearn.metrics
from skimage.feature import graycomatrix, graycoprops
from skimage import io
from sklearn import svm

from skimage.measure import shannon_entropy
from sklearn.model_selection import train_test_split

import numpy as np

from matplotlib import pyplot as plt

from PIL import Image, ImageFilter, ImageEnhance


class ImageClassifier:
    # Modelo para previsões
    __model: svm.SVC()
    
    # Kernel do modelo
    __model_kernel: str

    # Descritores de textura
    __texture_descriptors: list

    ## Arrays com os conjuntos de treino e teste, bem como as respostas
    __images_train:  np.empty(0, dtype=np.float64)
    __images_test:   np.empty(0, dtype=np.float64)
    __predictions:   np.empty(0, dtype=np.float64)
    __answers_train: np.empty(0, dtype=np.int32)
    __answers_test:  np.empty(0, dtype=np.int32)
    
    # Diretorio onde estão as pastas de classes BIRADS
    __images_dir: str

    # Tipos de arquivos
    __supported_img_extensions: list

    # Porcentagem de dados para treino. É balanceado igualmente entre
    # as classses BIRADS
    __percentage_train: int

    # Numero de cores
    __n_colors: float
    
    # Raio da matriz de suavização gaussiana. Só aceita inteiros. 0 desabilita
    __gaussian_radius: int
    
    # Em quanto cada um desses elementos será realçado
    # no pré-processamento. 1 os mantém igual, valores de 0 a 1
    # os reduzem e valores maiores que 1 aumentam
    __sharpness_boost_strength:  float
    __contrast_boost_strength :  float
    __brightness_boost_strength: float

    # Tempos de execução de diferentes funções são armazenadas aqui
    __runtime_metrics: dict

    # Distancias na matriz de co-ocorrencia
    __distances_glcm: np.empty(0, dtype=np.int32)

    # Angulos para calcular matriz de co-ocorrencia
    __angles_glcm: np.empty(0, dtype=np.float64)


    def __init__(self):
        self.__model_kernel = "linear"
        self.__model = svm.SVC(kernel=self.get_model_kernel(), C=1.4)

        self.__images_train  = np.empty(0, dtype=np.float64)
        self.__images_test   = np.empty(0, dtype=np.float64)
        self.__predictions   = np.empty(0, dtype=np.float64)
        self.__answers_train = np.empty(0, dtype=np.int32)
        self.__answers_test  = np.empty(0, dtype=np.int32)
        
        self.__supported_img_extensions = ['.jpg', '.jpeg', '.png']

        # Valores default. Outros valores podem ser escolhidos pela interface e pelo metodo set :)
        self.__n_colors = 32
        self.__gaussian_radius = 0
        self.__sharpness_boost_strength  = 1.4
        self.__contrast_boost_strength   = 2.2
        self.__brightness_boost_strength = 1.1
        self.__percentage_train = 0.75

        self.__runtime_metrics = {}

        self.__distances_glcm = np.array([
            1, 
            2, 
            4, 
            8, 
            16
        ])

        self.__angles_glcm = np.array([
            0, 
            np.pi/8,
            np.pi/4, 
            3*np.pi/8, 
            np.pi/2, 
            5*np.pi/8, 
            3*np.pi/4, 
            7*np.pi/8
        ])

        self.__texture_descriptors = np.array([
            "energy", 
            "homogeneity",
            "correlation",
            "dissimilarity"
        ])
        

    # Gets e Sets :)
    def set_model_kernel(self, k: str):
        self.__model_kernel = k
    
    def get_model_kernel(self):
        return self.__model_kernel

    def set_n_colors(self, n_colors: int):
        self.__n_colors = n_colors
    
    def get_n_colors(self):
        return self.__n_colors

    def set_gaussian_radius(self, strength: int):
        self.__gaussian_radius = strength
    
    def get_gaussian_radius(self):
        return self.__gaussian_radius
    
    def set_sharpness_boost_strength(self, strength: float):
        self.__sharpness_boost_strength = strength
    
    def get_sharpness_boost_strength(self):
        return self.__sharpness_boost_strength

    def set_contrast_boost_strength(self, strength: float):
        self.__contrast_boost_strength = strength
    
    def get_contrast_boost_strength(self):
        return self.__contrast_boost_strength

    def set_brightness_boost_strength(self, strength: float):
        self.__brightness_boost_strength = strength
    
    def get_brightness_boost_strength(self):
        return self.__brightness_boost_strength

    def set_percentage_train(self, percentage: int):
        self.__percentage_train = percentage/100
    
    def get_percentage_train(self):
        return self.__percentage_train

    def set_images_dir(self, dir: str):
        self.__images_dir = dir

    def get_images_dir(self):
        return self.__images_dir

    def get_supported_img_extensions(self):
        return self.__supported_img_extensions

    def get_images_train(self):
        return self.__images_train

    def get_answers_train(self):
        return self.__answers_train

    def get_images_test(self):
        return self.__images_test

    def get_answers_test(self):
        return self.__answers_test

    def get_predictions(self):
        return self.__predictions

    def set_distances_glcm(self, distances: list):
        self.__distances_glcm = distances
    
    def get_distances_glcm(self):
        return self.__distances_glcm

    def set_angles_glcm(self, angles: list):
        self.__angles_glcm = angles

    def get_angles_glcm(self):
        return self.__angles_glcm

    def set_texture_descriptors(self, descriptors: list):
        self.__texture_descriptors = descriptors

    def get_texture_descriptors(self):
        return self.__texture_descriptors

    def set_runtime_metric(self, metric_name: str, measured_time: float):
        self.__runtime_metrics[metric_name] = f"{measured_time:.6f} s"
    
    def get_runtime_metrics(self):
        return self.__runtime_metrics


    '''
    Aplica uma sequência de filtros na imagem para pré-processamento. O objetivo aqui é
    realçar características relevantes a fim de auxiliar o modelo na classificação
    @params: 
        image -> uma instancia de PIL.Image()
    '''
    def pre_process_img(self, image: Image):
        start = time.perf_counter()

        # Pré-processamento com suavização gaussiana e filtros para alterar sharpness,
        # contraste, cor e brilho
        image = image.filter(ImageFilter.GaussianBlur(radius=self.get_gaussian_radius()))

        enhancer_sharpness = ImageEnhance.Sharpness(image)
        image = enhancer_sharpness.enhance(self.get_sharpness_boost_strength())

        enhancer_contrast = ImageEnhance.Contrast(image)
        image = enhancer_contrast.enhance(self.get_contrast_boost_strength())
        
        enhancer_brightness = ImageEnhance.Brightness(image)
        image = enhancer_brightness.enhance(self.get_brightness_boost_strength())

        # Reamostragem dos tons de cinza
        image = image.quantize(self.get_n_colors())
        end = time.perf_counter()

        self.set_runtime_metric("Pré-processamento", end - start)

        return image


    '''
    Processa a imagem e retorna um array com os valores para os descritores de textura.
    @params: 
        filepath -> Um path válido para uma imagem a partir do pasta self.get_images_dir()
    '''
    def process_img(self, filepath: str):
        start = time.perf_counter()

        image =  Image.open(f"{filepath}")
        image = self.pre_process_img(image)

        # Gerando matriz de co-ocorrência
        glcm = graycomatrix(image, self.get_distances_glcm(), self.get_angles_glcm(), levels=self.get_n_colors())
        
        texture_descriptors = []
        # A entropia não pode ser pega com a função graycoprops, então ela precisa ser pega aqui
        texture_descriptors.append(shannon_entropy(glcm, base=2))

        # Extraindo valores para os descritores de textura
        for i in self.get_texture_descriptors():
            texture_descriptors.append(graycoprops(glcm, i))

        # Ajuntando todos em um array só
        descriptors = np.concatenate((texture_descriptors), axis=None)

        end = time.perf_counter()

        self.set_runtime_metric("Processamento de uma imagem", end - start)

        return descriptors


    '''
    Essa função separa as imagens das classes BIRADS de acordo com self.__percentage_train, com 
    balanceamento equilibrado entre as 4 classes.
    '''
    def split_train_test(self):
        '''
        Infelizmente não há como saber quantos elementos os arrays vão ocupar de antemão,
        e isso impossibilita usar arrays numpy para armazenar diretamente os dados resultados.
        Os conjuntos de teste e treino são armazenados aqui temporariamente e no fim da execução
        são convertidos para arrays numpy de volta.
        '''
        tmp_train_set = []
        tmp_test_set  = []

        self.__answers_test  = np.empty(0, dtype=np.int32)
        self.__answers_train = np.empty(0, dtype=np.int32)

        # Essas são as 4 classes BIRADS
        density_classes = ["1", "2", "3", "4"]

        # Início to cronômetro para separação
        start = time.perf_counter()

        for density_class in density_classes:
            path_images = [f for f in os.listdir(f"{self.get_images_dir()}/{density_class}") if f.endswith(tuple(self.get_supported_img_extensions()))]
            
            tmp_train, tmp_test = train_test_split(path_images, train_size=self.get_percentage_train(), shuffle=True)

            # Para cada imagem no conjunto de treino, extrair os descritores de textura abaixo e concatená-los
            # ao conjunto de treino
            for i in tmp_train:
                descriptors = self.process_img(f"{self.get_images_dir()}/{density_class}/{i}")

                tmp_train_set.append(descriptors)
                self.__answers_train = np.append(self.__answers_train, [density_class], axis=0)

            
            # Mesma coisa que antes, só que concatenar ao conjunto de teste. As operações são as mesmas.
            for i in tmp_test:
                descriptors = self.process_img(f"{self.get_images_dir()}/{density_class}/{i}")

                tmp_test_set.append(descriptors)
                self.__answers_test = np.append(self.__answers_test, [density_class], axis=0)
        
        # Fim da separação
        end = time.perf_counter()

        self.set_runtime_metric("Separação das imagens de treino e teste", end - start)

        # Transformando de volta para um array numpy
        self.__images_train = np.asarray(tmp_train_set)
        self.__images_test  = np.asarray(tmp_test_set)


    '''
    Treina o modelo com as imagens em self.get_images_train(). Deve ser chamada
    somente depois de self.split_train_test()
    '''
    def train_model(self):
        start = time.perf_counter()
        self.__model.fit(self.get_images_train(), self.get_answers_train())
        end   = time.perf_counter()

        self.set_runtime_metric("Etapa de treinamento do modelo", end-start)


    '''
    Faz a previsão das classes BIRADS das imagens de teste. Essa função deve ser chamada 
    somente depois de self.split_train_test() e de self.train_model()
    '''
    def predict_test_images(self):
        start = time.perf_counter()
        self.__predictions = self.__model.predict(self.get_images_test())
        end   = time.perf_counter()

        self.set_runtime_metric("Identificação das imagens de teste", end - start)


    '''
    Prevê a classe BIRADS de uma única imagem e retorna esse valor.
    @params:
        filepath -> Um path válido para uma imagem a partir do pasta self.get_images_dir()
    '''
    def predict_single_image(self, filepath: str):
        start = time.perf_counter()

        descriptors = self.process_img(filepath)
        birads_class = self.__model.predict([descriptors])
        
        end   = time.perf_counter()
        self.set_runtime_metric("Identificação de uma única imagem", end - start)

        return birads_class


    '''
    Retorna o histograma de uma única imagem
    @params:
        filepath -> Um path válido para uma imagem a partir do pasta self.get_images_dir()
    '''
    def get_single_image_histogram(self, filepath: str):
        start = time.perf_counter()

        image = Image.open(f"{filepath}")
        image = image.quantize(self.get_n_colors())

        plt.hist(np.asarray(image).ravel(), 256, [0,32], color=['gray'], histtype='stepfilled')
        plt.ylabel("Número de ocorrências")
        plt.xlabel("Tom de cinza")
        plt.savefig('./histograma.png')
        plt.clf()

        end = time.perf_counter()

        self.set_runtime_metric("Geração do Histograma", end - start)


    '''
    Faz a preview de uma única imagem
    @params:
        filepath -> Um path válido para uma imagem a partir do pasta self.get_images_dir()
    '''
    def preview_single_image(self, filepath: str):
        start = time.perf_counter()

        image = Image.open(f"{filepath}")
        image = self.pre_process_img(image)

        end = time.perf_counter()

        self.set_runtime_metric("Aplicação dos efeitos na imagem", end - start)
        return image


    '''
    Retorna os valores para as métricas de avaliação após o treinamento e os testes.
    Também gera uma imagem da matriz de confusão
    '''
    def get_prediction_metrics(self):
        start = time.perf_counter()

        accuracy = sklearn.metrics.accuracy_score(self.get_answers_test(), self.get_predictions())
        c_matrix = sklearn.metrics.confusion_matrix(self.get_answers_test(), self.get_predictions())
        especificidade = (100 - accuracy)/300

        plt.matshow(c_matrix, cmap=plt.cm.BuGn)

        # Mudando o colormap para facilitar a visualização. E também pra ficar mais legal :)
        plt.title("Matriz de Confusão", y=1, loc="center", pad=40, fontsize=16)
        plt.colorbar()

        for (i, j), z in np.ndenumerate(c_matrix):
            plt.text(j, i, '{:0.0f}'.format(z), ha='center', va='center', 
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
        
        # Cada tick representa uma das classes BIRADS. Tem um 0
        # pq o matplotlib precisa de um ponto para começar o desenho,
        # no caso o ponto (0,0) precisa existir
        ax = plt.gca()
        ax.set_xticklabels(("0","1","2","3","4"))
        ax.set_yticklabels(("0","1","2","3","4"))

        # Definindo os labels
        plt.ylabel("Classe correta", labelpad=15, fontsize=13)
        plt.xlabel("Classe estimada", labelpad=20, fontsize=13)

        plt.savefig(f"./matriz-confusao.png")
        plt.clf()

        end = time.perf_counter()

        self.set_runtime_metric("Geração das métricas de avaliação", end - start)

        return accuracy, especificidade