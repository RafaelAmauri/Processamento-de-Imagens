from classifier import ImageClassifier
import os

classifier = ImageClassifier()

'''
ESSENCIAIS:
    classifier.set_images_dir(str)       -> Recebe a localizacao da pasta Imagens para saber onde elas estão
    classifier.split_train_test()        -> Realiza automaticamente a separação das imagens de treino e teste
    classifier.train_model()             -> Treina o modelo de acordo com as imagens de treino
    classifier.set_percentage_train(int) -> Define a porcentagem das imagens que serão usadas para treino


PRÉ-PROCESSAMENTO:
    classifier.set_gaussian_radius(int)             -> Define o raio da matriz gaussiana para suavização
    classifier.set_sharpness_boost_strength(float)  -> Define a força do filtro de sharpen
    classifier.set_contrast_boost_strength(float)   -> Define a força do filtro de contraste
    classifier.set_brightness_boost_strength(float) -> Define a força do filtro de brilho
    classifier.set_color_boost_strength(float)      -> Define a força do filtro de cor
    classifier.set_n_colors(int)                    -> Define o número de cores usado para a quantização


EXTRAS: 
    classifier.predict_test_images()     -> Prediz a classe BIRADS das imagens de teste
    classifier.predict_single_image(str) -> Prediz a classe BIRADS de uma única imagem
    classifier.get_runtime_metrics()     -> Retorna os tempos de execução medidos para cada função
    classifier.set_model_kernel(str)     -> Define a kernel utilizada pelo modelo
 '''
 
classifier.set_images_dir("../Imagens")
classifier.set_gaussian_radius(0)
classifier.set_sharpness_boost_strength(1.4)
classifier.set_contrast_boost_strength(2.2)
classifier.set_brightness_boost_strength(1.1)

classifier.split_train_test()

classifier.train_model()
classifier.predict_single_image(f"{os.getcwd()}/../Imagens/1/p_d_left_cc(12).png")
#classifier.preview_singe_image("1/p_d_left_cc(12).png")
classifier.predict_test_images()
# print(classifier.get_prediction_metrics())

# print(classifier.get_runtime_metrics())

classifier.get_single_image_histogram(f"{os.getcwd()}/../Imagens/1/p_d_left_cc(12).png")

#classifier.get_single_image_histogram("1/p_d_left_cc(12).png")
