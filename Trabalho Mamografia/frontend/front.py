from frontend import utils
import PySimpleGUI as sg

from frontend.utils import *

from backend import classifier
from frontend.screens import menu
from frontend.screens import advanced_options

from frontend.screens.models import images
from PIL import Image, ImageTk

# Class to define all functions to work the frontend
class GUI:    
    def __init__(self, current_working_dir) -> None:
        self.current_working_dir = current_working_dir
        self.loop = True
        self.modelo = classifier.ImageClassifier()
        self.img_model = images.ImagemModel()

    # Draw the GUI and hold it, until the user asks to exit
    def drawGUI(self):
        self.main_screen()

    # Drawing the drawTrainGUI
    def main_screen(self):
        # Layout da tela para o usuário
        layout = menu.layout_tela
        layout_opcoes_avancadas: advanced_options.AdvancedOptions
        
        # Criando a janela
        window = sg.Window('Reconhecimento de Tumores', layout, resizable=False, font=('Helvetica', 16))

        # Eventos ocorridos
        while self.loop:
            event, values = window.read()

            # Sair da aplicação
            if event == sg.WIN_CLOSED: 
                self.loop = False
                break
            
            # Selecionar os arquivos do disco
            elif event == menu.keys.ABRIR_PASTA_BIRADS_KEY:
                # Validação da entrada
                self.modelo.set_images_dir(values[menu.keys.ABRIR_PASTA_BIRADS_KEY])
                del values[menu.keys.ABRIR_PASTA_BIRADS_BUTTON_TEXT]

                # Liberar botão para treinar
                window[menu.keys.TREINAR_MODELO_KEY].update(disabled=False)
            
            # Treinar modelo
            elif event == menu.keys.TREINAR_MODELO_KEY:
                self.modelo.split_train_test()
                self.modelo.train_model()
                
                # Liberar botões que precisam do treino
                window[menu.keys.PREVER_IMAGEM_KEY_BUTTON].update(disabled=False)
                window[menu.keys.PREVER_IMAGENS_KEY].update(disabled=False)
            
            # Prever imagem única
            elif event == menu.keys.PREVER_IMAGEM_KEY:
                # Pegando imagem selecionada pelo usuário
                imagem = values[menu.keys.PREVER_IMAGEM_KEY]
                self.img_model.apply_preview_image(imagem, window, menu.keys.TAB_USER_SELECTED_IMAGE_KEY)
                resultado_predicao = self.modelo.predict_single_image(imagem)
                
                # Mostrando imagem com os filtros aplicados
                self.img_model.apply_preview_PIL_image(self.modelo.preview_single_image(imagem), window, menu.keys.TAB_PREVIEW_IMAGE_KEY)
                
                # Adicionando o histograma
                self.modelo.get_single_image_histogram(imagem)
                histgr = Image.open('./histograma.png')
                self.img_model.apply_preview_PIL_image(histgr, window, menu.keys.TAB_IMAGE_HISTOGRAM_KEY)
                
                window[menu.keys.TAB_RESULT_PREDICTION_KEY].update(value=f"Classe BI-RADS identificada: {resultado_predicao[0]}")
                
            
            # Prever múltiplas imagens
            elif event == menu.keys.PREVER_IMAGENS_KEY:
                self.modelo.predict_test_images()
                window[menu.keys.OBTER_METRICAS_MODELO_KEY].update(disabled=False)
            
            
            # Obter métricas de classificação do modelo
            elif event == menu.keys.OBTER_METRICAS_MODELO_KEY:
                runtime_metricas = self.modelo.get_runtime_metrics()
                            
                texto_final = 'Métricas do tempo de execução:\n\n'
                # Formatar o print                
                for key in runtime_metricas:
                    texto_final += f"{key}: {runtime_metricas[key]}\n"
                
                texto_final += '\nMétricas do modelo:\n\n'
                accuracy, especificidade = self.modelo.get_prediction_metrics()
                texto_final += f"Acurácia: {accuracy}\nEspecificidade: {especificidade}"
                
                # Imagem da matriz de confusão
                sg.popup_ok(texto_final, image='./matriz-confusao.png', title='Métricas de Avaliação do Modelo', font=('Helvetica', 16))
            
            # Abrir opções avançadas
            elif event == menu.keys.OPCOES_AVANCADAS_KEY:
                acessar_opcoes = sg.popup_ok_cancel('Opções avançadas podem afetar o desempenho do modelo!\nUse com cautela!', title='Aviso!', font=('Helvetica', 16))
                
                if acessar_opcoes == 'OK':
                    layout_opcoes_avancadas = advanced_options.AdvancedOptions()
                    layout_opcoes_avancadas.construir_tela_opcoes_avancadas(self.modelo)
                    
                    # Travar botões que precisam do treino
                    window[menu.keys.PREVER_IMAGEM_KEY_BUTTON].update(disabled=True)
                    window[menu.keys.PREVER_IMAGENS_KEY].update(disabled=True)
            
            # Opções avançadas
            elif event == menu.keys.AJUDA_KEY:
                texto_de_ajuda = utils.read_file('./assets/texts/help.txt')
                sg.popup_ok(texto_de_ajuda, title='Ajuda', font=('Helvetica', 16))
            
            # Créditos da aplicação
            elif event == menu.keys.CREDITOS_KEY:
                texto_de_creditos = utils.read_file('./assets/texts/creditos.txt')
                sg.popup_ok(texto_de_creditos, title='Créditos', font=('Helvetica', 16))
            
        # Finalizar aplicação e fechar a janela
        window.close()
        