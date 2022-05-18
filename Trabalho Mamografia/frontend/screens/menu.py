import PySimpleGUI as sg
from frontend.screens.models import keys
from frontend.screens.models import texts
from frontend.screens.models import config

# Definindo opções padrões para os botões da tela
#, border_width=4, font=('Helvetica', 16)

# Configurações da tela
window_config = config.ConfigModel()

# Botões presentes à esquerda da tela
layout_esquerda = [
    [sg.Input(key=keys.ABRIR_PASTA_BIRADS_KEY, visible=False, enable_events=True)],
    [sg.Button(keys.ABRIR_PASTA_BIRADS_BUTTON_TEXT, key=None, target=(-1, 0), size=window_config.get_button_size(), border_width=4, button_type=sg.BUTTON_TYPE_BROWSE_FOLDER)],
    [sg.Button(keys.TREINAR_MODELO_BUTTON_TEXT, key=keys.TREINAR_MODELO_KEY, size=window_config.get_button_size(), border_width=4, disabled=True)],
    [sg.Input(key=keys.PREVER_IMAGEM_KEY, visible=False, enable_events=True)],
    [sg.Button(keys.PREVER_IMAGEM_BUTTON_TEXT, key=keys.PREVER_IMAGEM_KEY_BUTTON, target=(-1, 0), size=window_config.get_button_size(), border_width=4, disabled=True, button_type=sg.BUTTON_TYPE_BROWSE_FILE)],
    [sg.Button(keys.PREVER_IMAGENS_BUTTON_TEXT, key=keys.PREVER_IMAGENS_KEY, size=window_config.get_button_size(), border_width=4, disabled=True)],
    [sg.Button(keys.OBTER_METRICAS_MODELO_BUTTON_TEXT, key=keys.OBTER_METRICAS_MODELO_KEY, size=window_config.get_button_size(), border_width=4, disabled=True)],
    [sg.Button(keys.OPCOES_AVANCADAS_BUTTON_TEXT, key=keys.OPCOES_AVANCADAS_KEY, size=window_config.get_button_size(), border_width=4)],
    [sg.Button(keys.AJUDA_BUTTON_TEXT, key=keys.AJUDA_KEY, size=window_config.get_button_size(), border_width=4)],
    [sg.Button(keys.CREDITOS_BUTTON_TEXT, key=keys.CREDITOS_KEY, size=window_config.get_button_size(), border_width=4)]
]

# Tab 1 - Imagem selecionada
layout_tab1 = [
    [sg.Image(key=keys.TAB_USER_SELECTED_IMAGE_KEY)]
]

# Tab 2 - Preview da Imagem
layout_tab2 = [
    [sg.Image(key=keys.TAB_PREVIEW_IMAGE_KEY)]
]

# Tab 3 - Histograma
layout_tab3 = [
    [sg.Image(key=keys.TAB_IMAGE_HISTOGRAM_KEY)]
]

# Imagens exibidas à direita da tela
layout_direita = [
    [sg.TabGroup([[sg.Tab(keys.TAB_USER_SELECTED_IMAGE_TEXT, layout_tab1),
                   sg.Tab(keys.TAB_PREVIEW_IMAGE_TEXT, layout_tab2),
                   sg.Tab(keys.TAB_IMAGE_HISTOGRAM_TEXT, layout_tab3)]],
                   key=keys.TAB_GROUP_KEY)],
    [sg.Text(key=keys.TAB_RESULT_PREDICTION_KEY)]
]

# Layout da tela final
layout_tela = [
    [sg.vtop(sg.Column(layout_esquerda)), sg.VSeparator(), sg.vtop(sg.Column(layout_direita))],
]