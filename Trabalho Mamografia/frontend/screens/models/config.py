# Configuração da tela

class ConfigModel:
    def __init__(self) -> None:
        self.button_size = (40, 1)
    
    # Pegar tamanho dos botões
    def get_button_size(self):
        return self.button_size