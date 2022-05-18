# Arquivo para carregar as imagens do modelo
from PIL import Image, ImageTk
import PySimpleGUI as sg

class ImagemModel:
    def __init__(self) -> None:
        self.image_size = (384, 384) # 3 * tamanho da imagem no disco
    
    # Desenhar imagem na tela
    def apply_preview_image(self, image_path: str, window: sg.Window, window_key):
        # Aumentando imagem para encaixar no canvas
        image = Image.open(image_path)
        preview_img = ImageTk.PhotoImage(image.resize(self.image_size))
        window[window_key].update(data=preview_img)
        
    # Desenhar imagem na tela
    def apply_preview_PIL_image(self, image: Image, window: sg.Window, window_key):
        # Aumentando imagem para encaixar no canvas
        preview_img = ImageTk.PhotoImage(image.resize(self.image_size))
        window[window_key].update(data=preview_img)