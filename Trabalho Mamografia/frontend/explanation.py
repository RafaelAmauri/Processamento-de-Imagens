from frontend import utils
import PySimpleGUI as sg

def draw_credits_window():
    credits_text = utils.read_file("./frontend/assets/texts/creditos.txt")

    layout_creditos = [
        [sg.Text(credits_text)],
        [sg.Button("Retornar", border_width=4)]
    ]

    creditos_window = sg.Window('Créditos', layout_creditos)
    event, _ = creditos_window.read()

    if event == ('Retornar', sg.WIN_CLOSED):
        creditos_window.close()


def draw_help_window():

    help_text = utils.read_file("./frontend/assets/texts/help.txt")

    layout_help = [
        [sg.Text(help_text)],
        [sg.Button("Retornar")]
    ]

    help_window = sg.Window('Informações sobre o trabalho', layout_help)
    event, _ = help_window.read()

    if event == ('Retornar', sg.WIN_CLOSED):
        help_window.close()