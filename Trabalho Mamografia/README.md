# Reconhecimento de padrões por textura em imagens mamógraficas

## Trabalho feito em grupo para a disciplina Processamento de Imagens do curso de Ciência da Computação da PUC Minas com o professor Alexei Manso Correa Machado

Desenvolvedores:

[Lucas Santiago](https://github.com/LucasSnatiago "Lucas Santiago") - Frontend

Rafael Amauri (eu :D) - Backend

[Thiago Henriques](https://github.com/ThiagoHN "Thiago Henriques") - Documentação


### ⚠️ Aviso
O programa foi feito para ser utilizado no Linux. Embora alguns testes mostraram que ele roda sem problemas no Windows,
o grupo não teve interesse nem incentivo para dar suporte à versão Windows. É fortemente recomendado utilizar no Linux!

### 🚀 Como utilizar ?
Primeiramente, é necessário que você tenha acesso ao conjunto de imagens disponibilizadas na disciplina. Infelizmente não podemos redistribuir elas aqui, então você precisa ter elas de antemão.
Com a posse das imagens em mão, para instalar as dependências do programa basta rodar:

```
pip3 install -r requirements.txt
```

Para executar o programa, rode:

```
python3 main.py
```

### Dica de utilização
Caso não queira utilizar a interface para rodar o programa, dê uma olhada no arquivo backend/classifier_test.py para ter uma ideia de como as coisas funcionam. Tenha em mente que esse arquivo só executa algumas das várias funções que foram implementadas para o classificador. Se tiver interesse, veja o código-fonte dele em backend/classifier.py
