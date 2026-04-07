import numpy as np


def histogram_stretch(channel):
    """
    Realiza expansão linear do histograma de um canal da imagem.

    Finalidade:
    - Normalizar os valores de intensidade do canal para o intervalo [0, 255].
    - Aumentar o contraste da imagem quando os valores estão concentrados
      em uma faixa pequena de intensidades.

    Funcionamento:
    - Calcula o menor e o maior valor presentes no canal.
    - Aplica uma transformação linear que redistribui os valores
      proporcionalmente dentro do intervalo [0,255].

    Fórmula aplicada:
        novo_valor = (valor - min) * (255 / (max - min))

    Parâmetros:
        channel (np.ndarray): canal da imagem em float ou inteiro.

    Retorno:
        np.ndarray: canal normalizado no formato uint8.
    """

    # Determina os limites de intensidade do canal
    min_val = np.min(channel)
    max_val = np.max(channel)

    # Caso todos os valores sejam iguais, evita divisão por zero
    # retornando um canal totalmente preto.
    if max_val - min_val == 0:
        return np.zeros_like(channel, dtype=np.uint8)

    # Aplica a expansão linear das intensidades
    stretched = (channel - min_val) * (255.0 / (max_val - min_val))

    # Garante que os valores estejam dentro do intervalo válido
    # e converte para o formato padrão de imagem (uint8).
    return np.clip(stretched, 0, 255).astype(np.uint8)



def to_uint8_clip(img):
    """
    Converte a saída da correlação comum para formato de imagem padrão.

    Finalidade:
    - Garantir que os valores da imagem estejam no intervalo válido [0,255].
    - Converter o tipo de dados para uint8, compatível com
      bibliotecas de visualização e salvamento de imagens.

    Funcionamento:
    - Aplica clipping para limitar valores fora da faixa permitida.
    - Converte o resultado para inteiro de 8 bits.

    Parâmetros:
        img (np.ndarray): imagem resultante da correlação.

    Retorno:
        np.ndarray: imagem final no formato uint8.
    """

    return np.clip(img, 0, 255).astype(np.uint8)