import numpy as np


def histogram_stretch(channel):
    """
    Realiza expansão linear do histograma de um canal da imagem.

    """
    # Determina os limites de intensidade do canal
    min_val = np.min(channel)
    max_val = np.max(channel)

    # Caso todos os valores sejam iguais, evita divisão por zero retornando um canal totalmente preto.
   
    if max_val - min_val == 0:
        return np.zeros_like(channel, dtype=np.uint8)

    # Aplica a expansão linear das intensidades
    stretched = (channel - min_val) * (255.0 / (max_val - min_val))

    # Garante que os valores estejam dentro do intervalo válido e converte para o formato padrão de imagem (uint8).
 
    return np.clip(stretched, 0, 255).astype(np.uint8)



def to_uint8_clip(img):
    """
    Converte a saída da correlação comum para formato de imagem padrão.
    """

    return np.clip(img, 0, 255).astype(np.uint8)