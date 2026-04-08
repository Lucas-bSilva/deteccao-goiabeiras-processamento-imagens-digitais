import numpy as np


def apply_activation(x, activation):
    """
    Aplica a função de ativação ao resultado da correlação.

    """
    if activation == "relu":
        return np.maximum(x, 0)
    return x


def atrous_correlation_rgb(img, kernel, r=1, stride=1, activation="identity"):
    """
    Aplica correlação espacial dilatada (à trous) em uma imagem RGB.

    """

    # Valida o tipo da imagem de entrada para garantir compatibilidade
   
    if img.dtype != np.uint8:
        raise ValueError("Imagem deve estar em uint8.")

    # Converte o kernel para float32 para garantir precisão adequada
    
    kernel = np.array(kernel, dtype=np.float32)
    m, n = kernel.shape

    # Obtém as dimensões da imagem:
    # H = altura, W = largura, C = número de canais.
    H, W, C = img.shape

    # Calcula o tamanho efetivo do kernel após a dilatação.
    
    # pelo kernel cresce conforme o valor de r.
    eff_h = 1 + (m - 1) * r
    eff_w = 1 + (n - 1) * r

    # Garante que a imagem seja grande o suficiente para receber o kernel dilatado sem sair da região válida.
    
    if H < eff_h or W < eff_w:
        raise ValueError("Kernel maior que imagem.")

    # Calcula as dimensões da saída considerando:
    # - ausência de padding
    # - tamanho efetivo do kernel
    # - passo definido por stride
    Hout = (H - eff_h) // stride + 1
    Wout = (W - eff_w) // stride + 1

    # Inicializa a imagem de saída com zeros.
    
    output = np.zeros((Hout, Wout, 3), dtype=np.float32)

    # Processa cada canal de cor independentemente:
    # canal 0 = R, canal 1 = G, canal 2 = B.
    for c in range(3):
        for i in range(Hout):
            for j in range(Wout):
                acc = 0.0

                # Percorre todos os elementos do kernel.
                # A posição de amostragem na imagem considera:
                # - stride: deslocamento da janela
                # - r: espaçamento interno entre os pontos do kernel
                for ki in range(m):
                    for kj in range(n):
                        y = i * stride + ki * r
                        x = j * stride + kj * r

                        # Acumula a soma ponderada entre pixel e peso correspondente do kernel.
                        
                        acc += img[y, x, c] * kernel[ki, kj]

                # Armazena o valor calculado para o pixel de saída na posição correspondente.
                
                output[i, j, c] = acc

    # Aplica a ativação ao resultado final da correlação.
    output = apply_activation(output, activation)

    return output