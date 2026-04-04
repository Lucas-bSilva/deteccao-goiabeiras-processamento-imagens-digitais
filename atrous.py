import numpy as np


def apply_activation(x, activation):
    """
    Aplica a função de ativação ao resultado da correlação.

    Finalidade:
    - Ajustar o mapa de saída conforme a ativação especificada
      na configuração do filtro.
    - Permitir suporte a diferentes comportamentos pós-correlação.

    Regras implementadas:
    - "relu": substitui valores negativos por zero.
    - qualquer outro valor: mantém a saída inalterada
      (comportamento equivalente à função identidade).

    Parâmetros:
        x (np.ndarray): matriz resultante da correlação.
        activation (str): nome da ativação definida no JSON.

    Retorno:
        np.ndarray: resultado após aplicação da ativação.
    """
    if activation == "relu":
        return np.maximum(x, 0)
    return x


def atrous_correlation_rgb(img, kernel, r=1, stride=1, activation="identity"):
    """
    Aplica correlação espacial dilatada (à trous) em uma imagem RGB.

    Características da implementação:
    - Processa os três canais da imagem separadamente (R, G e B).
    - Utiliza correlação manual, sem bibliotecas prontas de filtragem.
    - Não utiliza padding; portanto, a saída é calculada apenas
      na região válida da imagem.
    - Retorna o resultado em float32, preservando a precisão
      numérica antes do pós-processamento final.

    Parâmetros:
        img (np.ndarray): imagem de entrada no formato RGB e tipo uint8.
        kernel (list | np.ndarray): máscara de pesos utilizada na correlação.
        r (int): taxa de dilatação do kernel.
        stride (int): passo de deslocamento da janela de correlação.
        activation (str): função de ativação aplicada ao final.

    Retorno:
        np.ndarray: imagem resultante da correlação, em float32.
    """

    # Valida o tipo da imagem de entrada para garantir compatibilidade
    # com o processamento esperado no projeto.
    if img.dtype != np.uint8:
        raise ValueError("Imagem deve estar em uint8.")

    # Converte o kernel para float32 para garantir precisão adequada
    # durante as multiplicações e somas da correlação.
    kernel = np.array(kernel, dtype=np.float32)
    m, n = kernel.shape

    # Obtém as dimensões da imagem:
    # H = altura, W = largura, C = número de canais.
    H, W, C = img.shape

    # Calcula o tamanho efetivo do kernel após a dilatação.
    # Mesmo sem aumentar a quantidade de pesos, a área coberta
    # pelo kernel cresce conforme o valor de r.
    eff_h = 1 + (m - 1) * r
    eff_w = 1 + (n - 1) * r

    # Garante que a imagem seja grande o suficiente para receber
    # o kernel dilatado sem sair da região válida.
    if H < eff_h or W < eff_w:
        raise ValueError("Kernel maior que imagem.")

    # Calcula as dimensões da saída considerando:
    # - ausência de padding
    # - tamanho efetivo do kernel
    # - passo definido por stride
    Hout = (H - eff_h) // stride + 1
    Wout = (W - eff_w) // stride + 1

    # Inicializa a imagem de saída com zeros.
    # O tipo float32 preserva valores intermediários antes da conversão final.
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

                        # Acumula a soma ponderada entre pixel e peso
                        # correspondente do kernel.
                        acc += img[y, x, c] * kernel[ki, kj]

                # Armazena o valor calculado para o pixel de saída
                # na posição correspondente.
                output[i, j, c] = acc

    # Aplica a ativação ao resultado final da correlação.
    output = apply_activation(output, activation)

    return output