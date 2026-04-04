import argparse
import json
import os

import numpy as np
from PIL import Image
import tkinter as tk
from PIL import ImageTk

from atrous import atrous_correlation_rgb
from utils import sobel_postprocess, to_uint8_clip


def load_config(path):
    """
    Carrega o arquivo JSON de configuração do filtro.

    Finalidade:
    - Ler os parâmetros externos do processamento, como kernel,
      taxa de dilatação (r), stride, ativação e indicadores
      específicos do filtro.
    - Separar a configuração da lógica do código, tornando o
      programa mais flexível e reutilizável.

    Parâmetros:
        path (str): caminho do arquivo JSON.

    Retorno:
        dict: dicionário com os parâmetros do filtro.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def show_images(original, result, title="Resultado"):
    """
    Exibe a imagem original e a imagem processada lado a lado.

    Finalidade:
    - Facilitar a comparação visual entre entrada e saída.
    - Apoiar análise qualitativa dos efeitos do filtro aplicado.

    Funcionamento:
    - Converte os arrays NumPy para imagens PIL.
    - Cria uma imagem combinada com original e resultado.
    - Exibe a composição em uma janela Tkinter.

    Parâmetros:
        original (np.ndarray): imagem original em formato RGB.
        result (np.ndarray): imagem resultante do processamento.
        title (str): título da janela de visualização.
    """
    root = tk.Tk()
    root.title(title)

    original_img = Image.fromarray(original)
    result_img = Image.fromarray(result)

    # Define o tamanho da imagem composta que irá acomodar
    # as duas imagens lado a lado.
    width = original_img.width + result_img.width
    height = max(original_img.height, result_img.height)

    combined = Image.new("RGB", (width, height))
    combined.paste(original_img, (0, 0))
    combined.paste(result_img, (original_img.width, 0))

    tk_img = ImageTk.PhotoImage(combined)
    label = tk.Label(root, image=tk_img)
    label.pack()

    # Mantém a referência da imagem vinculada ao widget,
    # evitando problemas de coleta de lixo na interface.
    label.image = tk_img

    root.mainloop()


def main():
    """
    Controla o fluxo principal de execução do programa.

    Etapas executadas:
    1. Lê os argumentos da linha de comando.
    2. Carrega a imagem de entrada em RGB.
    3. Lê a configuração do filtro a partir do JSON.
    4. Aplica a correlação atrous na imagem.
    5. Realiza o pós-processamento necessário:
       - Sobel: valor absoluto + normalização
       - Demais filtros: clipping para faixa válida
    6. Cria o diretório de saída, se necessário.
    7. Salva a imagem resultante.
    8. Exibe a comparação visual, caso solicitado.

    Essa função não implementa a matemática do filtro,
    mas coordena todas as etapas do processamento.
    """
    parser = argparse.ArgumentParser(
        description="Aplica correlação espacial/dilatada em uma imagem RGB com configuração via JSON."
    )
    parser.add_argument("-i", "--input", required=True, help="Caminho da imagem de entrada.")
    parser.add_argument("-c", "--config", required=True, help="Caminho do arquivo JSON do filtro.")
    parser.add_argument("-o", "--output", required=True, help="Caminho da imagem de saída.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Exibe a imagem original e o resultado lado a lado."
    )

    args = parser.parse_args()

    # Carrega a imagem de entrada e garante o formato RGB,
    # compatível com o processamento canal a canal.
    img = Image.open(args.input).convert("RGB")
    img_np = np.array(img, dtype=np.uint8)

    # Lê os parâmetros do filtro definidos no arquivo JSON.
    config = load_config(args.config)

    # Aplica a correlação atrous utilizando os parâmetros
    # informados na configuração.
    result = atrous_correlation_rgb(
        img_np,
        kernel=config["kernel"],
        r=config["r"],
        stride=config["stride"],
        activation=config["activation"]
    )

    # Filtros Sobel geram gradientes que exigem tratamento
    # adicional para visualização adequada.
    if config.get("is_sobel", False):
        result = sobel_postprocess(result)
    else:
        # Para os demais filtros, basta limitar os valores
        # ao intervalo válido de imagem [0, 255].
        result = to_uint8_clip(result)

    # Garante que o diretório de saída exista antes de salvar
    # o arquivo resultante.
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    Image.fromarray(result).save(args.output)

    # Exibe a comparação visual apenas se solicitado pelo usuário.
    if args.show:
        show_images(img_np, result, config["name"])


if __name__ == "__main__":
    main()