# PDI – Detecção e Contagem de Pés de Goiaba em Imagens Aéreas

Este projeto implementa um pipeline de **Processamento Digital de Imagens (PDI)** para detecção e **contagem estimada de pés de goiaba** a partir de imagens capturadas por drone.

A solução reaproveita o motor de correlação dilatada (*à trous*) desenvolvido anteriormente e integra novas etapas de segmentação, filtragem e análise estrutural.

---

##  Objetivo

Realizar a detecção automática de copas de goiabeiras e estimar a quantidade de plantas em imagens aéreas, utilizando técnicas clássicas de PDI como:

* suavização
* realce de características
* segmentação
* operações morfológicas
* análise de componentes conectados

---

##  Reaproveitamento do Projeto Anterior

* `atrous.py` → motor de correlação dilatada RGB
* `main.py` → execução baseada em configuração JSON
* `utils.py` → funções auxiliares (normalização, conversão, etc.)

---

##  Componentes Adicionados

* `count_trees.py` → pipeline completo de detecção e contagem
* `configs/tree_count_default.json` → parâmetros do algoritmo
* `configs/gaussian5.json` → filtro gaussiano
* `configs/sobel_h.json` / `sobel_v.json` → operadores de Sobel

---

##  Pipeline de Processamento

1. Leitura da imagem RGB
2. Redimensionamento (para otimização)
3. Suavização Gaussiana (opcional)
4. Cálculo do **Green Score** (realce da vegetação)
5. Detecção de bordas (Sobel)
6. Limiarização por percentil
7. Operações morfológicas (abertura/fechamento)
8. Rotulação de componentes conectados
9. Filtragem por área
10. Geração da contagem e visualização final

---

##  Estrutura do Projeto

```text
PDI_Goiabeiras/
├── atrous.py
├── main.py
├── utils.py
├── count_trees.py
├── configs/
│   ├── tree_count_default.json
│   ├── gaussian5.json
│   ├── sobel_h.json
│   └── sobel_v.json
├── drone_images/
│   ├── area1.png
│   ├── area2.png
│   ├── area3.png
│   └── area4.png
├── results/
│   └── trees/
└── README.md
```

---

##  Dependências

```bash
pip install numpy pillow
```

---

##  Execução do Projeto

###  Executar para cada imagem selecionada

```bash
python count_trees.py -i drone_images/area1.png -c configs/tree_count_default.json -o results/trees/area1
python count_trees.py -i drone_images/area2.png -c configs/tree_count_default.json -o results/trees/area2
python count_trees.py -i drone_images/area3.png -c configs/tree_count_default.json -o results/trees/area3
python count_trees.py -i drone_images/area4.png -c configs/tree_count_default.json -o results/trees/area4
```

---

##  Saídas Geradas

Para cada execução, será criada uma pasta contendo:

* `01_green_score.png` → mapa de vegetação
* `02_smoothed.png` → imagem suavizada
* `03_sobel_magnitude.png` → bordas detectadas
* `04_final_mask.png` → segmentação final
* `05_overlay_count.png` → **detecção com caixas (resultado final)**
* `06_count_report.txt` → contagem estimada

---

##  Interpretação dos Resultados

* O arquivo mais importante é:

```text
05_overlay_count.png
```

Ele apresenta:

* localização das árvores detectadas
* bounding boxes
* resultado visual final

---

##  Ajuste de Parâmetros

###  Muitas detecções (falso positivo)

* aumentar `green_threshold_percentile`
* aumentar `min_area`

###  Poucas detecções

* diminuir `green_threshold_percentile`
* diminuir `min_area`

###  Máscara fragmentada

* aumentar `binary_close_iterations`

###  Ruído excessivo

* aumentar `binary_open_iterations`

---

##  Estratégia de Uso

* Testar inicialmente com imagens reduzidas (`resize_factor`)
* Ajustar parâmetros no JSON
* Validar resultados visuais antes da contagem final

---


##  Observações

* O desempenho depende do tamanho da imagem
* Imagens de drone podem exigir redução (`resize_factor`)
* O método é baseado em técnicas clássas de PDI (sem uso de IA)

---

##  Conclusão

O projeto demonstra a aplicação prática de técnicas de Processamento Digital de Imagens para análise agrícola, permitindo a automação da contagem de plantas com base em características visuais e estruturais.

---
