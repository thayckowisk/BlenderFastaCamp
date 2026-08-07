# Detecção de Cadeiras com Dados Sintéticos Gerados no Blender

Projeto final de visão computacional com dados sintéticos, desenvolvido por thayckowisk. O projeto implementa um pipeline para geração de imagens sintéticas de uma cadeira no Blender, anotação automática no padrão YOLO, treinamento de um detector YOLOv8n e avaliação em um conjunto sintético de teste retido.

## Objetivo

Desenvolver um detector de uma classe (`chair`) utilizando exclusivamente dados sintéticos, e medir o quanto esse detector generaliza para fotos reais (domain gap). A proposta demonstra como uma cena 3D simples, combinada com automação em Python, reduz o trabalho manual de captura e anotação de imagens para treinar um modelo de detecção.

## Tecnologias utilizadas

- Blender e API Python `bpy`;
- Python 3;
- Ultralytics YOLOv8n;
- PyTorch;
- OpenCV, NumPy, Matplotlib;
- Jupyter Notebook;
- Formato de anotação YOLO (detecção e segmentação).

## Estrutura do projeto

```
card 12 - Projeto final/
├── projeto final.blend            # cena 3D configurada no Blender
├── 2_gerar_dataset_cadeira (3).py # geração e anotação automática do dataset de detecção
├── 6_montar_conjunto_teste.py     # monta o conjunto de teste retido e verifica vazamento
├── 5_mascaras_para_yolo_seg.py    # converte máscaras em rótulos YOLOv8-seg
├── treinamento_cadeira.ipynb      # notebook de treino, avaliação e exportação de resultados
├── chairs_dataset_sample/         # amostra do dataset de detecção (10 imagens por split)
│   ├── data.yaml
│   ├── images/{train,val,test}
│   └── labels/{train,val,test}
├── chairs_seg_sample/             # amostra do dataset de segmentação
│   ├── data.yaml
│   ├── images/{train,val}
│   ├── labels/{train,val}
│   └── _conferencia/              # polígonos desenhados sobre as imagens, para conferência visual
├── chairs_real/images/            # fotos reais usadas para avaliar o domain gap
├── weights/best.pt                # pesos do melhor modelo treinado
└── relatorio_assets/              # métricas, configuração do experimento e gráficos do treino
```

A geração automática do dataset completo produz centenas de imagens; aqui ficam apenas amostras de ~10 imagens por split, o suficiente para inspecionar o formato dos dados sem sobrecarregar o repositório.

## Dataset sintético

Imagens de 416 x 416 pixels, cada uma contendo uma cadeira. As caixas delimitadoras são calculadas a partir da projeção dos vértices reais da malha 3D na câmera (não da bounding box local do objeto), o que evita rótulos maiores que o objeto quando a cadeira gira.

Durante a geração são randomizados: rotação e cor/rugosidade da cadeira, posição e distância da câmera, cor do fundo e energia/temperatura das luzes — a técnica de *domain randomization* usada para reduzir o domain gap entre o dado sintético e fotos reais.

O conjunto de teste é gerado separadamente, com uma seed diferente do treino/validação, e a montagem verifica por hash SHA-256 que nenhuma imagem de teste é idêntica a uma imagem já vista no treino ou na validação — a prova de que o conjunto é de fato disjunto, e não apenas uma refatiação do mesmo lote.

## Treinamento

O treinamento e a avaliação rodam em `treinamento_cadeira.ipynb`, usando Ultralytics YOLOv8n. A configuração do experimento e os resultados finais estão documentados em `relatorio_assets/`:

| Parâmetro | Valor |
|---|---|
| Modelo base | yolov8n.pt |
| Épocas | 30 |
| Patience | 15 |
| Resolução | 416 |
| Batch | 16 |
| Seed | 42 |
| Device | CPU |
| Imagens treino | 400 |
| Imagens validação | 100 |
| Classes | chair |

## Resultados

Desempenho no conjunto sintético de validação:

| Métrica | Resultado |
|---|---|
| Precisão | 0,9995 |
| Revocação | 1,0000 |
| mAP@50 | 0,9950 |
| mAP@50-95 | 0,9950 |

`weights/best.pt` contém os pesos do melhor modelo obtido. `relatorio_assets/` também traz a matriz de confusão, curva Precisão-Revocação, exemplos de predição no conjunto sintético e em fotos reais.

## Instruções de uso

### 1. Ambiente

```bash
git clone https://github.com/thayckowisk/BlenderFastaCamp.git
cd BlenderFastaCamp
git checkout cadeiras-teste-e-docs
cd "card 12 - Projeto final"
pip install -r ../requirements.txt
```

### 2. Geração do dataset (opcional — só necessária para regerar do zero)

A geração roda dentro do Blender, na aba Scripting, com a cena `projeto final.blend` aberta e a cadeira já nomeada na cena. O script varre poses de câmera, cor, luz e fundo, descarta enquadramentos ruins e grava as imagens e os rótulos YOLO automaticamente. Depois de gerar treino/validação, o conjunto de teste é montado à parte — com uma seed diferente — e a montagem confirma por hash que não há sobreposição com treino/validação.

Para a etapa de segmentação, as máscaras renderizadas passam por uma conversão separada (fora do Blender, com OpenCV) que extrai o contorno de cada objeto e grava os polígonos no formato YOLOv8-seg.

### 3. Treino e avaliação

Abra `treinamento_cadeira.ipynb` e execute as células em ordem: validação do dataset, treino do YOLOv8n, cálculo de métricas em validação e no conjunto de teste retido, e comparação com fotos reais para observar o domain gap. A exportação final grava as figuras e tabelas em `relatorio_assets/`.

### 4. Rodar uma predição com o modelo treinado

```python
from ultralytics import YOLO

modelo = YOLO("card 12 - Projeto final/weights/best.pt")
modelo.predict(
    source="card 12 - Projeto final/chairs_dataset_sample/images/test",
    conf=0.25,
    imgsz=416,
    save=True,
)
```

## Limitações e domain gap

As métricas acima foram obtidas em imagens sintéticas produzidas pelo mesmo pipeline usado no treino, então não comprovam por si só desempenho equivalente em fotografias reais — por isso o notebook também avalia o modelo nas fotos de `chairs_real/images`. A cena 3D é simples (uma cadeira, um cenário controlado) e o dataset é pequeno para os padrões de detecção. Como continuidade, o próximo passo natural é variar mais os modelos 3D, materiais e ângulos de câmera, ampliar o conjunto de fotos reais e aplicar fine-tuning com dados reais.

## Autor

thayckowisk — Fastcamp de Dados Sintéticos para IA e Visão Computacional
