# -*- coding: utf-8 -*-
"""
SCRIPT 5 -- Converte as máscaras em rótulos YOLOv8-seg. Roda FORA do Blender.

    pip install opencv-python numpy
    python 5_mascaras_para_yolo_seg.py

Por que existe: o YOLOv8-seg não é treinado com imagens de máscara, e sim com
POLÍGONOS normalizados, uma linha por instância:

    classe x1 y1 x2 y2 ... xn yn

Este script extrai o contorno de cada máscara, simplifica (para não gerar
centenas de pontos por objeto) e grava nesse formato. OpenCV não está
disponível dentro do Blender, por isso a conversão é um passo separado.

Saída (layout padrão do Ultralytics):

    C:/tmp/chairs_seg_yolo/
        data.yaml
        images/train + images/val
        labels/train + labels/val
        _conferencia/      polígonos desenhados sobre as imagens
"""

import os
import shutil
import random
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit('Falta o OpenCV. Rode:  pip install opencv-python')

# =========================== CONFIG ===========================
ENTRADA = Path('C:/tmp/chairs_seg')        # saída do script 4
SAIDA = Path('C:/tmp/chairs_seg_yolo')     # dataset pronto para treinar

CLASSE_ID = 0
CLASSE_NOME = 'chair'

FRACAO_VAL = 0.2
SEED = 42

# simplificação do contorno: fração do perímetro usada como tolerância.
# 0.002 mantém a silhueta da cadeira; valores maiores achatam as pernas.
EPSILON_REL = 0.002
MIN_AREA_PX = 200        # descarta respingo de máscara
MIN_PONTOS = 6           # polígono com menos que isso não descreve o objeto
N_CONFERENCIA = 12
# ==============================================================

random.seed(SEED)


def _costurar_buraco(externo, buraco):
    """Emenda um vão interno no contorno externo por uma 'fenda'.

    O formato YOLO-seg é um polígono único por instância -- não tem como
    declarar buraco. O truque padrão é abrir uma fenda de largura zero
    ligando o contorno externo ao buraco, percorrer o buraco e voltar.
    Ao rasterizar, a fenda se fecha sobre si mesma e o vão fica vazio.
    """
    d = ((externo[:, None, :] - buraco[None, :, :]) ** 2).sum(axis=2)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    buraco = buraco[::-1]                      # orientação oposta à do externo
    j = len(buraco) - 1 - j
    anel = np.concatenate([buraco[j:], buraco[:j + 1]])
    return np.concatenate([externo[:i + 1], anel, externo[i:]])


def mascara_para_poligonos(caminho_mascara):
    """Extrai polígonos normalizados de uma máscara binária.

    Retorna (lista_de_poligonos, diagnostico).
    """
    m = cv2.imread(str(caminho_mascara), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return [], 'não consegui ler a máscara'

    H, W = m.shape
    pico = int(m.max())
    if pico == 0:
        return [], 'máscara vazia (objeto não apareceu no render)'

    # limiar relativo ao pico: robusto mesmo se o color management tiver
    # comprimido os valores (branco virando ~186 em vez de 255)
    _, bin_ = cv2.threshold(m, pico * 0.5, 255, cv2.THRESH_BINARY)

    # RETR_CCOMP separa contornos externos (hierarquia -1) dos vãos internos.
    # Usar RETR_EXTERNAL tapava os vãos entre as ripas do encosto e entre as
    # pernas, inflando a máscara em ~30% -- erro grande para segmentação.
    contornos, hier = cv2.findContours(bin_, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return [], 'nenhum contorno encontrado'
    hier = hier[0]

    def simplificar(c):
        eps = EPSILON_REL * cv2.arcLength(c, True)
        return cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float32)

    externos = [(k, c) for k, c in enumerate(contornos)
                if hier[k][3] == -1 and cv2.contourArea(c) >= MIN_AREA_PX]
    if not externos:
        return [], 'nenhum contorno acima da área mínima'

    # fica o maior componente: com uma cadeira por imagem, os menores são
    # partes soltas ou respingo
    k_pai, c_pai = max(externos, key=lambda kc: cv2.contourArea(kc[1]))
    poligono = simplificar(c_pai)

    n_buracos = 0
    for k, c in enumerate(contornos):
        if hier[k][3] != k_pai or cv2.contourArea(c) < MIN_AREA_PX:
            continue
        poligono = _costurar_buraco(poligono, simplificar(c))
        n_buracos += 1

    if len(poligono) < MIN_PONTOS:
        return [], 'polígono degenerado'

    poligono[:, 0] /= W
    poligono[:, 1] /= H
    return ([np.clip(poligono, 0.0, 1.0)],
            f'{len(externos)} externo(s), {n_buracos} vão(s), {len(poligono)} pontos')


def main():
    dir_img = ENTRADA / 'images'
    dir_msk = ENTRADA / 'masks'
    if not dir_img.exists() or not dir_msk.exists():
        raise SystemExit(f'Não encontrei {dir_img} e {dir_msk}. Rode o script 4 primeiro.')

    imagens = sorted(p for p in dir_img.iterdir()
                     if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
    print(f'{len(imagens)} imagem(ns) em {dir_img}')

    for sub in ('images/train', 'images/val', 'labels/train', 'labels/val', '_conferencia'):
        (SAIDA / sub).mkdir(parents=True, exist_ok=True)

    indices = list(range(len(imagens)))
    random.shuffle(indices)
    n_val = int(len(indices) * FRACAO_VAL)
    val = set(indices[:n_val])

    convertidos = {'train': 0, 'val': 0}
    problemas = []
    pontos_por_poly = []

    for i, img in enumerate(imagens):
        mascara = dir_msk / img.name
        if not mascara.exists():
            problemas.append(f'{img.name}: sem máscara correspondente')
            continue

        poligonos, diag = mascara_para_poligonos(mascara)
        if not poligonos:
            problemas.append(f'{img.name}: {diag}')
            continue

        split = 'val' if i in val else 'train'
        shutil.copy(img, SAIDA / 'images' / split / img.name)

        linhas = []
        for p in poligonos:
            coords = ' '.join(f'{v:.6f}' for v in p.flatten())
            linhas.append(f'{CLASSE_ID} {coords}')
            pontos_por_poly.append(len(p))

        (SAIDA / 'labels' / split / (img.stem + '.txt')).write_text(
            '\n'.join(linhas) + '\n')
        convertidos[split] += 1

    # data.yaml
    (SAIDA / 'data.yaml').write_text(
        f'path: {SAIDA.as_posix()}\n'
        'train: images/train\n'
        'val: images/val\n'
        'nc: 1\n'
        f"names: ['{CLASSE_NOME}']\n"
    )

    # conferência visual
    todos = sorted((SAIDA / 'images' / 'train').iterdir())
    amostra = random.sample(todos, min(N_CONFERENCIA, len(todos))) if todos else []
    for p in amostra:
        img = cv2.imread(str(p))
        H, W = img.shape[:2]
        lbl = SAIDA / 'labels' / 'train' / (p.stem + '.txt')
        overlay = img.copy()
        for linha in lbl.read_text().splitlines():
            vals = linha.split()
            pts = np.array(vals[1:], dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= W
            pts[:, 1] *= H
            pts = pts.astype(np.int32)
            cv2.fillPoly(overlay, [pts], (0, 0, 255))
            cv2.polylines(img, [pts], True, (0, 255, 255), 2)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        cv2.imwrite(str(SAIDA / '_conferencia' / p.name), img)

    print()
    print('=========================================================')
    print(f"convertidos: {convertidos['train']} treino / {convertidos['val']} validação")
    if pontos_por_poly:
        pp = sorted(pontos_por_poly)
        print(f'pontos por polígono: min {pp[0]} | mediana {pp[len(pp)//2]} | max {pp[-1]}')
        if pp[len(pp)//2] < 8:
            print('  [ALERTA] polígonos muito simples — reduza EPSILON_REL')
        if pp[len(pp)//2] > 120:
            print('  [ALERTA] polígonos muito densos — aumente EPSILON_REL')
    if problemas:
        print(f'problemas: {len(problemas)}')
        for x in problemas[:10]:
            print('  -', x)
    else:
        print('problemas: nenhum')
    print()
    print(f'conferência visual: {SAIDA / "_conferencia"}')
    print('Olhe essa pasta: o polígono precisa acompanhar a silhueta da cadeira,')
    print('inclusive os vãos entre as ripas. Se estiver "engordado", ajuste EPSILON_REL.')
    print()
    print('Para treinar:')
    print(f'  yolo segment train data={SAIDA.as_posix()}/data.yaml '
          'model=yolov8n-seg.pt epochs=60 imgsz=416')
    print('=========================================================')


if __name__ == '__main__':
    main()
