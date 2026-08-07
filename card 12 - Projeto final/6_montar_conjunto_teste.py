# -*- coding: utf-8 -*-
"""
SCRIPT 6 -- Monta o conjunto de TESTE no dataset.

*** RODA NO TERMINAL / VS CODE. NUNCA na aba Scripting do Blender. ***
Este script nao usa bpy. Se colado na aba Scripting por engano, um erro
aqui pode inclusive fechar o Blender inteiro (o Python embutido no
Blender nem sempre segura corretamente uma excecao de saida do script).

    python 6_montar_conjunto_teste.py

Antes de rodar, gere as imagens de teste no Blender:
    abra o 2_gerar_dataset_cadeira.py e altere apenas a config:
        PASTA_SAIDA = 'C:/tmp/chairs_test'
        N_TREINO    = 100
        N_VAL       = 0
        SEED        = 777
    rode com a mesma cena montada (script 1).

O que este script faz:
  1. Copia as imagens e rótulos gerados para images/test e labels/test
     do dataset principal, renomeando com prefixo para não colidir
  2. Verifica por hash SHA-256 que NENHUMA imagem de teste é idêntica a
     alguma de treino ou validação -- é a prova de que o conjunto é
     disjunto, e não apenas uma refatiação do mesmo lote
  3. Reescreve o data.yaml incluindo a entrada 'test'

Por que não refatiar as 500 imagens existentes: o modelo já foi treinado
nas 400 de treino. Mover parte delas para teste vazaria dado visto no
treino para a avaliação final, inflando a métrica -- justamente o que um
conjunto de teste deveria evitar. Renders novos com outra semente não têm
esse problema e dispensam retreinar.
"""

import hashlib
import shutil
from pathlib import Path

# =========================== CONFIG ===========================
DATASET = Path('C:/tmp/chairs_dataset')     # dataset principal (train/val)
ORIGEM = Path('C:/tmp/chairs_test')         # saída da geração com SEED=777
SUB_ORIGEM = 'train'                        # subpasta gerada pelo script 2
PREFIXO = 'test_'                           # evita colisão de nomes

CLASSE_NOME = 'chair'
# ==============================================================


def sha(caminho, bloco=1 << 20):
    h = hashlib.sha256()
    with open(caminho, 'rb') as f:
        for parte in iter(lambda: f.read(bloco), b''):
            h.update(parte)
    return h.hexdigest()


def imagens(pasta):
    if not pasta.exists():
        return []
    return sorted(p for p in pasta.iterdir()
                  if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))


org_img = ORIGEM / 'images' / SUB_ORIGEM
org_lbl = ORIGEM / 'labels' / SUB_ORIGEM
if not org_img.exists():
    raise RuntimeError(
        'Nao encontrei %s.\n'
        'Rode primeiro o 2_gerar_dataset_cadeira.py com PASTA_SAIDA=%s e SEED diferente.\n'
        'IMPORTANTE: este script (6) roda no terminal/VS Code, NAO dentro do Blender -- '
        'ele nao usa bpy, e um erro aqui dentro do Blender pode fechar o programa.'
        % (org_img, ORIGEM))

dst_img = DATASET / 'images' / 'test'
dst_lbl = DATASET / 'labels' / 'test'
dst_img.mkdir(parents=True, exist_ok=True)
dst_lbl.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------
# 1. Hashes do que ja existe (treino + validacao)
# ---------------------------------------------------------------
existentes = {}
for split in ('train', 'val'):
    for p in imagens(DATASET / 'images' / split):
        existentes[sha(p)] = '%s/%s' % (split, p.name)
print('imagens ja no dataset (treino + val): %d' % len(existentes))

# ---------------------------------------------------------------
# 2. Copia com verificacao de sobreposicao
# ---------------------------------------------------------------
novas = imagens(org_img)
print('imagens de teste geradas: %d' % len(novas))

copiadas = 0
colisoes = []
sem_label = []
vistos = set()

for p in novas:
    h = sha(p)
    if h in existentes:
        colisoes.append((p.name, existentes[h]))
        continue
    if h in vistos:
        continue                      # duplicata dentro do proprio lote de teste
    vistos.add(h)

    lbl = org_lbl / (p.stem + '.txt')
    if not lbl.exists():
        sem_label.append(p.name)
        continue

    nome = PREFIXO + p.name
    shutil.copy(p, dst_img / nome)
    shutil.copy(lbl, dst_lbl / (PREFIXO + p.stem + '.txt'))
    copiadas += 1

# ---------------------------------------------------------------
# 3. data.yaml com a entrada de teste
# ---------------------------------------------------------------
(DATASET / 'data.yaml').write_text(
    'path: %s\n'
    'train: images/train\n'
    'val: images/val\n'
    'test: images/test\n'
    'nc: 1\n'
    "names: ['%s']\n" % (DATASET.as_posix(), CLASSE_NOME)
)

# ---------------------------------------------------------------
# Relatorio
# ---------------------------------------------------------------
print('')
print('=========================================================')
print('copiadas para o conjunto de teste: %d' % copiadas)
print('')
if colisoes:
    print('[ATENCAO] %d imagem(ns) de teste sao IDENTICAS a imagens de treino/val:'
          % len(colisoes))
    for a, b in colisoes[:10]:
        print('   %s == %s' % (a, b))
    print('   Isso significa vazamento. Gere o teste com uma SEED diferente.')
else:
    print('VERIFICACAO DE VAZAMENTO: nenhuma imagem de teste coincide com')
    print('treino ou validacao. Os conjuntos sao disjuntos.')
if sem_label:
    print('[aviso] %d imagem(ns) sem rotulo, ignoradas' % len(sem_label))

print('')
print('contagem final:')
for split in ('train', 'val', 'test'):
    print('  %-6s %4d imagens' % (split, len(imagens(DATASET / 'images' / split))))
print('')
print('data.yaml atualizado com a entrada test.')
print('Agora avalie no notebook (nao precisa retreinar):')
print("    metrics_teste = model_best.val(data=str(yaml_path), split='test',")
print("                                   imgsz=IMGSZ, device=DEVICE, plots=True)")
print('=========================================================')
