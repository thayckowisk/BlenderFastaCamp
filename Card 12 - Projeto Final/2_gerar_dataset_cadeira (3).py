# -*- coding: utf-8 -*-
"""
SCRIPT 2/2 -- Gera o dataset sintetico da chair_01 (deteccao, formato YOLO).

Rode DEPOIS do 1_montar_cena_cadeira.py, na mesma sessao do Blender.
Compativel com Blender 2.83.

Saida (layout padrao do Ultralytics YOLO):

    C:/tmp/chairs_dataset/
        data.yaml
        images/train/000000.png ...
        images/val/000000.png ...
        labels/train/000000.txt ...
        labels/val/000000.txt ...

Cada .txt tem uma linha:  0 x_centro y_centro largura altura   (normalizado 0-1)

DIFERENCAS EM RELACAO A VERSAO ANTERIOR (e por que importam):

1. Bounding box AJUSTADA (tight): projeta os 164 vertices reais da malha,
   nao os 8 cantos de obj.bound_box. A bound_box e' alinhada aos eixos
   LOCAIS -- quando a cadeira gira, ela superestima a caixa e o rotulo fica
   maior que o objeto, o que derruba o IoU no treino.

2. Camera por coordenadas esfericas (azimute/elevacao/raio) com Track To.
   Garante enquadramento e permite randomizar a DISTANCIA, coisa que o rig
   de curvas aninhadas do projeto da caneca nao permitia.

3. Jitter no alvo: o ponto que a camera mira sai um pouco do centro em cada
   render. Sem isso a cadeira cai sempre no meio do quadro e o modelo
   aprende vies de centro, indo mal em imagens reais.

4. Domain randomization: cor/rugosidade da cadeira, cor do backdrop,
   energia e temperatura das luzes, rotacao do rig de luz. E' o que reduz o
   domain gap -- ponto que o desafio pede para discutir no relatorio.

5. Controle de qualidade da amostra: descarta o render se a cadeira estiver
   muito truncada pela borda ou pequena demais no quadro, e recalcula a pose
   em vez de gravar um rotulo ruim.
"""

import bpy
import math
import random
import time
import os
from mathutils import Vector, Euler
from bpy_extras.object_utils import world_to_camera_view

# =========================== CONFIG ===========================
PASTA_SAIDA = 'C:/tmp/chairs_dataset'

N_TREINO = 400
N_VAL = 100

RESOLUCAO = 416          # imagem quadrada RESOLUCAO x RESOLUCAO
AMOSTRAS_CYCLES = 48     # samples do Cycles (32-64 e' suficiente nessa resolucao)

SEED = 42                # fixa a aleatoriedade -> dataset reproduzivel

# camera (metros / graus) -- calculado para a cadeira de 0.884 m com lente 50mm:
# 1.8 m -> ocupa ~68% do quadro | 3.0 m -> ~41%
DIST_MIN, DIST_MAX = 1.8, 3.0
ELEV_MIN, ELEV_MAX = 5.0, 65.0       # 5 graus = quase no chao, 65 = vista de cima
LENTE_MIN, LENTE_MAX = 40.0, 60.0

JITTER_ALVO = 0.18       # fracao da altura da cadeira que o alvo pode deslocar

# qualidade minima da amostra
MIN_AREA_BBOX = 0.010    # bbox deve ocupar >= 1% da imagem
MIN_FRACAO_VISIVEL = 0.75  # >= 75% da caixa dentro do quadro (aceita corte leve)

# Contraste minimo de luminancia entre a cadeira e o backdrop.
# Fundo e cadeira eram sorteados de forma INDEPENDENTE: quando os dois caiam
# claros (fundo 0.75 + bege 0.80), a cadeira sumia no fundo. Agora a cor da
# cadeira e' escolhida DEPOIS do fundo, so entre as que contrastam com ele.
# A cor do fundo continua sorteada livremente -- nao foi mexida.
MIN_CONTRASTE_LUM = 0.22

CLASSE_ID = 0
CLASSE_NOME = 'chair'
# ==============================================================

random.seed(SEED)
scn = bpy.context.scene

# --------------------------------------------------------------------------
# Localiza o que o script 1 montou
# --------------------------------------------------------------------------
CHAIR_NAME = scn.get('chair_name', 'chair_01')
chair = bpy.data.objects.get(CHAIR_NAME)
cam = scn.camera
target = bpy.data.objects.get('ObjectTarget')
rig = bpy.data.objects.get('LightRig')
backdrop = bpy.data.objects.get('Backdrop')

faltando = [n for n, o in (('cadeira %s' % CHAIR_NAME, chair), ('Camera', cam),
                           ('ObjectTarget', target)) if o is None]
if faltando:
    raise RuntimeError('Faltando na cena: %s. Rode o 1_montar_cena_cadeira.py primeiro.'
                       % ', '.join(faltando))

ALTURA = scn.get('chair_altura', 0.884)
CENTRO = Vector(scn.get('chair_centro', (0.0, 0.0, ALTURA / 2)))
# rotacao de repouso da cadeira (correcao de eixo do FBX). O script 1 aplica
# a transform e isso vira (0,0,0); o fallback usa a rotacao atual do objeto.
ROT_BASE = tuple(scn.get('chair_rot_base', tuple(chair.rotation_euler)))


def coletar_malhas(obj):
    """obj + descendentes MESH (feito na mao: children_recursive so existe no 3.1+)."""
    encontrados, pilha = [], [obj]
    while pilha:
        a = pilha.pop()
        if a.type == 'MESH':
            encontrados.append(a)
        pilha.extend(a.children)
    return encontrados


MALHAS = coletar_malhas(chair)
MATERIAIS = []
for m in MALHAS:
    for slot in m.material_slots:
        if slot.material is not None and slot.material.use_nodes:
            b = slot.material.node_tree.nodes.get('Principled BSDF')
            if b is not None:
                MATERIAIS.append(b)

LUZES = [o for o in bpy.data.objects
         if o.type == 'LIGHT' and o.name in ('KeyLight', 'FillLight', 'RimLight')]
ENERGIA_BASE = {o.name: o.data.energy for o in LUZES}

BSDF_BACKDROP = None
if backdrop is not None and backdrop.data.materials:
    mb = backdrop.data.materials[0]
    if mb.use_nodes:
        BSDF_BACKDROP = mb.node_tree.nodes.get('Principled BSDF')


# --------------------------------------------------------------------------
# Render settings
# --------------------------------------------------------------------------
scn.render.resolution_x = RESOLUCAO
scn.render.resolution_y = RESOLUCAO
scn.render.resolution_percentage = 100
scn.render.image_settings.file_format = 'PNG'
scn.render.image_settings.color_mode = 'RGB'

if scn.render.engine == 'CYCLES':
    scn.cycles.samples = AMOSTRAS_CYCLES
    # o denoise mudou de lugar entre versoes -- tenta os dois caminhos
    if hasattr(scn.cycles, 'use_denoising'):
        scn.cycles.use_denoising = True            # 2.90+
    vl = bpy.context.view_layer
    if hasattr(vl, 'cycles') and hasattr(vl.cycles, 'use_denoising'):
        vl.cycles.use_denoising = True             # 2.8x
    scn.cycles.use_adaptive_sampling = getattr(scn.cycles, 'use_adaptive_sampling', False)

print('Render: %dx%d | engine=%s | samples=%s'
      % (RESOLUCAO, RESOLUCAO, scn.render.engine, getattr(scn.cycles, 'samples', 'n/a')))


# --------------------------------------------------------------------------
# Randomizacao
# --------------------------------------------------------------------------
PALETA_CADEIRA = [
    (0.42, 0.24, 0.11),   # madeira clara
    (0.24, 0.13, 0.06),   # madeira escura
    (0.55, 0.38, 0.22),   # pinho
    (0.75, 0.75, 0.76),   # metal claro
    (0.12, 0.12, 0.13),   # preto
    (0.80, 0.79, 0.74),   # bege
    (0.16, 0.28, 0.45),   # azul
    (0.45, 0.14, 0.14),   # vermelho escuro
    (0.20, 0.36, 0.22),   # verde
]


def definir_entrada(node, nomes, valor):
    for n in nomes:
        if n in node.inputs:
            node.inputs[n].default_value = valor
            return True
    return False


def assentar_no_chao():
    """Empurra a cadeira no Z ate o vertice mais baixo encostar em z=0.

    Trava de seguranca: nao importa o que a rotacao faca, a cadeira nunca
    afunda no chao nem flutua. Custa 164 vertices por render -- irrelevante
    perto do tempo de um render do Cycles.
    """
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    menor_z = None
    for obj in MALHAS:
        oe = obj.evaluated_get(deps)
        me = oe.to_mesh()
        mw = oe.matrix_world
        for v in me.vertices:
            z = (mw @ v.co).z
            if menor_z is None or z < menor_z:
                menor_z = z
        oe.to_mesh_clear()
    if menor_z is not None:
        chair.location.z -= menor_z


def luminancia(rgb):
    """Luminancia relativa (Rec. 709) -- o quanto a cor 'pesa' visualmente."""
    r, g, b = rgb[0], rgb[1], rgb[2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def randomizar_cadeira(lum_fundo):
    """Rotacao no eixo vertical (SOMADA a rotacao base) + variacao de material.

    A cor sai apenas do subconjunto da paleta que contrasta com o fundo ja
    sorteado. Se nenhuma bater o limite (fundo em luminancia intermediaria),
    pega a de maior diferenca disponivel -- nunca fica sem opcao.

    Somar em ROT_BASE em vez de sobrescrever e' o que evita apagar a rotacao
    de correcao de eixo do FBX.
    """
    chair.rotation_euler = Euler((
        ROT_BASE[0],
        ROT_BASE[1],
        ROT_BASE[2] + random.uniform(0, 2 * math.pi),
    ), 'XYZ')
    assentar_no_chao()

    candidatas = [c for c in PALETA_CADEIRA
                  if abs(luminancia(c) - lum_fundo) >= MIN_CONTRASTE_LUM]
    if candidatas:
        r, g, b = random.choice(candidatas)
    else:
        r, g, b = max(PALETA_CADEIRA, key=lambda c: abs(luminancia(c) - lum_fundo))
    jit = lambda v: min(1.0, max(0.0, v + random.uniform(-0.05, 0.05)))
    cor = (jit(r), jit(g), jit(b), 1.0)
    contraste = abs(luminancia(cor) - lum_fundo)
    # metal espelha o backdrop e some nele mesmo com a cor base contrastando,
    # entao so entra quando sobra folga de contraste
    metal = 0.85 if (contraste > MIN_CONTRASTE_LUM * 1.6 and random.random() < 0.12) else 0.0
    for bsdf in MATERIAIS:
        definir_entrada(bsdf, ['Base Color'], cor)
        definir_entrada(bsdf, ['Roughness'], random.uniform(0.25, 0.85))
        definir_entrada(bsdf, ['Metallic'], metal)


def randomizar_backdrop():
    """Sorteia a cor do fundo e devolve a luminancia dela.

    A faixa de cor NAO foi alterada -- quem se adapta e' a cadeira.
    """
    if BSDF_BACKDROP is None:
        return 0.45
    # cinza aleatorio, as vezes com um leve tom de cor
    v = random.uniform(0.15, 0.75)
    if random.random() < 0.35:
        cor = (min(1.0, v + random.uniform(-0.08, 0.08)),
               min(1.0, v + random.uniform(-0.08, 0.08)),
               min(1.0, v + random.uniform(-0.08, 0.08)), 1.0)
    else:
        cor = (v, v, v, 1.0)
    definir_entrada(BSDF_BACKDROP, ['Base Color'], cor)
    definir_entrada(BSDF_BACKDROP, ['Roughness'], random.uniform(0.6, 1.0))
    return luminancia(cor)


def randomizar_luz(lum_fundo):
    """Gira o rig inteiro (muda a direcao da sombra) e varia energia/cor.

    A energia e' reduzida quando o fundo esta claro: backdrop claro devolve
    mais luz indireta, e era a soma disso com energia alta que lavava a
    imagem e apagava a silhueta da cadeira.
    """
    if rig is not None:
        rig.rotation_euler = Euler((0.0, 0.0, random.uniform(0, 2 * math.pi)), 'XYZ')
    compensacao = 1.0 - 0.40 * lum_fundo      # fundo 0.75 -> 70% da energia
    for o in LUZES:
        base = ENERGIA_BASE.get(o.name, 150.0)
        o.data.energy = base * random.uniform(0.55, 1.45) * compensacao
        # temperatura de cor: de quente (tungstenio) a fria (sombra de ceu)
        t = random.uniform(-1.0, 1.0)
        o.data.color = (1.0, 1.0 - 0.06 * abs(t), max(0.0, 1.0 - 0.14 * t))


def randomizar_camera():
    """Posiciona a camera em coordenadas esfericas em volta da cadeira."""
    dist = random.uniform(DIST_MIN, DIST_MAX)
    azim = random.uniform(0, 2 * math.pi)
    elev = math.radians(random.uniform(ELEV_MIN, ELEV_MAX))

    alvo = Vector((
        random.uniform(-JITTER_ALVO, JITTER_ALVO) * ALTURA,
        random.uniform(-JITTER_ALVO, JITTER_ALVO) * ALTURA,
        CENTRO.z + random.uniform(-JITTER_ALVO, JITTER_ALVO) * ALTURA,
    ))
    target.location = alvo

    cam.location = Vector((
        alvo.x + dist * math.cos(elev) * math.cos(azim),
        alvo.y + dist * math.cos(elev) * math.sin(azim),
        alvo.z + dist * math.sin(elev),
    ))
    cam.data.lens = random.uniform(LENTE_MIN, LENTE_MAX)


# --------------------------------------------------------------------------
# Bounding box ajustada (a partir dos vertices reais)
# --------------------------------------------------------------------------
def bbox_projetada():
    """Projeta todos os vertices da cadeira na camera.

    Retorna (bbox_cortada, fracao_visivel) ou (None, 0.0) se invalida.
    bbox_cortada = (xmin, ymin, xmax, ymax) normalizado, origem no topo-esquerda.
    """
    deps = bpy.context.evaluated_depsgraph_get()
    xs, ys = [], []

    for obj in MALHAS:
        obj_eval = obj.evaluated_get(deps)
        malha = obj_eval.to_mesh()
        mw = obj_eval.matrix_world
        for v in malha.vertices:
            co = world_to_camera_view(scn, cam, mw @ v.co)
            if co.z <= 0.0:
                obj_eval.to_mesh_clear()
                return None, 0.0      # vertice atras da camera: projecao invalida
            xs.append(co.x)
            ys.append(1.0 - co.y)     # inverte Y: convencao de imagem (0 = topo)
        obj_eval.to_mesh_clear()

    if not xs:
        return None, 0.0

    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    area_total = (x1 - x0) * (y1 - y0)
    if area_total <= 0.0:
        return None, 0.0

    cx0, cx1 = max(0.0, x0), min(1.0, x1)
    cy0, cy1 = max(0.0, y0), min(1.0, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return None, 0.0             # totalmente fora do quadro

    area_visivel = (cx1 - cx0) * (cy1 - cy0)
    return (cx0, cy0, cx1, cy1), area_visivel / area_total


def amostra_valida(bbox, fracao):
    if bbox is None:
        return False
    x0, y0, x1, y1 = bbox
    area = (x1 - x0) * (y1 - y0)
    return area >= MIN_AREA_BBOX and fracao >= MIN_FRACAO_VISIVEL


# --------------------------------------------------------------------------
# Geracao
# --------------------------------------------------------------------------
def caminho(*partes):
    return os.path.join(PASTA_SAIDA, *partes).replace('\\', '/')


for sub in ('images/train', 'images/val', 'labels/train', 'labels/val'):
    os.makedirs(caminho(*sub.split('/')), exist_ok=True)

splits = [('train', N_TREINO), ('val', N_VAL)]
total = N_TREINO + N_VAL
feitos = 0
descartes = 0
t0 = time.time()

print('Gerando %d imagens (%d treino / %d val)...' % (total, N_TREINO, N_VAL))
print('=========================================================')

for split, quantidade in splits:
    i = 0
    tentativas = 0
    limite = quantidade * 12     # trava de seguranca contra loop infinito

    while i < quantidade and tentativas < limite:
        tentativas += 1

        # ordem importa: o fundo e' sorteado primeiro e a cadeira e a luz se
        # adaptam a ele (contraste e energia). Invertido, voltavam as imagens
        # em que a cadeira clara sumia no fundo claro.
        lum_fundo = randomizar_backdrop()
        randomizar_cadeira(lum_fundo)
        randomizar_camera()
        randomizar_luz(lum_fundo)

        # necessario para as matrizes refletirem as mudancas ANTES de projetar
        bpy.context.view_layer.update()

        bbox, fracao = bbox_projetada()
        if not amostra_valida(bbox, fracao):
            descartes += 1
            continue             # pose ruim: sorteia de novo sem gastar render

        x0, y0, x1, y1 = bbox
        xc, yc = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        w, h = x1 - x0, y1 - y0

        nome = '%06d' % i
        scn.render.filepath = caminho('images', split, nome + '.png')
        bpy.ops.render.render(write_still=True)

        with open(caminho('labels', split, nome + '.txt'), 'w') as f:
            f.write('%d %.6f %.6f %.6f %.6f\n' % (CLASSE_ID, xc, yc, w, h))

        i += 1
        feitos += 1
        seg = (time.time() - t0) / feitos
        restante = seg * (total - feitos)
        print('[%s %d/%d] %s.png | %.1fs/img | restante ~%s'
              % (split, i, quantidade, nome, seg,
                 time.strftime('%H:%M:%S', time.gmtime(restante))))

    if i < quantidade:
        print('[aviso] split "%s" parou em %d/%d apos %d tentativas. '
              'Afrouxe MIN_FRACAO_VISIVEL ou aumente DIST_MIN.'
              % (split, i, quantidade, tentativas))

# --------------------------------------------------------------------------
# data.yaml para o Ultralytics
# --------------------------------------------------------------------------
with open(caminho('data.yaml'), 'w') as f:
    f.write('path: %s\n' % PASTA_SAIDA)
    f.write('train: images/train\n')
    f.write('val: images/val\n')
    f.write('nc: 1\n')
    f.write("names: ['%s']\n" % CLASSE_NOME)

print('')
print('=========================================================')
print('Concluido: %d imagens em %.1f min' % (feitos, (time.time() - t0) / 60.0))
print('  %d poses descartadas por enquadramento ruim' % descartes)
print('  data.yaml gravado em %s' % caminho('data.yaml'))
print('')
print('Antes de treinar, rode o 3_verificar_anotacoes.py (fora do Blender)')
print('para conferir visualmente se as caixas batem com as cadeiras.')
print('=========================================================')
