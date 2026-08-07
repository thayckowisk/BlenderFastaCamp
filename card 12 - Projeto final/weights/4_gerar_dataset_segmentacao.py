# -*- coding: utf-8 -*-
"""
SCRIPT 4 -- Dataset de SEGMENTAÇÃO da chair_01 (imagem + máscara).

Rode DEPOIS do 1_montar_cena_cadeira.py, na mesma sessão.
Compatível com Blender 2.83.

Como a máscara é gerada
-----------------------
Usa o passe de Object Index (Pass Index) do Blender -- mesma técnica de
projetos de segmentação sintética. Cada malha da cadeira recebe pass_index=1;
o backdrop fica em 0. O compositor pega esse passe, o nó ID Mask converte em
0/1 e um nó File Output grava a máscara junto com o RGB, no MESMO render.

Vantagem sobre desenhar máscara à mão: é exata por construção, no nível do
pixel, incluindo os vãos entre as ripas do encosto -- um anotador humano
jamais recortaria aqueles buracos com precisão.

Saída
-----
    C:/tmp/chairs_seg/
        images/0000.png     RGB
        masks/0000.png      máscara (branco = cadeira, preto = fundo)

Depois rode o 5_mascaras_para_yolo_seg.py (fora do Blender) para converter
as máscaras em polígonos no formato YOLOv8-seg.
"""

import bpy
import math
import random
import time
import os
from mathutils import Vector, Euler
from bpy_extras.object_utils import world_to_camera_view

# =========================== CONFIG ===========================
PASTA_SAIDA = 'C:/tmp/chairs_seg'

N_IMAGENS = 500
RESOLUCAO = 416
AMOSTRAS_CYCLES = 48
SEED = 42

DIST_MIN, DIST_MAX = 1.8, 3.0
ELEV_MIN, ELEV_MAX = 5.0, 65.0
LENTE_MIN, LENTE_MAX = 40.0, 60.0
JITTER_ALVO = 0.18

MIN_AREA_BBOX = 0.010
MIN_FRACAO_VISIVEL = 0.75
MIN_CONTRASTE_LUM = 0.22

PASS_INDEX_CADEIRA = 1
# ==============================================================

random.seed(SEED)
scn = bpy.context.scene

CHAIR_NAME = scn.get('chair_name', 'chair_01')
chair = bpy.data.objects.get(CHAIR_NAME)
cam = scn.camera
target = bpy.data.objects.get('ObjectTarget')
rig = bpy.data.objects.get('LightRig')
backdrop = bpy.data.objects.get('Backdrop')

if chair is None or cam is None or target is None:
    raise RuntimeError('Cena incompleta -- rode o 1_montar_cena_cadeira.py primeiro.')

ALTURA = scn.get('chair_altura', 0.884)
CENTRO = Vector(scn.get('chair_centro', (0.0, 0.0, ALTURA / 2)))
ROT_BASE = tuple(scn.get('chair_rot_base', tuple(chair.rotation_euler)))


def coletar_malhas(obj):
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
# 1. Pass Index nos objetos
# --------------------------------------------------------------------------
for m in MALHAS:
    m.pass_index = PASS_INDEX_CADEIRA
if backdrop is not None:
    backdrop.pass_index = 0
print('pass_index=%d aplicado em %d malha(s)' % (PASS_INDEX_CADEIRA, len(MALHAS)))

# --------------------------------------------------------------------------
# 2. Render settings + passe de Object Index
# --------------------------------------------------------------------------
scn.render.resolution_x = RESOLUCAO
scn.render.resolution_y = RESOLUCAO
scn.render.resolution_percentage = 100
scn.render.image_settings.file_format = 'PNG'
scn.render.image_settings.color_mode = 'RGB'

view_layer = bpy.context.view_layer
view_layer.use_pass_object_index = True     # habilita o passe IndexOB

if scn.render.engine == 'CYCLES':
    scn.cycles.samples = AMOSTRAS_CYCLES
    if hasattr(scn.cycles, 'use_denoising'):
        scn.cycles.use_denoising = True
    if hasattr(view_layer, 'cycles') and hasattr(view_layer.cycles, 'use_denoising'):
        view_layer.cycles.use_denoising = True

# --------------------------------------------------------------------------
# 3. Compositor: RGB + máscara saem do MESMO render
# --------------------------------------------------------------------------
scn.use_nodes = True
nt = scn.node_tree
nt.nodes.clear()

n_rl = nt.nodes.new('CompositorNodeRLayers')
n_rl.location = (-400, 0)

n_idmask = nt.nodes.new('CompositorNodeIDMask')
n_idmask.location = (-150, -200)
n_idmask.index = PASS_INDEX_CADEIRA
n_idmask.use_antialiasing = True    # borda suave; o script 5 limiariza

n_out = nt.nodes.new('CompositorNodeOutputFile')
n_out.location = (200, 0)
n_out.base_path = PASTA_SAIDA
n_out.format.file_format = 'PNG'

# dois slots: um para o RGB, outro para a máscara
n_out.file_slots.clear()
n_out.file_slots.new('images/')
n_out.file_slots.new('masks/')

# a máscara NÃO pode passar pelo Filmic -- ele mudaria os valores do índice.
# Nem toda versão expõe color_management no formato do slot, então tenta e
# segue em frente se não existir (o script 5 limiariza de forma relativa).
try:
    slot_mask = n_out.file_slots[1]
    slot_mask.use_node_format = False
    slot_mask.format.file_format = 'PNG'
    slot_mask.format.color_mode = 'BW'
    slot_mask.format.color_depth = '8'
    slot_mask.format.color_management = 'OVERRIDE'
    slot_mask.format.view_settings.view_transform = 'Standard'
    print('máscara: color management fixado em Standard')
except Exception as e:
    print('[aviso] não consegui sobrescrever o color management da máscara: %s' % e)

if 'IndexOB' not in n_rl.outputs:
    raise RuntimeError('Saída IndexOB ausente -- o passe de Object Index não ativou.')

nt.links.new(n_rl.outputs['Image'], n_out.inputs[0])
nt.links.new(n_rl.outputs['IndexOB'], n_idmask.inputs['ID value'])
nt.links.new(n_idmask.outputs['Alpha'], n_out.inputs[1])

os.makedirs(os.path.join(PASTA_SAIDA, 'images'), exist_ok=True)
os.makedirs(os.path.join(PASTA_SAIDA, 'masks'), exist_ok=True)

# --------------------------------------------------------------------------
# 4. Randomização (mesma do script 2 de detecção)
# --------------------------------------------------------------------------
PALETA_CADEIRA = [
    (0.42, 0.24, 0.11), (0.24, 0.13, 0.06), (0.55, 0.38, 0.22),
    (0.75, 0.75, 0.76), (0.12, 0.12, 0.13), (0.80, 0.79, 0.74),
    (0.16, 0.28, 0.45), (0.45, 0.14, 0.14), (0.20, 0.36, 0.22),
]


def luminancia(rgb):
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def definir_entrada(node, nomes, valor):
    for n in nomes:
        if n in node.inputs:
            node.inputs[n].default_value = valor
            return True
    return False


def assentar_no_chao():
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


def randomizar_cadeira(lum_fundo):
    chair.rotation_euler = Euler((ROT_BASE[0], ROT_BASE[1],
                                  ROT_BASE[2] + random.uniform(0, 2 * math.pi)), 'XYZ')
    assentar_no_chao()
    candidatas = [c for c in PALETA_CADEIRA
                  if abs(luminancia(c) - lum_fundo) >= MIN_CONTRASTE_LUM]
    r, g, b = (random.choice(candidatas) if candidatas else
               max(PALETA_CADEIRA, key=lambda c: abs(luminancia(c) - lum_fundo)))
    jit = lambda v: min(1.0, max(0.0, v + random.uniform(-0.05, 0.05)))
    cor = (jit(r), jit(g), jit(b), 1.0)
    contraste = abs(luminancia(cor) - lum_fundo)
    metal = 0.85 if (contraste > MIN_CONTRASTE_LUM * 1.6 and random.random() < 0.12) else 0.0
    for bsdf in MATERIAIS:
        definir_entrada(bsdf, ['Base Color'], cor)
        definir_entrada(bsdf, ['Roughness'], random.uniform(0.25, 0.85))
        definir_entrada(bsdf, ['Metallic'], metal)


def randomizar_backdrop():
    if BSDF_BACKDROP is None:
        return 0.45
    v = random.uniform(0.15, 0.75)
    if random.random() < 0.35:
        cor = tuple(min(1.0, v + random.uniform(-0.08, 0.08)) for _ in range(3)) + (1.0,)
    else:
        cor = (v, v, v, 1.0)
    definir_entrada(BSDF_BACKDROP, ['Base Color'], cor)
    definir_entrada(BSDF_BACKDROP, ['Roughness'], random.uniform(0.6, 1.0))
    return luminancia(cor)


def randomizar_luz(lum_fundo):
    if rig is not None:
        rig.rotation_euler = Euler((0.0, 0.0, random.uniform(0, 2 * math.pi)), 'XYZ')
    compensacao = 1.0 - 0.40 * lum_fundo
    for o in LUZES:
        base = ENERGIA_BASE.get(o.name, 150.0)
        o.data.energy = base * random.uniform(0.55, 1.45) * compensacao
        t = random.uniform(-1.0, 1.0)
        o.data.color = (1.0, 1.0 - 0.06 * abs(t), max(0.0, 1.0 - 0.14 * t))


def randomizar_camera():
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
# 5. Pré-checagem de enquadramento (barata, evita render desperdiçado)
# --------------------------------------------------------------------------
def enquadramento_ok():
    deps = bpy.context.evaluated_depsgraph_get()
    xs, ys = [], []
    for obj in MALHAS:
        oe = obj.evaluated_get(deps)
        me = oe.to_mesh()
        mw = oe.matrix_world
        for v in me.vertices:
            co = world_to_camera_view(scn, cam, mw @ v.co)
            if co.z <= 0.0:
                oe.to_mesh_clear()
                return False
            xs.append(co.x)
            ys.append(1.0 - co.y)
        oe.to_mesh_clear()
    if not xs:
        return False
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    total = (x1 - x0) * (y1 - y0)
    if total <= 0:
        return False
    cx0, cx1 = max(0.0, x0), min(1.0, x1)
    cy0, cy1 = max(0.0, y0), min(1.0, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return False
    visivel = (cx1 - cx0) * (cy1 - cy0)
    return visivel >= MIN_AREA_BBOX and (visivel / total) >= MIN_FRACAO_VISIVEL


# --------------------------------------------------------------------------
# 6. Loop de geração
# --------------------------------------------------------------------------
print('Gerando %d pares imagem+máscara em %s' % (N_IMAGENS, PASTA_SAIDA))
print('=========================================================')

t0 = time.time()
gerados = 0
descartes = 0
tentativas = 0
limite = N_IMAGENS * 12

while gerados < N_IMAGENS and tentativas < limite:
    tentativas += 1

    lum_fundo = randomizar_backdrop()
    randomizar_cadeira(lum_fundo)
    randomizar_camera()
    randomizar_luz(lum_fundo)
    bpy.context.view_layer.update()

    if not enquadramento_ok():
        descartes += 1
        continue

    # o número do frame vira o nome do arquivo: images/0000.png, masks/0000.png
    scn.frame_set(gerados)
    bpy.ops.render.render(write_still=False)   # o File Output grava os dois

    gerados += 1
    seg = (time.time() - t0) / gerados
    restante = seg * (N_IMAGENS - gerados)
    print('[%d/%d] %04d | %.1fs/img | restante ~%s'
          % (gerados, N_IMAGENS, gerados - 1, seg,
             time.strftime('%H:%M:%S', time.gmtime(restante))))

print('')
print('=========================================================')
print('Concluído: %d pares em %.1f min' % (gerados, (time.time() - t0) / 60.0))
print('  %d poses descartadas por enquadramento' % descartes)
print('  imagens: %s/images' % PASTA_SAIDA)
print('  máscaras: %s/masks' % PASTA_SAIDA)
print('')
print('Agora rode o 5_mascaras_para_yolo_seg.py (fora do Blender).')
print('=========================================================')
