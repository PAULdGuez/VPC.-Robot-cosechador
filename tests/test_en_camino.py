"""Tres modos para una orden nueva con la punta en camino, y el rojo que los acompaña (criterios J1 a J7, aprobados por
Paul el 25-nov-2026).

«En camino» = la punta todavía no llega a donde la mandó un slider de «Punta X Y Z». Lo que pasa al mover OTRO slider
en ese momento lo decide el selector «En camino», en la esquina de las pestañas:
- Reemplazar: deja lo que hacía y va en línea recta al punto que marcan los tres sliders.
- En cola: termina el tramo que lleva y después recorre los siguientes, en el orden pedido.
- Cancelar: se detiene donde va y recorre solo el tramo nuevo.

Lo esperado (el punto de salida, la esquina, el final, el borde del rojo) se calcula ANTES de mover: leerlo después de
actuar esconde justo el defecto que se busca (así pasó en verde la primera prueba de J8).
"""
import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5 import QtCore, QtWidgets

import auditoria_interfaz as au
from cosechador.movimiento.trayectorias import posicion
from cosechador.robot.cinematica_inversa import TOLERANCIA_TRAMOS_MM, angulos_servo, tramos_alcanzables
from cosechador.ui import ventana as modulo_ventana
from cosechador.ui.ventana import AJUSTE_EN_CAMINO, PASOS_POR_REPINTADO_JOG, PESTANA_XYZ
from test_editor_puntos import reabrir  # noqa: F401 (ES UN FIXTURE)
from test_grabacion_ventana import eventos, ventana  # noqa: F401 (ventana ES EL FIXTURE)
from test_sliders import (colocar, con_el_eje, correr_pasos, defectos_de_lo_pintado, negativas_del_planeador,
                          poses_al_azar, tramos_de)

MODOS = ("reemplazar", "en_cola", "cancelar")
MINIMO = (1243, 648)       # EL TAMAÑO MÍNIMO DE LA VENTANA, MEDIDO ANTES DE PONER EL SELECTOR: NO DEBE CRECER
BASE = (575, 996, 1147)    # DESDE AQUÍ, X → 875 Y DESPUÉS Y → 1296 SE PUEDEN RECORRER EN LOS TRES MODOS (MEDIDO)
X_NUEVA, Y_NUEVA = 875, 1296
EN_LA_RECTA_MM = 0.5       # CADA PASO DEL PLAN ES UN PUNTO DE LA RECTA: MEDIO MILÍMETRO ES HOLGURA DE SOBRA


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INSTRUMENTO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def procesar():
    for _ in range(3):
        QtWidgets.QApplication.processEvents()


def modo(w, clave):
    """La persona elige el modo en el selector."""
    w.selector_en_camino.setCurrentIndex(w.selector_en_camino.findData(clave))
    assert w.selector_en_camino.currentData() == clave


def en_base(w, robot, clave):
    """El robot quieto en BASE, con el modo `clave`. Devuelve el punto exacto de salida (con sus decimales)."""
    modo(w, clave)
    colocar(w, angulos_servo(robot, BASE))
    return tuple(w.posicion)


def a_medio_camino(w, pasos=6):
    """Unos pasos de lo pendiente: la punta queda en camino. Devuelve dónde va."""
    correr_pasos(w, pasos)
    assert w.cola, "control: el movimiento se acabó; la punta ya no va en camino"
    return tuple(w.posicion)


def recorrer(w):
    """Ejecuta todo lo pendiente, paso por paso, y devuelve por dónde fue pasando la punta."""
    w.temporizador_movimiento.stop()
    camino = []
    while w.cola:
        w._siguiente_paso()
        w.reloj.t += w.temporizador_movimiento.interval() / 1000
        camino.append(tuple(w.posicion))
    w._siguiente_paso()  # COLA VACÍA: EL MOVIMIENTO TERMINA
    return camino


def lejos_del_tramo(p, a, b):
    """Distancia (mm) del punto p al tramo recto a → b."""
    p, a, b = (np.asarray(v, dtype=float) for v in (p, a, b))
    t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
    return float(np.linalg.norm(p - (a + t * (b - a))))


def fuera_del_camino(camino, *esquinas):
    """Los puntos por donde pasó la punta que no caen sobre las rectas que unen esas esquinas. [] = no se salió."""
    return [tuple(round(c, 2) for c in p) for p in camino
            if min(lejos_del_tramo(p, a, b) for a, b in zip(esquinas, esquinas[1:])) > EN_LA_RECTA_MM]


def cuando_pasa_por(camino, punto):
    """En qué paso la punta está sobre `punto` (None si nunca)."""
    paso = min(range(len(camino)), key=lambda i: math.dist(camino[i], punto))
    return paso if math.dist(camino[paso], punto) <= 0.01 else None


def donde_marcan(w):
    return [s.value() for s in w.sliders_xyz]


def borde_desde(w, punto, eje, extremo):
    """Hasta dónde deja llevar ese slider el rojo calculado desde `punto`, y no desde lo que el slider tenga pintado."""
    s = w.sliders_xyz[eje]
    tramos = tramos_alcanzables(w.robot, punto, eje, min(s.minimum(), punto[eje]), max(s.maximum(), punto[eje]))
    for a, b in tramos:
        if a - TOLERANCIA_TRAMOS_MM <= punto[eje] <= b + TOLERANCIA_TRAMOS_MM:
            return min(max(extremo, math.ceil(a)), math.floor(b))
    return extremo


def en_la_esquina_de_las_pestanas(w, control):
    """El rectángulo del control, en coordenadas de las pestañas, si está en la franja de la barra y a su derecha."""
    pestanas = w.PESTANAS_MOVIMIENTO
    barra = pestanas.tabBar().geometry()
    zona = QtCore.QRect(control.mapTo(pestanas, QtCore.QPoint(0, 0)), control.size())
    assert zona.left() > barra.right() and zona.right() <= pestanas.width(), "no está a la derecha de las pestañas"
    assert zona.center().y() <= barra.bottom(), "no está a la altura de la barra de pestañas"
    return zona


def auditar_la_banda(w):
    """El auditor de interfaz, sin los dos solapes que el panel de VISIÓN ya trae al tamaño mínimo de la ventana (Tono y
    Saturación: son de antes y de otro lado, y están anotados en los pendientes; la suite audita desde 1280×760)."""
    return [h for h in au.auditar(w) if "VS_TONO" not in h and "VS_SATURACION" not in h]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% J1: EL SELECTOR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_j01_el_selector_esta_en_la_esquina_de_las_pestanas_y_solo_se_ve_en_punta_xyz(ventana):
    w, pestanas, selector = ventana, ventana.PESTANAS_MOVIMIENTO, ventana.selector_en_camino
    w.resize(*MINIMO)
    w.show()                                              # OFFSCREEN: NO SE ABRE NADA A LA VISTA
    assert [selector.itemText(i) for i in range(selector.count())] == ["Reemplazar", "En cola", "Cancelar"]
    assert all(nombre in selector.toolTip() for nombre in ("Reemplazar", "En cola", "Cancelar")), "la ayuda no explica los modos"
    for indice in range(pestanas.count()):
        pestanas.setCurrentIndex(indice)
        procesar()
        donde, barra = pestanas.tabText(indice), pestanas.tabBar()
        assert selector.isVisible() == (indice == PESTANA_XYZ), donde
        assert w.etiqueta_en_camino.isVisible() == (indice == PESTANA_XYZ), donde
        assert barra.width() >= barra.sizeHint().width(), f"{donde}: las pestañas quedaron recortadas"
        assert auditar_la_banda(w) == [], donde           # NI SE ENCIMA, NI SE CORTA, NI PIERDE CONTRASTE
    pestanas.setCurrentIndex(PESTANA_XYZ)
    procesar()
    assert w.etiqueta_en_camino.text() == "En camino"
    en_la_esquina_de_las_pestanas(w, selector)
    minimo = w.minimumSizeHint()
    assert minimo.width() <= MINIMO[0] and minimo.height() <= MINIMO[1], "el selector agrandó la ventana"


def test_j01_la_primera_vez_abre_en_cola_y_recuerda_el_modo_elegido(ventana, reabrir):
    w = ventana
    assert w.selector_en_camino.currentData() == "en_cola"
    modo(w, "cancelar")
    otra = reabrir(w)                                     # «CERRAR LA APP Y VOLVER A ABRIRLA»
    assert otra.selector_en_camino.currentData() == "cancelar"
    modo(otra, "reemplazar")
    assert reabrir(otra).selector_en_camino.currentData() == "reemplazar"


def test_j01_un_modo_guardado_que_la_app_no_conoce_abre_en_cola(ventana, reabrir):
    w = ventana
    w.ajustes.setValue(AJUSTE_EN_CAMINO, "abortar")       # P. EJ. UN NOMBRE DE UNA VERSIÓN ANTERIOR
    assert reabrir(w).selector_en_camino.currentData() == "en_cola"


def test_j01_mientras_se_graba_el_aviso_y_el_selector_caben_juntos_en_la_esquina(ventana):
    """El aviso «Grabando» ya vivía en esa esquina, y se graba moviendo desde «Punta X Y Z»: se ven los dos a la vez."""
    w, pestanas = ventana, ventana.PESTANAS_MOVIMIENTO
    w.resize(*MINIMO)
    w.show()
    pestanas.setCurrentIndex(PESTANA_XYZ)
    w.alternar_grabacion()
    procesar()
    assert w.aviso_grabando.isVisible() and w.selector_en_camino.isVisible()
    aviso, selector = (en_la_esquina_de_las_pestanas(w, c) for c in (w.aviso_grabando, w.selector_en_camino))
    assert not aviso.intersects(selector)
    assert aviso.width() >= w.aviso_grabando.sizeHint().width(), "el aviso quedó recortado"
    barra = pestanas.tabBar()
    assert barra.width() >= barra.sizeHint().width(), "las pestañas quedaron recortadas"
    assert auditar_la_banda(w) == []
    w.alternar_grabacion()                                # NO SE MOVIÓ: NO HAY NADA QUE GUARDAR
    procesar()
    assert not w.aviso_grabando.isVisible() and w.etiqueta_en_camino.isVisible()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% J2: REEMPLAZAR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_j02_reemplazar_deja_lo_que_hacia_y_va_en_recta_al_punto_de_los_sliders(ventana, robot):
    w = ventana
    salida = en_base(w, robot, "reemplazar")
    esquina, final = con_el_eje(salida, 0, X_NUEVA), (X_NUEVA, Y_NUEVA, salida[2])
    w.SLIDER_X.setValue(X_NUEVA)
    aqui = a_medio_camino(w)
    assert salida[0] + 10 < aqui[0] < X_NUEVA - 10, "control: la punta no quedó a medio camino"
    w.SLIDER_Y.setValue(Y_NUEVA)
    camino = recorrer(w)
    assert camino[-1] == pytest.approx(final, abs=0.01)
    assert fuera_del_camino(camino, aqui, final) == []            # UNA SOLA RECTA, DESDE DONDE IBA
    assert cuando_pasa_por(camino, esquina) is None               # NO TERMINÓ PRIMERO EL TRAMO DE X
    assert donde_marcan(w) == [X_NUEVA, Y_NUEVA, round(salida[2])]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% J3: EN COLA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_j03_en_cola_termina_el_tramo_que_lleva_y_despues_recorre_el_siguiente(ventana, robot):
    w = ventana
    salida = en_base(w, robot, "en_cola")
    esquina = con_el_eje(salida, 0, X_NUEVA)
    final = con_el_eje(esquina, 1, Y_NUEVA)
    w.SLIDER_X.setValue(X_NUEVA)
    aqui = a_medio_camino(w)
    w.SLIDER_Y.setValue(Y_NUEVA)
    camino = recorrer(w)
    assert camino[-1] == pytest.approx(final, abs=0.01)
    assert fuera_del_camino(camino, aqui, esquina, final) == []
    llega = cuando_pasa_por(camino, esquina)
    assert llega is not None, "la punta no terminó el tramo que llevaba"
    assert all(abs(p[1] - salida[1]) <= EN_LA_RECTA_MM for p in camino[:llega])   # ANTES DE LA ESQUINA, SOLO SE MOVIÓ X
    assert all(abs(p[0] - X_NUEVA) <= EN_LA_RECTA_MM for p in camino[llega:])     # DESPUÉS, SOLO Y
    assert donde_marcan(w) == [X_NUEVA, Y_NUEVA, round(salida[2])]


def test_j03_en_cola_los_tramos_se_recorren_en_el_orden_pedido(ventana, robot):
    """Cuatro órdenes con la punta en camino, y un slider que se pide dos veces (X, Y, Z y otra vez X)."""
    w = ventana
    salida = en_base(w, robot, "en_cola")
    z_nueva, x_otra = round(salida[2]) - 100, 775
    e1 = con_el_eje(salida, 0, X_NUEVA)
    e2 = con_el_eje(e1, 1, Y_NUEVA)
    e3 = con_el_eje(e2, 2, z_nueva)
    final = con_el_eje(e3, 0, x_otra)
    w.SLIDER_X.setValue(X_NUEVA)
    aqui = a_medio_camino(w)
    w.SLIDER_Y.setValue(Y_NUEVA)
    w.SLIDER_Z.setValue(z_nueva)
    a_medio_camino(w, 3)
    w.SLIDER_X.setValue(x_otra)
    assert donde_marcan(w) == [x_otra, Y_NUEVA, z_nueva], "control: el rojo frenó alguno de los pedidos"
    camino = [aqui] + recorrer(w)
    assert camino[-1] == pytest.approx(final, abs=0.01)
    assert fuera_del_camino(camino, aqui, e1, e2, e3, final) == []
    pasos = [cuando_pasa_por(camino, e) for e in (e1, e2, e3, final)]
    assert None not in pasos and pasos == sorted(pasos), f"las esquinas no se visitaron en el orden pedido: {pasos}"
    assert negativas_del_planeador(w) == []


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% J4: CANCELAR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_j04_cancelar_se_detiene_donde_va_el_slider_regresa_y_recorre_solo_el_tramo_nuevo(ventana, robot):
    w = ventana
    salida = en_base(w, robot, "cancelar")
    w.SLIDER_X.setValue(X_NUEVA)
    aqui = a_medio_camino(w)
    assert salida[0] + 10 < aqui[0] < X_NUEVA - 10, "control: la punta no quedó a medio camino"
    final = con_el_eje(aqui, 1, Y_NUEVA)
    w.SLIDER_Y.setValue(Y_NUEVA)
    assert w.SLIDER_X.value() == round(aqui[0]), "el slider que iba en camino no regresó a donde se detuvo la punta"
    camino = recorrer(w)
    assert camino[-1] == pytest.approx(final, abs=0.01)
    assert fuera_del_camino(camino, aqui, final) == []
    assert max(p[0] for p in camino) <= aqui[0] + EN_LA_RECTA_MM, "la punta siguió avanzando sobre X"
    assert donde_marcan(w) == [round(aqui[0]), Y_NUEVA, round(salida[2])]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% J2 A J4: UNA ORDEN QUE NO SE PUEDE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@pytest.mark.parametrize("clave", MODOS)
def test_j02_j03_j04_una_orden_que_el_planeador_niega_no_cambia_lo_que_ya_iba(ventana, robot, clave, monkeypatch):
    """El riesgo declarado: no se mueve, sale un aviso y el slider regresa. Lo que ya iba en camino sigue igual."""
    w = ventana
    salida = en_base(w, robot, clave)
    w.SLIDER_X.setValue(X_NUEVA)
    a_medio_camino(w)
    pendiente = list(w.cola)
    monkeypatch.setattr(modulo_ventana, "tramo_continuo", lambda *a, **k: (None, "pasa por una zona que el robot no alcanza"))
    w.SLIDER_Y.setValue(Y_NUEVA)
    monkeypatch.undo()
    assert list(w.cola) == pendiente
    assert donde_marcan(w) == [X_NUEVA, round(salida[1]), round(salida[2])]
    assert len(negativas_del_planeador(w)) == 1
    assert recorrer(w)[-1] == pytest.approx(con_el_eje(salida, 0, X_NUEVA), abs=0.01)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% J5: EL MISMO SLIDER, VARIAS VECES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@pytest.mark.parametrize("clave", MODOS)
def test_j05_mover_varias_veces_el_mismo_slider_solo_cambia_donde_termina_ese_tramo(ventana, robot, clave):
    w = ventana
    salida = en_base(w, robot, clave)
    final = con_el_eje(salida, 0, 590)
    w.SLIDER_X.setValue(X_NUEVA)
    visto = [a_medio_camino(w, 4)]
    for valor in (700, 925, 760):                         # TRES CAMBIOS SEGUIDOS, SIN DAR TIEMPO A NADA
        w.SLIDER_X.setValue(valor)
    visto.append(a_medio_camino(w, 3))
    assert visto[-1][0] > 590 + 10, "control: la punta todavía no pasaba de 590"
    w.SLIDER_X.setValue(590)                              # AHORA EL FINAL QUEDA ATRÁS DE LA PUNTA: SE REGRESA
    camino = recorrer(w)
    assert camino[-1] == pytest.approx(final, abs=0.01)
    assert all(math.dist(p[1:], salida[1:]) <= EN_LA_RECTA_MM for p in visto + camino), "la punta se salió de la recta de X"
    assert max(p[0] for p in camino) <= visto[-1][0] + EN_LA_RECTA_MM, "la punta siguió hacia un final que ya no era"
    assert donde_marcan(w) == [590, round(salida[1]), round(salida[2])]


def test_j05_en_cola_el_tramo_que_no_ha_empezado_se_cambia_sin_tocar_el_que_va_en_camino(ventana, robot):
    w = ventana
    salida = en_base(w, robot, "en_cola")
    esquina = con_el_eje(salida, 0, X_NUEVA)
    final = con_el_eje(esquina, 1, 1096)
    w.SLIDER_X.setValue(X_NUEVA)
    aqui = a_medio_camino(w)
    for valor in (Y_NUEVA, 1196, 1096):
        w.SLIDER_Y.setValue(valor)
    camino = recorrer(w)
    assert camino[-1] == pytest.approx(final, abs=0.01)
    assert fuera_del_camino(camino, aqui, esquina, final) == []
    assert cuando_pasa_por(camino, esquina) is not None, "la punta no terminó el tramo que llevaba"
    assert max(p[1] for p in camino) <= 1096 + EN_LA_RECTA_MM, "quedó algo del tramo hacia 1296"
    assert donde_marcan(w) == [X_NUEVA, 1096, round(salida[2])]


def test_j05_en_cola_regresar_un_slider_a_donde_sale_su_tramo_quita_ese_tramo(ventana, robot):
    w = ventana
    salida = en_base(w, robot, "en_cola")
    esquina = con_el_eje(salida, 0, X_NUEVA)
    w.SLIDER_X.setValue(X_NUEVA)
    aqui = a_medio_camino(w)
    w.SLIDER_Y.setValue(Y_NUEVA)
    w.SLIDER_Y.setValue(round(salida[1]))
    camino = recorrer(w)
    assert camino[-1] == pytest.approx(esquina, abs=0.01)
    assert fuera_del_camino(camino, aqui, esquina) == []
    assert donde_marcan(w) == [X_NUEVA, round(salida[1]), round(salida[2])]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% J6: EL ROJO, CON EL ROBOT EN CAMINO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def una_orden_larga(w, n, servos, minimo_mm=60):
    """Deja el robot en `servos` y calcula una orden de un solo slider (por turnos X, Y, Z; hacia un lado y hacia el otro)
    que el rojo deja pasar. Devuelve (eje, pedido, salida), o None si desde esa pose no hay a dónde ir."""
    colocar(w, servos)
    salida, eje = tuple(w.posicion), n % 3
    slider = w.sliders_xyz[eje]
    pedido = slider.acotar(slider.value() + (200 if n % 2 else -200), salida[eje])
    return (eje, pedido, salida) if abs(pedido - salida[eje]) >= minimo_mm else None


@pytest.mark.parametrize("clave", ("reemplazar", "en_cola"))
def test_j06_con_la_punta_en_camino_el_rojo_es_el_del_punto_que_marcan_los_sliders(ventana, clave):
    """En estos dos modos, si se mueve otro slider el tramo nuevo termina (o sale) en el punto al que la punta va: el rojo
    dice hasta dónde se puede llevar cada slider desde ESE punto, y no cambia mientras la punta llega."""
    w = ventana
    modo(w, clave)
    revisadas, cambiaron = 0, 0
    for n, servos in enumerate(poses_al_azar(w, 8)):
        orden = una_orden_larga(w, n, servos)
        if orden is None:
            continue
        eje, pedido, salida = orden
        quieto = tramos_de(w)
        w.sliders_xyz[eje].setValue(pedido)
        al_aceptar = tramos_de(w)
        a_medio_camino(w, 4)
        assert tramos_de(w) == al_aceptar
        assert defectos_de_lo_pintado(w, desde=con_el_eje(salida, eje, pedido)) == []
        revisadas += 1
        cambiaron += al_aceptar != quieto
        w.detener()
    assert revisadas >= 4 and cambiaron >= 4, "control: casi ninguna pose tenía a dónde ir; la prueba no mide nada"


def test_j06_en_cancelar_el_rojo_sigue_a_la_punta_mientras_va_en_camino(ventana):
    """En «Cancelar» el tramo nuevo sale de donde va la punta: el rojo se vuelve a calcular cada pocos pasos."""
    w = ventana
    modo(w, "cancelar")
    revisadas = 0
    for n, servos in enumerate(poses_al_azar(w, 6)):
        orden = una_orden_larga(w, n, servos)
        if orden is None:
            continue
        eje, pedido, _ = orden
        w.sliders_xyz[eje].setValue(pedido)
        for _ in range(2):
            a_medio_camino(w, PASOS_POR_REPINTADO_JOG)
            assert defectos_de_lo_pintado(w) == []        # DESDE DONDE VA LA PUNTA EN ESE MOMENTO
        revisadas += 1
        w.detener()
    assert revisadas >= 3, "control: casi ninguna pose tenía a dónde ir; la prueba no mide nada"


def test_j06_en_cancelar_el_slider_frena_con_el_rojo_de_donde_va_la_punta(ventana):
    """Entre un repintado y el siguiente la punta avanza. Al mover otro slider, su rojo se calcula en ese momento: se
    detiene en el borde de verdad y la punta llega hasta ahí (con el rojo viejo el planeador se negaba la mitad de las veces)."""
    w = ventana
    modo(w, "cancelar")
    pedidos = 0
    for n, servos in enumerate(poses_al_azar(w, 24)):
        orden = una_orden_larga(w, n, servos)
        if orden is None:
            continue
        eje, pedido, _ = orden
        w.sliders_xyz[eje].setValue(pedido)
        a_medio_camino(w, PASOS_POR_REPINTADO_JOG + 1)    # UN PASO DESPUÉS DEL ÚLTIMO REPINTADO
        aqui, otro = tuple(w.posicion), (eje + 1) % 3
        slider = w.sliders_xyz[otro]
        extremo = slider.maximum() if (n // 3) % 2 else slider.minimum()
        borde = borde_desde(w, aqui, otro, extremo)
        if abs(borde - aqui[otro]) < 0.5:
            w.detener()
            continue                                      # LA PUNTA YA ESTABA EN ESE BORDE
        slider.setValue(extremo)
        camino = [paso.servos for paso in w.cola if paso.servos is not None]
        assert slider.value() == borde, f"{'XYZ'[otro]} se detuvo en {slider.value()} y el borde está en {borde}"
        assert camino and tuple(posicion(w.robot, camino[-1])) == pytest.approx(con_el_eje(aqui, otro, borde), abs=0.5)
        pedidos += 1
        w.detener()
    assert pedidos >= 12, "control: casi ninguna pose tenía a dónde ir; la prueba no mide nada"
    assert negativas_del_planeador(w) == []


def test_j06_con_la_perilla_agarrada_el_rojo_de_los_otros_sliders_se_pone_al_dia(ventana, robot):
    """Durante un arrastre no se recalcula el rojo de los otros dos sliders en cada pixel: se ponen al día cada pocos
    pasos y al soltar."""
    w = ventana
    salida = en_base(w, robot, "en_cola")
    w.SLIDER_X.setSliderDown(True)                        # LA PERSONA AGARRA LA PERILLA
    for valor in range(round(salida[0]) + 25, X_NUEVA + 1, 25):
        w.SLIDER_X.setValue(valor)
    a_medio_camino(w, PASOS_POR_REPINTADO_JOG)
    assert defectos_de_lo_pintado(w, desde=con_el_eje(salida, 0, X_NUEVA)) == [], "con la perilla agarrada"
    w.SLIDER_X.setValue(X_NUEVA - 100)
    w.SLIDER_X.setSliderDown(False)                       # Y LA SUELTA
    assert w.cola, "control: el movimiento se acabó"
    assert defectos_de_lo_pintado(w, desde=con_el_eje(salida, 0, X_NUEVA - 100)) == [], "al soltar"


@pytest.mark.parametrize("clave", MODOS)
def test_j06_arrastrando_sin_parar_el_rojo_de_los_otros_sliders_no_espera_a_que_se_suelte(ventana, robot, clave):
    """Un ratón manda órdenes más seguido de lo que el robot da pasos. El rojo de los otros dos sliders se pone al día
    cada pocos pasos aunque la perilla no deje de moverse (un contador que cada orden regresa a cero no llega nunca)."""
    w = ventana
    salida = en_base(w, robot, clave)
    w.SLIDER_X.setSliderDown(True)
    repintados, antes, vueltas = 0, tramos_de(w)[1:], 12
    for vuelta in range(1, vueltas + 1):
        w.SLIDER_X.setValue(round(salida[0]) + 20 * vuelta)   # UNA ORDEN…
        correr_pasos(w, 1)                                    # …Y UN PASO DEL ROBOT, SIN SOLTAR
        ahora = tramos_de(w)[1:]
        repintados += ahora != antes
        antes = ahora
    assert w.cola and w.SLIDER_X.isSliderDown(), "control: el arrastre se acabó"
    assert repintados >= vueltas // PASOS_POR_REPINTADO_JOG, f"el rojo de Y y Z solo cambió {repintados} veces en {vueltas} pasos"


def test_j06_al_cambiar_de_modo_con_la_punta_en_camino_el_rojo_cambia_con_el(ventana, robot):
    w = ventana
    salida = en_base(w, robot, "en_cola")
    w.SLIDER_X.setValue(X_NUEVA)
    a_medio_camino(w, 7)
    modo(w, "cancelar")
    assert defectos_de_lo_pintado(w) == []                                             # DESDE DONDE VA LA PUNTA
    modo(w, "reemplazar")
    assert defectos_de_lo_pintado(w, desde=con_el_eje(salida, 0, X_NUEVA)) == []       # DESDE EL PUNTO AL QUE VA


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% J7: DETENER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@pytest.mark.parametrize("clave", MODOS)
def test_j07_detener_borra_lo_acumulado_y_los_sliders_y_el_rojo_quedan_donde_se_detuvo(ventana, robot, clave):
    w = ventana
    en_base(w, robot, clave)
    w.SLIDER_X.setValue(X_NUEVA)
    a_medio_camino(w, 4)
    w.SLIDER_Y.setValue(Y_NUEVA)
    a_medio_camino(w, 3)
    w.detener()
    aqui = tuple(w.posicion)
    assert not w.cola and not w.BTN_DETENER.isEnabled()
    assert donde_marcan(w) == [round(c) for c in aqui]
    assert defectos_de_lo_pintado(w) == []
    assert len([e for e in eventos(w, "aviso") if "Detenido en" in e.mensaje]) == 1
    # LO SIGUIENTE SALE DE DONDE SE DETUVO: DE LO ACUMULADO NO QUEDA NADA
    final = con_el_eje(aqui, 2, round(aqui[2]) - 100)
    w.SLIDER_Z.setValue(round(aqui[2]) - 100)
    camino = recorrer(w)
    assert camino[-1] == pytest.approx(final, abs=0.01)
    assert fuera_del_camino(camino, aqui, final) == []


def test_j07_un_movimiento_de_servos_olvida_lo_pendiente_de_la_punta(ventana, robot):
    """En «Servos» no cambia nada: mover un servo reemplaza lo que faltaba, también lo acumulado en «En cola»."""
    w = ventana
    en_base(w, robot, "en_cola")
    w.SLIDER_X.setValue(X_NUEVA)
    a_medio_camino(w, 4)
    w.SLIDER_Y.setValue(Y_NUEVA)
    w.SLIDER_1.setValue(w.SLIDER_1.value() + 20)
    recorrer(w)
    aqui = tuple(w.posicion)
    assert w.servos[0] == pytest.approx(w.SLIDER_1.value()) and donde_marcan(w) == [round(c) for c in aqui]
    assert defectos_de_lo_pintado(w) == []
    final = con_el_eje(aqui, 2, round(aqui[2]) - 60)
    w.SLIDER_Z.setValue(round(aqui[2]) - 60)
    assert recorrer(w)[-1] == pytest.approx(final, abs=0.01)


@pytest.mark.parametrize("clave", ["reemplazar", "en_cola"])
def test_j07_con_un_servo_en_camino_la_siguiente_orden_de_la_punta_sale_de_donde_va(ventana, robot, clave):
    """Lo pendiente de la punta se olvida en cuanto se mueve un servo, no hasta que el servo termina: una orden nueva de
    la punta se mide desde donde va, y no desde el punto al que iba antes de tocar el servo."""
    w = ventana
    en_base(w, robot, clave)
    w.SLIDER_X.setValue(X_NUEVA)
    a_medio_camino(w, 4)
    w.SLIDER_Y.setValue(Y_NUEVA)
    w.SLIDER_1.setValue(w.SLIDER_1.value() + 60)               # 60°: CINCO PASOS (20° SE ACABAN EN DOS)
    aqui = a_medio_camino(w, 2)                                # EL SERVO VA EN CAMINO
    assert math.dist(aqui[:2], (X_NUEVA, Y_NUEVA)) > 50, "control: la punta ya estaba donde iba; la prueba no distingue nada"
    final = con_el_eje(aqui, 2, round(aqui[2]) - 60)
    w.SLIDER_Z.setValue(round(aqui[2]) - 60)
    assert negativas_del_planeador(w) == [], "control: el planeador negó la orden; hay que elegir otra"
    camino = recorrer(w)
    assert camino[-1] == pytest.approx(final, abs=0.01)
    assert fuera_del_camino(camino, aqui, final) == []
