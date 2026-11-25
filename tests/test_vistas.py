"""Las vistas a la vez (pestañas «Todas · 3D · Lateral · Superior · Ficha») y la columna ACCIONES + el registro en la banda de abajo.

Cada prueba lleva en su nombre el número del criterio que comprueba (F10.n del visor, R1–R5 del registro y la columna).
La ventana es la de la app (QT_QPA_PLATFORM=offscreen): la vista 3D en vivo no se dibuja sin pantalla, así que lo que toca a
su imagen se comprueba con la MISMA escena (clase Escena) en un lienzo fuera de pantalla.

TOPES DE VENTANA: medidos en la base de esta rama (commit acb95ff) con tests/medir_minimo (ver el reporte): sin el editor de
puntos, 1243×645; con él, 1271×645. Ninguna pestaña ni columna puede pasar de ahí.
"""
import itertools
import os
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import pyvista as pv
import vtk
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets

import auditoria_interfaz as au
from cosechador import registro as reg
from cosechador.robot.cinematica import servo_a_dh
from cosechador.robot.espacio_trabajo import contorno_del_corte, giro_de_la_base
from cosechador.ui import ficha, planos, visor as modulo_visor
from cosechador.ui.escena3d import Escena
from cosechador.ui.ventana import PESTANA_PROGRAMA, PESTANA_SERVOS, PESTANA_XYZ, VentanaPrincipal
from test_editor_puntos import _tres, abrir, home_s, poses  # noqa: F401 (poses y home_s SON FIXTURES)
from test_grabacion_ventana import correr, ventana  # noqa: F401 (ventana ES EL FIXTURE)

Qt, QTest = QtCore.Qt, QtTest.QTest
NOMBRES = ("Todas", "3D", "Lateral", "Superior", "Ficha")
TODAS, V3D, LATERAL, SUPERIOR, FICHA = range(5)
TOPE_SIN_EDITOR, TOPE_CON_EDITOR = (1243, 645), (1271, 645)     # MEDIDOS EN acb95ff
TAMANOS = [(1280, 720), (1440, 860), (1512, 863)]
LETRA_MINIMA = 12                                                # px: LA DECLARA EL AUDITOR (au.LETRA_MINIMA)
HUECO_MAXIMO = modulo_visor.SEPARACION                           # px ENTRE VISTAS; DECLARADO


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% AYUDAS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@pytest.fixture(scope="module")
def lazos(robot):
    return contorno_del_corte(robot)


def pasar(n=6):
    for _ in range(n):
        QtWidgets.QApplication.processEvents()


def estado(w, tam=(1280, 720), pestana=TODAS, vision=True, acciones=True, volumen=False, lazos=None, movimiento=PESTANA_SERVOS):
    """Deja la ventana en un estado conocido (sin editor de puntos)."""
    w.cerrar_editor()
    w.show()                                                     # OFFSCREEN: NO SE ABRE NADA A LA VISTA
    w.resize(*tam)
    w.contraer_vision(not vision, recordar=False)
    w.contraer_acciones(not acciones, recordar=False)
    w.PESTANAS_MOVIMIENTO.setCurrentIndex(movimiento)
    w.visor.ver(pestana)
    if volumen:
        w.visor.poner_volumen(lazos, True)
    else:
        w.visor.ocultar_volumen()
    pasar()


def rect(w, widget):
    """Rectángulo de `widget` en coordenadas del visor."""
    return QtCore.QRect(widget.mapTo(w.visor, QtCore.QPoint(0, 0)), widget.size())


def rect_en(w, widget):
    """Rectángulo de `widget` en coordenadas de la ventana."""
    return QtCore.QRect(widget.mapTo(w, QtCore.QPoint(0, 0)), widget.size())


def a_la_vista(w):
    return [v for v in w.visor.vistas if v.isVisible()]


def minimo(w):
    m = w.minimumSizeHint()
    return m.width(), m.height()


def clic(widget, punto=None):
    QTest.mouseClick(widget, Qt.LeftButton, Qt.NoModifier, punto if punto is not None else widget.rect().center())
    pasar()


def clic_en_pestana(w, indice):
    barra = w.visor.pestanas
    QTest.mouseClick(barra, Qt.LeftButton, Qt.NoModifier, barra.tabRect(indice).center())
    pasar()


def variante(robot, limites=None, medidas=None):
    """Una copia del robot con otros límites de servo ({articulación: (mínimo, máximo)}) u otras medidas ({articulación: a})."""
    articulaciones = []
    for i, j in enumerate(robot.articulaciones):
        if limites and i in limites:
            j = replace(j, servo=replace(j.servo, minimo=limites[i][0], maximo=limites[i][1]))
        if medidas and i in medidas:
            j = replace(j, a=medidas[i])
        articulaciones.append(j)
    return replace(robot, articulaciones=tuple(articulaciones))


def textos(dibujo):
    return [r.texto for r in dibujo.rotulos()]


def solo(dibujo, ancho=620, alto=400, volumen=True):
    """Un dibujo suelto (sin ventana) del tamaño pedido, con o sin volumen."""
    dibujo.fig.set_size_inches(ancho / 100, alto / 100)
    dibujo.con_titulo, dibujo.volumen_visible = True, volumen
    dibujo.componer()
    dibujo.lienzo.draw()
    return dibujo


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% F10.1 · LAS CINCO PESTAÑAS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_f10_1_hay_cinco_pestanas_sobre_el_visor_en_ese_orden(ventana):
    w = ventana
    barra = w.visor.pestanas
    assert [barra.tabText(i) for i in range(barra.count())] == list(NOMBRES)
    estado(w)
    assert rect(w, barra).bottom() <= rect(w, w.visor.mosaico).top() + 1     # ESTÁN SOBRE EL VISOR, NO A UN LADO
    assert w.visor.principal == TODAS                                        # LA PRIMERA VEZ ABRE EN «Todas»


@pytest.mark.parametrize("pestana", [V3D, LATERAL, SUPERIOR, FICHA])
def test_f10_1_la_app_abre_en_la_ultima_pestana_usada(ventana, robot, pestana):
    w = ventana
    estado(w)
    w.visor.ver(pestana)
    w.ajustes.sync()
    ajustes = QtCore.QSettings(w.ajustes.fileName(), QtCore.QSettings.IniFormat)
    otra = VentanaPrincipal(robot, ajustes=ajustes, carpeta_programas=w.carpeta_programas, registro=reg.Registro(carpeta=None))
    try:
        assert otra.visor.principal == pestana
    finally:
        otra.close()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% F10.2 · «TODAS»: LA DISPOSICIÓN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@pytest.mark.parametrize("tam", TAMANOS)
def test_f10_2_en_todas_la_3d_va_a_la_izquierda_y_es_mas_grande_y_las_otras_dos_a_la_derecha_una_sobre_otra(ventana, tam):
    w = ventana
    estado(w, tam)
    v3d, lat, sup = (rect(w, v) for v in (w.visor.v3d, w.visor.lateral.lienzo, w.visor.superior.lienzo))
    assert v3d.right() < lat.left() and v3d.right() < sup.left()              # LA 3D, A LA IZQUIERDA DE LAS OTRAS DOS
    assert lat.bottom() < sup.top()                                           # LA LATERAL, ARRIBA DE LA SUPERIOR
    assert abs(lat.left() - sup.left()) <= 1 and abs(lat.width() - sup.width()) <= 1   # UNA SOBRE OTRA: MISMA COLUMNA
    area = lambda r: r.width() * r.height()
    assert area(v3d) > area(lat) and area(v3d) > area(sup)                    # LA 3D, LA DE MAYOR ÁREA


@pytest.mark.parametrize("tam", TAMANOS)
def test_f10_2_en_todas_las_tres_vistas_llenan_el_visor_con_huecos_de_a_lo_mas_4_px(ventana, tam):
    w = ventana
    estado(w, tam)
    zona = rect(w, w.visor.mosaico)
    v3d, lat, sup = (rect(w, v) for v in (w.visor.v3d, w.visor.lateral.lienzo, w.visor.superior.lienzo))
    huecos = {"3D al borde izquierdo": v3d.left() - zona.left(), "3D al borde de arriba": v3d.top() - zona.top(),
              "3D al borde de abajo": zona.bottom() - v3d.bottom(), "entre la 3D y la columna": lat.left() - v3d.right() - 1,
              "columna al borde derecho": zona.right() - lat.right(), "lateral al borde de arriba": lat.top() - zona.top(),
              "entre lateral y superior": sup.top() - lat.bottom() - 1, "superior al borde de abajo": zona.bottom() - sup.bottom()}
    assert max(huecos.values()) <= HUECO_MAXIMO, huecos
    assert min(huecos.values()) >= 0, huecos                                  # NINGUNA SE SALE
    # EL ÁREA QUE QUEDA SIN VISTA, EN PORCENTAJE (LO DEMÁS ES EL GRIS ENTRE VISTAS)
    cubierta = sum(r.width() * r.height() for r in (v3d, lat, sup)) / (zona.width() * zona.height())
    assert cubierta > 0.97, cubierta


def test_f10_2_en_todas_el_visor_de_1280_por_720_mide_lo_que_se_declara(ventana):
    estado(ventana, (1280, 720))
    assert ventana.visor.width() > 560 and ventana.visor.height() > 380        # LO MEDIDO: 624×423 CON TODO ABIERTO


@pytest.mark.parametrize("pestana", [TODAS, V3D])
def test_f10_2_la_3d_conserva_al_menos_520_px_de_alto_a_1440_por_860_en_todas_y_sola(ventana, pestana):
    """El criterio de test_interfaz.py::test_k1 (la 3D, 520 px de alto) medido en la ventana real y con el visor nuevo: allá el
    recuadro está vacío (la fixture no lleva escena), aquí se mide la vista 3D de verdad."""
    estado(ventana, (1440, 860), pestana=pestana)
    assert ventana.visor.v3d.height() >= 520, ventana.visor.v3d.height()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% F10.3 · LAS TRES SE MUEVEN A LA VEZ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def _brazos(w):
    """El trazo del robot en la lateral y en la superior, tal como están dibujados."""
    return [np.array(d.l_brazo.get_xydata(), float) for d in (w.visor.lateral, w.visor.superior)]


def _poses_dibujadas(w):
    return [tuple(d.servos) for d in (w.visor.lateral, w.visor.superior)]


MEDIOS = {
    "slider de un servo": lambda w: (w.SLIDER_2.setValue(w.SLIDER_2.value() + 25), correr(w)),
    "slider de la punta XYZ": lambda w: (w.SLIDER_Z.setValue(w.SLIDER_Z.value() - 120), correr(w)),
    "volver a casa": lambda w: (w._ir_a_home(), correr(w)),
    "poner_servos directo": lambda w: w.poner_servos((60, 75, 115)),
}


@pytest.mark.parametrize("medio", list(MEDIOS))
def test_f10_3_al_mover_el_robot_por_cualquier_medio_las_tres_vistas_se_mueven_a_la_vez(ventana, medio):
    w = ventana
    estado(w)
    w.poner_servos((30, 30, 60))
    correr(w)
    antes_3d = [np.array(a.user_matrix) for a in w.escena.actores]
    antes = _brazos(w)
    MEDIOS[medio](w)
    pasar()
    despues_3d = [np.array(a.user_matrix) for a in w.escena.actores]
    assert any(not np.allclose(a, b) for a, b in zip(antes_3d, despues_3d))          # SE MOVIÓ LA 3D…
    assert all(np.abs(a - b).max() > 1e-6 for a, b in zip(antes, _brazos(w)))          # …Y LA LATERAL Y LA SUPERIOR
    assert _poses_dibujadas(w) == [tuple(w.servos)] * 2                                # LAS TRES EN LA MISMA POSTURA


def test_f10_3_con_una_vista_sola_las_escondidas_salen_al_dia_al_volver_a_todas(ventana):
    w = ventana
    estado(w, pestana=V3D)
    w.poner_servos((100, 50, 80))
    correr(w)
    w.visor.ver(TODAS)
    pasar()
    assert _poses_dibujadas(w) == [tuple(w.servos)] * 2
    esperado = [w.visor.geo.lateral(w.servos)[:, :2], w.visor.geo.superior(w.servos)]
    for d, e in zip((w.visor.lateral, w.visor.superior), esperado):
        assert np.allclose(d.l_brazo.get_xydata(), e[: len(d.l_brazo.get_xydata())], atol=1e-6)


@pytest.mark.parametrize("base", [0, 37, 90, 180, 270])
def test_f10_3_la_lateral_sigue_al_brazo_y_siempre_lo_ve_de_perfil(robot, base):
    geo = planos.Geometria(robot)
    resto = (45, 90)
    de_perfil = geo.lateral((0, *resto))
    girada = geo.lateral((base, *resto))
    assert np.allclose(girada, de_perfil, atol=1e-6)                            # GIRAR LA BASE NO CAMBIA EL DIBUJO LATERAL
    # EL «ALCANCE» DE CADA PUNTO ES SU DISTANCIA REAL AL EJE DE LA BASE (CON SIGNO): EL BRAZO NO SE VE «ACORTADO»
    p = geo.puntos((base, *resto))
    assert np.allclose(np.abs(girada[:, 0]), np.hypot(p[:, 0], p[:, 1]), atol=1e-6)


def test_f10_3_y_la_superior_si_gira_con_la_base(robot):
    geo = planos.Geometria(robot)
    a, b = geo.superior((0, 90, 90)), geo.superior((90, 90, 90))
    assert not np.allclose(a, b)
    assert np.allclose(np.hypot(*a.T), np.hypot(*b.T), atol=1e-6)               # MISMA LONGITUD, OTRA DIRECCIÓN


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% F10.4 / F10.5 · UNA VISTA SOLA Y VOLVER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@pytest.mark.parametrize("pestana", [V3D, LATERAL, SUPERIOR])
def test_f10_4_picar_una_pestana_deja_esa_vista_sola_ocupando_todo_el_visor(ventana, pestana):
    w = ventana
    estado(w)
    clic_en_pestana(w, pestana)
    assert w.visor.principal == pestana
    assert a_la_vista(w) == [w.visor.vistas[pestana - 1]]
    assert rect(w, a_la_vista(w)[0]) == rect(w, w.visor.mosaico)                 # TODO EL VISOR


@pytest.mark.parametrize("vista", range(3))
def test_f10_4_picar_una_vista_del_mosaico_la_deja_sola(ventana, vista):
    w = ventana
    estado(w)
    clic(w.visor.vistas[vista])
    pasar(8)
    assert w.visor.principal == vista + 1
    assert a_la_vista(w) == [w.visor.vistas[vista]]
    assert rect(w, a_la_vista(w)[0]) == rect(w, w.visor.mosaico)


def test_f10_4_arrastrar_sobre_la_3d_sigue_girando_la_camara_y_no_la_agranda(ventana, monkeypatch):
    """OFFSCREEN la cámara en vivo no se mueve (no hay ventana gráfica), así que se prueba lo que sí se puede ver: que la 3D
    RECIBE el arrastre completo (el visor no se queda con los eventos) y que el arrastre no la agranda."""
    w = ventana
    estado(w)
    v3d = w.visor.v3d
    recibidos = []
    for nombre in ("mousePressEvent", "mouseMoveEvent", "mouseReleaseEvent"):
        original = getattr(type(v3d), nombre)
        monkeypatch.setattr(type(v3d), nombre, lambda self, e, o=original, n=nombre: (recibidos.append(n), o(self, e))[1])
    centro = v3d.rect().center()
    QTest.mousePress(v3d, Qt.LeftButton, Qt.NoModifier, centro)
    for paso in range(1, 7):                      # MOVIMIENTOS CON EL BOTÓN IZQUIERDO APRETADO (QTest.mouseMove NO LO APRIETA)
        punto = QtCore.QPointF(centro + QtCore.QPoint(8 * paso, 4 * paso))
        QtWidgets.QApplication.sendEvent(v3d, QtGui.QMouseEvent(QtCore.QEvent.MouseMove, punto, Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
        pasar(1)
    QTest.mouseRelease(v3d, Qt.LeftButton, Qt.NoModifier, centro + QtCore.QPoint(48, 24))
    pasar(8)
    assert w.visor.principal == TODAS and len(a_la_vista(w)) == 3
    assert recibidos.count("mousePressEvent") == 1 and recibidos.count("mouseReleaseEvent") == 1
    assert recibidos.count("mouseMoveEvent") >= 6                                      # LA 3D RECIBIÓ TODO EL ARRASTRE


@pytest.mark.parametrize("pestana", [V3D, LATERAL, SUPERIOR])
def test_f10_5_picar_otra_vez_la_pestana_activa_regresa_a_las_tres(ventana, pestana):
    w = ventana
    estado(w)
    clic_en_pestana(w, pestana)
    clic_en_pestana(w, pestana)
    assert w.visor.principal == TODAS and len(a_la_vista(w)) == 3


@pytest.mark.parametrize("pestana", [V3D, LATERAL, SUPERIOR])
def test_f10_5_picar_todas_regresa_a_las_tres(ventana, pestana):
    w = ventana
    estado(w, pestana=pestana)
    clic_en_pestana(w, TODAS)
    assert w.visor.principal == TODAS and len(a_la_vista(w)) == 3


def test_f10_5_picar_una_vista_ya_sola_no_hace_nada_y_la_secuencia_completa_termina_donde_debe(ventana):
    w = ventana
    estado(w)
    secuencia, esperado = [2, 2, 2, 3, 3, 0, 1, 1], [2, 0, 2, 3, 0, 0, 1, 0]
    visto = []
    for indice in secuencia:
        clic_en_pestana(w, indice)
        visto.append(w.visor.principal)
    assert visto == esperado


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% F10.6 / F10.7 · LO QUE DIBUJAN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_f10_6_con_volumen_la_lateral_enseña_contorno_cuatro_medidas_y_el_gris_bajo_la_base(robot, lazos):
    d = solo(planos.Lateral(planos.Geometria(robot, lazos)))
    dicho = " | ".join(textos(d))
    for medida, valor in (("al frente", 1820), ("hacia atrás", 1710), ("arriba", 2170), ("abajo", 470)):
        assert f"{medida} {valor}" in dicho, (medida, valor, dicho)
    gris = [p for p in d.ax.patches if p.get_facecolor()[:3] == pytest.approx(
        tuple(int(planos.GRIS_BAJO_LA_BASE[i:i + 2], 16) / 255 for i in (1, 3, 5)), abs=1e-3)]
    assert gris, "no hay relleno gris bajo la base"
    assert any(ln.get_xydata().shape[0] > 20 for ln in d.ax.lines + [p for p in d.ax.patches if hasattr(p, "get_path")] if hasattr(ln, "get_xydata")) or d.ax.patches


def test_f10_6_con_volumen_la_superior_enseña_el_giro_de_la_base(robot, lazos):
    d = solo(planos.Superior(planos.Geometria(robot, lazos)))
    dicho = " | ".join(textos(d))
    assert "giro de la base 180°" in dicho and "R 1820" in dicho and "R 1710" in dicho, dicho


@pytest.mark.parametrize("clase", [planos.Lateral, planos.Superior])
def test_f10_6_sin_volumen_quedan_robot_nivel_de_la_base_y_ejes_sin_ninguna_medida_del_volumen(robot, lazos, clase):
    d = solo(clase(planos.Geometria(robot, lazos)), volumen=False)
    dicho = " | ".join(textos(d))
    assert not any(palabra in dicho for palabra in ("al frente", "hacia atrás", "arriba 2170", "abajo", "giro de la base", "R 1"))
    assert len(d.l_brazo.get_xydata()) == robot.gdl + 1 and len(d.p_punta.get_xydata()) == 1   # EL ROBOT ESTÁ (BASE, ARTICULACIONES Y PUNTA)
    assert any("(mm)" in t for t in textos(d))                                    # LOS EJES, CON SUS UNIDADES
    if clase is planos.Lateral:
        assert "nivel de la base" in dicho
    else:                                                                         # EL CÍRCULO TENUE DEL ALCANCE
        assert any(type(p).__name__ in ("Circle", "Wedge", "Arc") for p in d.ax.patches)


def test_f10_7_si_cambian_los_limites_o_las_medidas_del_robot_los_dibujos_y_sus_medidas_cambian_solos(ventana, robot):
    w = ventana
    estado(w, pestana=LATERAL, volumen=True, lazos=contorno_del_corte(robot))
    antes = textos(w.visor.lateral)
    otro = variante(robot, medidas={2: 500.0})                                      # EL ANTEBRAZO, 200 mm MÁS CORTO
    nuevos = contorno_del_corte(otro)
    w.visor.robot = otro
    w.visor.poner_volumen(nuevos, True)
    pasar()
    despues = textos(w.visor.lateral)
    assert antes != despues
    nueva_geo = planos.Geometria(otro, nuevos)
    assert any(f"al frente {round(nueva_geo.frente)}" in t for t in despues)
    # LA FICHA TOMA LOS MISMOS NÚMEROS
    valores = dict(ficha.datos_de_la_ficha(nueva_geo)[1])
    assert valores["Alcance al frente"] == f"{round(nueva_geo.frente)} mm"
    w.visor.robot = robot
    w.visor.poner_volumen(contorno_del_corte(robot), False)


@pytest.mark.parametrize("giro", [0, 37, 90, 180, 270, 360])
def test_f10_7_la_superior_vale_para_cualquier_giro_de_la_base(robot, lazos, giro):
    geo = planos.Geometria(robot, lazos)
    geo.desde, geo.hasta = 0.0, float(giro)
    d = solo(planos.Superior(geo, servos=(60, 75, 115)))
    # EL CONTORNO: EN CADA DIRECCIÓN DEL FRENTE LLEGA A «frente», EN LA OPUESTA A «atrás», Y FUERA DE LOS SECTORES A 0
    for fi in range(0, 360, 5):
        dentro_frente = giro >= 360 or (fi % 360) <= giro
        dentro_atras = giro >= 360 or ((fi - 180) % 360) <= giro
        esperado = max(geo.frente if dentro_frente else 0, geo.atras if dentro_atras else 0)
        assert geo.radio(fi) == pytest.approx(esperado)
    cubierto = sum(q - p for p, q, r in geo.tramos() if r > 0)                      # LOS GRADOS QUE EL VOLUMEN CUBRE VISTO DESDE ARRIBA
    assert cubierto == pytest.approx(360 if giro >= 180 else 2 * giro, abs=1e-6)
    assert au.rotulos_de(d) == []
    if 0 < giro < 360:
        assert any("giro de la base" in t for t in textos(d))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% F10.8 · «NÚMEROS» %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_f10_8_el_boton_numeros_esta_junto_a_trayectoria_y_empieza_encendido(ventana):
    w = ventana
    estado(w)
    t, n = rect_en(w, w.CHK_TRAYECTORIA), rect_en(w, w.CHK_NUMEROS_REJILLA)
    assert w.CHK_NUMEROS_REJILLA.text() == "Números" and w.CHK_NUMEROS_REJILLA.isChecked()
    assert n.left() > t.right() and abs(n.center().y() - t.center().y()) <= 1
    assert n.left() - t.right() <= 24                                               # «JUNTO A»: SIN OTRO CONTROL ENTRE LOS DOS


def _rejilla_con_medidas(w):
    return w.escena.plotter.renderer.cube_axes_actor


def _piso(w):
    """El piso de la escena: la malla de alambre gris (no la rejilla de medidas)."""
    return [a for a in w.escena.plotter.renderer.actors.values()
            if isinstance(a, vtk.vtkActor) and a.GetProperty().GetRepresentation() == 1 and abs(a.GetProperty().GetOpacity() - 0.3) < 1e-6]


def test_f10_8_apagado_se_van_numeros_nombres_de_ejes_y_caja_y_queda_la_rejilla_del_piso(ventana, lazos):
    w = ventana
    estado(w, volumen=True, lazos=lazos)
    w.escena.mostrar_superficie(lazos, giro_de_la_base(w.robot))
    w.CHK_NUMEROS_REJILLA.setChecked(True)
    assert _rejilla_con_medidas(w) is not None and _piso(w)
    w.CHK_NUMEROS_REJILLA.setChecked(False)
    assert _rejilla_con_medidas(w) is None and w.escena.numeros_visibles is False   # SE FUERON NÚMEROS, NOMBRES Y CAJA
    assert _piso(w)                                                                  # LA REJILLA DEL PISO SIGUE
    w.CHK_NUMEROS_REJILLA.setChecked(True)
    assert _rejilla_con_medidas(w) is not None
    w.escena.ocultar_volumen()


def test_f10_8_apagado_o_encendido_se_recuerda_al_reabrir(ventana, robot):
    w = ventana
    estado(w)
    w.CHK_NUMEROS_REJILLA.setChecked(False)
    w.ajustes.sync()
    ajustes = QtCore.QSettings(w.ajustes.fileName(), QtCore.QSettings.IniFormat)
    otra = VentanaPrincipal(robot, ajustes=ajustes, carpeta_programas=w.carpeta_programas, registro=reg.Registro(carpeta=None))
    try:
        assert otra.CHK_NUMEROS_REJILLA.isChecked() is False and otra.escena.numeros_visibles is False
    finally:
        otra.close()
        w.CHK_NUMEROS_REJILLA.setChecked(True)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% F10.9 · BOTONES DE CÁMARA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@pytest.mark.parametrize("pestana", range(5))
def test_f10_9_los_botones_de_camara_solo_estan_encendidos_con_la_3d_a_la_vista(ventana, pestana):
    w = ventana
    estado(w, pestana=pestana)
    botones = (w.BTN_VISTA_ISO, w.BTN_VISTA_ARRIBA, w.BTN_VISTA_FRENTE, w.BTN_VISTA_LADO, w.BTN_ENCUADRAR)
    assert [b.isEnabled() for b in botones] == [pestana in (TODAS, V3D)] * 5
    estado(w)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% F10.10 / R5 · EL TAMAÑO MÍNIMO NO CRECE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@pytest.mark.parametrize("pestana", range(5))
@pytest.mark.parametrize("acciones", [True, False])
def test_f10_10_las_pestanas_no_agrandan_el_minimo_de_la_ventana(ventana, pestana, acciones):
    w = ventana
    estado(w, pestana=pestana, acciones=acciones)
    ancho, alto = minimo(w)
    assert ancho <= TOPE_SIN_EDITOR[0] and alto <= TOPE_SIN_EDITOR[1], (NOMBRES[pestana], acciones, (ancho, alto))


@pytest.mark.parametrize("pestana", range(5))
def test_r5_el_minimo_de_la_ventana_no_crece_tampoco_con_el_editor_de_puntos_abierto(ventana, poses, pestana):
    w = ventana
    estado(w, pestana=pestana)
    w.PESTANAS_MOVIMIENTO.setCurrentIndex(PESTANA_PROGRAMA)
    abrir(w, "p.yaml", *_tres(poses))
    pasar()
    ancho, alto = minimo(w)
    assert ancho <= TOPE_CON_EDITOR[0] and alto <= TOPE_CON_EDITOR[1], (NOMBRES[pestana], (ancho, alto))
    w.cerrar_editor()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% F10.11 · LA FICHA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@pytest.mark.parametrize("tam", TAMANOS)
def test_f10_11_la_ficha_enseña_lateral_y_superior_juntas_la_tabla_debajo_en_dos_columnas_y_la_3d_no_se_repite(ventana, lazos, tam):
    w = ventana
    estado(w, tam, pestana=FICHA, volumen=True, lazos=lazos)
    pagina = w.visor.ficha
    lat, sup, tabla = (rect(w, x) for x in (pagina.lateral.lienzo, pagina.superior.lienzo, pagina.tabla))
    assert lat.right() < sup.left() and abs(lat.center().y() - sup.center().y()) <= 1       # LADO A LADO
    assert tabla.top() > max(lat.bottom(), sup.bottom())                                      # TABLA DEBAJO
    assert not w.visor.v3d.isVisible()                                                        # LA 3D NO SE REPITE EN PANTALLA
    columnas = {rect(w, e).left() for e in pagina.tabla.findChildren(QtWidgets.QLabel) if e.property("rol") == "nota" and e.isVisible()}
    assert len(columnas) == 2, columnas                                                       # DOS COLUMNAS DE ETIQUETAS


@pytest.mark.parametrize("tam", TAMANOS)
@pytest.mark.parametrize("volumen", [True, False])
def test_f10_11_la_tabla_va_en_letra_de_12_px_y_ningun_valor_se_parte_ni_se_corta(ventana, lazos, tam, volumen):
    w = ventana
    estado(w, tam, pestana=FICHA, volumen=volumen, lazos=lazos)
    etiquetas = [e for e in w.visor.ficha.tabla.findChildren(QtWidgets.QLabel) if e.isVisible() and e.property("rol") in ("nota", "dato")]
    assert len(etiquetas) >= 10
    for e in etiquetas:
        assert QtGui.QFontInfo(e.font()).pixelSize() == 12, (e.text(), QtGui.QFontInfo(e.font()).pixelSize())
        assert "\n" not in e.text() and (not e.wordWrap() or e.text() == ficha.SIN_VOLUMEN)
        if not e.wordWrap():
            assert e.sizeHint().width() <= e.width() + 1, (e.text(), e.sizeHint().width(), e.width())   # NO SALE CORTADO
            assert e.height() <= e.fontMetrics().height() + 10, (e.text(), e.height())                    # UN SOLO RENGLÓN


def test_f10_11_el_boton_guardar_ficha_esta_a_la_derecha_de_las_pestanas_y_solo_se_ve_en_la_ficha(ventana):
    w = ventana
    for pestana in range(5):
        estado(w, pestana=pestana)
        assert w.visor.btn_guardar_ficha.isVisible() == (pestana == FICHA)
    estado(w, pestana=FICHA)
    boton, barra = rect(w, w.visor.btn_guardar_ficha), rect(w, w.visor.pestanas)
    assert w.visor.btn_guardar_ficha.text() == "Guardar ficha…"
    assert boton.left() > barra.right() and boton.top() < barra.bottom() and boton.bottom() > barra.top()   # MISMA FRANJA, A LA DERECHA
    assert boton.right() <= rect(w, w.visor).right()                                                        # DENTRO DEL VISOR


def test_f10_11_guardar_ficha_escribe_la_hoja_completa_de_1600_por_1000_con_la_3d(ventana, lazos, tmp_path):
    w = ventana
    estado(w, pestana=FICHA, volumen=True, lazos=lazos)
    ruta = str(tmp_path / "ficha.png")
    assert w.visor.guardar_ficha(ruta) == ruta and os.path.exists(ruta)
    imagen = QtGui.QImage(ruta)
    assert (imagen.width(), imagen.height()) == (1600, 1000)
    # LA 3D (RECUADRO DE LA HOJA) NO SALE VACÍA: HAY MUCHOS MÁS COLORES QUE EL FONDO
    zona = imagen.copy(40, 678, 430, 290).convertToFormat(QtGui.QImage.Format_RGB888)
    bits = zona.constBits()
    bits.setsize(zona.height() * zona.bytesPerLine())
    pix = np.frombuffer(bits, np.uint8).reshape(zona.height(), zona.bytesPerLine())[:, :zona.width() * 3].reshape(-1, 3)
    fondo = pix[0]
    assert (np.abs(pix.astype(int) - fondo.astype(int)).sum(axis=1) > 30).mean() > 0.05


def test_f10_11_la_ficha_guardada_y_la_de_pantalla_dicen_lo_mismo(ventana, lazos):
    w = ventana
    estado(w, pestana=FICHA, volumen=True, lazos=lazos)
    construccion, alcance = ficha.datos_de_la_ficha(w.visor.geo)
    en_pantalla = [e.text() for e in w.visor.ficha.tabla.findChildren(QtWidgets.QLabel) if e.property("rol") in ("nota", "dato")]
    for nombre, valor in construccion + alcance:
        assert nombre in en_pantalla and valor in en_pantalla


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LA 3D: SIN OJOS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def _imagen_y_cuadro(robot, servos, lazos, ancho, alto):
    """La escena 3D en un lienzo fuera de pantalla del tamaño dado: (¿qué fracción de la imagen no es fondo?, caja del robot en px)."""
    pl = pv.Plotter(off_screen=True, window_size=(ancho, alto))
    try:
        escena = Escena(pl, robot)
        escena.dibujar(servo_a_dh(robot, servos))
        if lazos:
            escena.mostrar_superficie(lazos, giro_de_la_base(robot))
            escena.encuadrar(zoom=1.0)
        else:
            escena.encuadrar()
        imagen = pl.screenshot(return_img=True)
        tinta = float((np.abs(imagen.astype(int) - imagen[0, 0].astype(int)).sum(axis=2) > 30).mean())
        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()
        xs, ys = [], []
        for actor in escena.actores:
            m, b = np.array(actor.user_matrix), actor.mapper.dataset.bounds
            for x, y, z in itertools.product(b[0:2], b[2:4], b[4:6]):
                coord.SetValue(*(m @ np.array([x, y, z, 1.0]))[:3])
                d = coord.GetComputedDoubleDisplayValue(pl.renderer)
                xs.append(d[0])
                ys.append(d[1])
        return tinta, (min(xs), max(xs), min(ys), max(ys))
    finally:
        pl.close()


@pytest.mark.parametrize("pestana", [TODAS, V3D])
@pytest.mark.parametrize("volumen", [False, True])
@pytest.mark.parametrize("servos", [(0, 20, 0), (90, 100, 90), (180, 180, 180)])
def test_f10_2_la_3d_no_sale_vacia_y_el_robot_cae_completo_en_el_cuadro_en_todas_y_en_3d(ventana, robot, lazos, pestana, volumen, servos):
    w = ventana
    estado(w, pestana=pestana)
    ancho, alto = w.visor.v3d.width(), w.visor.v3d.height()                 # EL TAMAÑO QUE LA 3D TIENE EN ESA PESTAÑA, A 1280×720
    tinta, (x0, x1, y0, y1) = _imagen_y_cuadro(robot, servos, lazos if volumen else None, ancho, alto)
    assert tinta > 0.02, f"la imagen sale casi vacía ({tinta:.3f}) a {ancho}×{alto}"
    assert x0 >= 0 and y0 >= 0 and x1 <= ancho and y1 <= alto, (x0, x1, y0, y1, ancho, alto)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% R1 / R2 · EL REGISTRO EN LA BANDA DE ABAJO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def _elegir_evento(w, texto):
    """Elige en la lista el renglón de ese evento (el texto está en el aviso emergente: el renglón solo lleva el mensaje)."""
    lista = w.LISTA_REGISTRO
    fila = next(i for i in range(lista.count()) if texto in lista.item(i).toolTip() or texto in lista.item(i).text())
    lista.setCurrentRow(fila)
    pasar()


def _llenar_registro(w, n=12):
    for i in range(n):
        w.avisar(f"Evento {i}: " + "un mensaje bastante largo " * 6, "aviso" if i % 2 else "info", "Cinemática inversa")
    pasar()


@pytest.mark.parametrize("tam", TAMANOS)
def test_r1_el_registro_esta_a_la_derecha_de_los_sliders_y_con_su_mismo_alto(ventana, tam):
    w = ventana
    estado(w, tam)
    r, tabs = w.FONDO_REGISTRO.geometry(), w.PESTANAS_MOVIMIENTO.geometry()
    ultimo = max(w.SLIDER_VELOCIDAD.geometry().bottom(), w.SELECTOR_POSTURA.geometry().bottom())
    sliders = rect_en(w, w.PESTANAS_MOVIMIENTO)
    assert w.FONDO_REGISTRO.mapTo(w, QtCore.QPoint(0, 0)).x() > sliders.right()               # A SU DERECHA
    arriba_r = w.FONDO_REGISTRO.mapTo(w, QtCore.QPoint(0, 0)).y()
    abajo_r = arriba_r + w.FONDO_REGISTRO.height()
    abajo_s = w.SELECTOR_POSTURA.mapTo(w, QtCore.QPoint(0, w.SELECTOR_POSTURA.height())).y()
    assert abs(arriba_r - sliders.top()) <= 1 and abs(abajo_r - abajo_s) <= 6, (arriba_r, sliders.top(), abajo_r, abajo_s)  # MISMO ALTO


def test_r1_lista_a_la_izquierda_detalle_a_la_derecha_y_filtro_copiar_y_exportar_a_un_clic(ventana):
    w = ventana
    estado(w)
    lista, detalle = rect_en(w, w.LISTA_REGISTRO), rect_en(w, w.DETALLE_REGISTRO)
    assert lista.right() < detalle.left()
    zona = rect_en(w, w.FONDO_REGISTRO)
    for control in (w.FILTRO_REGISTRO, w.BTN_COPIAR_REGISTRO, w.BTN_EXPORTAR_REGISTRO):
        assert control.isVisible() and control.isEnabled() and zona.contains(rect_en(w, control)), control.objectName()


def test_r1_copiar_y_exportar_responden_a_un_solo_clic(ventana, monkeypatch, tmp_path):
    w = ventana
    estado(w)
    _llenar_registro(w, 3)
    copiado = []
    monkeypatch.setattr(QtWidgets.QApplication.clipboard().__class__, "setText", lambda self, t, *a: copiado.append(t))
    _elegir_evento(w, "Evento 1")
    clic(w.BTN_COPIAR_REGISTRO)
    assert copiado and "Evento 1:" in copiado[-1]


@pytest.mark.parametrize("tam", [(1280, 720)])
def test_r2_cada_evento_ocupa_un_renglon_y_caben_cinco_a_la_vista(ventana, tam):
    w = ventana
    estado(w, tam)
    _llenar_registro(w, 12)
    w.avisar("Corto", "info", "Cinemática inversa")                          # UN MENSAJE CORTO Y OTROS LARGOS: SI ALGUNO ENVUELVE, LAS ALTURAS DIFIEREN
    w.avisar("Primera línea\nsegunda línea", "aviso", "Cinemática inversa")   # UN MENSAJE DE VARIAS LÍNEAS SIGUE SIENDO UN RENGLÓN
    pasar()
    lista = w.LISTA_REGISTRO
    alturas = {lista.sizeHintForRow(i) for i in range(lista.count())}
    assert len(alturas) == 1, alturas                                          # UN MENSAJE LARGO NO CRECE: TODOS MIDEN LO MISMO
    assert all("\n" not in lista.item(i).text() for i in range(lista.count()))  # UN RENGLÓN DE TEXTO
    visibles = lista.viewport().height() // lista.sizeHintForRow(0)
    assert visibles >= 5, (visibles, lista.viewport().height(), lista.sizeHintForRow(0))


def test_r2_el_texto_completo_se_lee_en_el_detalle(ventana):
    w = ventana
    estado(w)
    _llenar_registro(w, 4)
    _elegir_evento(w, "Evento 1")
    texto = w.DETALLE_REGISTRO.toPlainText()
    assert "Evento 1:" in texto
    assert texto.count("un mensaje bastante largo") == 6                        # EL MENSAJE ENTERO, NO CORTADO CON «…»


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% R3 / R4 · LA COLUMNA ACCIONES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_r3_la_columna_derecha_se_llama_acciones_y_trae_volumen_ci_y_cosechar(ventana):
    w = ventana
    estado(w)
    assert w.ETIQUETA_ACCIONES.text() == "ACCIONES"
    for nombre in ("VER_VOLUMEN", "LARGO_HERRAMIENTA", "CALCULAR", "COSECHAR"):
        control = getattr(w, nombre, None)
        assert control is not None and w.FONDO_ACCIONES.isAncestorOf(control), nombre
    assert "Herramienta (mm)" in [e.text() for e in w.FONDO_ACCIONES.findChildren(QtWidgets.QLabel)]


def test_r3_acciones_se_contrae_a_una_franja_igual_que_vision(ventana):
    w = ventana
    estado(w)
    assert w.FONDO_ACCIONES.isVisible() and not w.FRANJA_ACCIONES.isVisible()
    clic(w.BTN_CONTRAER_ACCIONES)
    assert not w.FONDO_ACCIONES.isVisible() and w.FRANJA_ACCIONES.isVisible()
    clic(w.FRANJA_ACCIONES)
    assert w.FONDO_ACCIONES.isVisible() and not w.FRANJA_ACCIONES.isVisible()
    # LA MISMA FRANJA QUE LA DE VISIÓN: MISMO ANCHO Y MISMA LETRA
    assert w.FRANJA_ACCIONES.maximumWidth() == w.FRANJA_VISION.maximumWidth()
    assert w.FRANJA_ACCIONES.styleSheet() == w.FRANJA_VISION.styleSheet() == ""
    assert w.FRANJA_ACCIONES.property("rol") == w.FRANJA_VISION.property("rol")


def test_r3_acciones_comparte_el_mecanismo_de_vision_y_se_recuerda(ventana, robot):
    w = ventana
    estado(w)
    w.contraer_acciones(True)
    assert w.ajustes.value("acciones/contraida") == "true" and w.ajustes.value("vision/contraida") in (None, "false", False)
    w.ajustes.sync()
    ajustes = QtCore.QSettings(w.ajustes.fileName(), QtCore.QSettings.IniFormat)
    otra = VentanaPrincipal(robot, ajustes=ajustes, carpeta_programas=w.carpeta_programas, registro=reg.Registro(carpeta=None))
    try:
        assert otra.FRANJA_ACCIONES.isHidden() is False and otra.FONDO_ACCIONES.isHidden() is True
    finally:
        otra.close()
        w.contraer_acciones(False)
    import inspect
    from cosechador.ui import ventana as v
    assert "contraer_columna" in inspect.getsource(v.VentanaPrincipal.contraer_vision)      # UN SOLO MECANISMO
    assert "contraer_columna" in inspect.getsource(v.VentanaPrincipal.contraer_acciones)


@pytest.mark.parametrize("tam", TAMANOS)
def test_r4_al_contraer_acciones_el_visor_crece_exactamente_lo_que_la_columna_cedio(ventana, tam):
    w = ventana
    estado(w, tam)
    ancho_abierta, columna, franja = w.visor.width(), w.FONDO_ACCIONES.width(), w.FRANJA_ACCIONES.maximumWidth()
    w.contraer_acciones(True, recordar=False)
    pasar()
    assert w.visor.width() - ancho_abierta == columna - w.FRANJA_ACCIONES.width()
    assert w.FRANJA_ACCIONES.width() <= franja
    w.contraer_acciones(False, recordar=False)
    pasar()
    assert w.visor.width() == ancho_abierta                                          # Y AL REABRIRLA, VUELVE A SU ANCHO


def test_r4_ancho_del_visor_en_cada_estado_a_1280_por_720(ventana):
    """Documenta (y fija) cuánto ancho tiene el visor con VISIÓN y ACCIONES abiertas o en franja."""
    w = ventana
    medido = {}
    for vision, acciones in itertools.product((True, False), repeat=2):
        estado(w, (1280, 720), vision=vision, acciones=acciones)
        medido[(vision, acciones)] = w.visor.width()
    assert medido[(True, True)] < medido[(True, False)] < medido[(False, False)]
    assert medido[(True, True)] < medido[(False, True)] < medido[(False, False)]
    estado(w)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EL AUDITOR: TIPOS NUEVOS Y DEFECTOS SEMBRADOS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_el_auditor_revisa_los_tipos_nuevos(ventana):
    w = ventana
    estado(w)
    vistos = {type(x).__name__ for x in au.hojas(w)}
    assert {"Pestanas", "FigureCanvasQTAgg", "QtInteractor"} <= vistos, vistos          # (Pestanas ES UN QTabBar)
    assert any(isinstance(x, QtWidgets.QTabBar) for x in au.hojas(w))
    estado(w, pestana=FICHA)
    assert "QLabel" in {type(x).__name__ for x in au.hojas(w, (QtWidgets.QLabel,))}
    assert any(w.visor.ficha.tabla.isAncestorOf(x) for x in au.hojas(w, (QtWidgets.QLabel,)))      # LAS CELDAS DE LA TABLA
    estado(w)


def test_el_auditor_caza_defectos_sembrados_en_los_tipos_nuevos(ventana):
    w = ventana
    estado(w)
    assert au.auditar(w) == []

    # PESTAÑAS: LA TIRA ENCIMADA SOBRE EL LIENZO DE LA LATERAL
    barra = w.visor.pestanas
    original = barra.geometry()
    barra.setParent(w.visor.mosaico)
    barra.setGeometry(w.visor.lateral.lienzo.geometry())
    barra.show()
    assert any("PESTANAS_VISTAS" in h and h.startswith("solape") for h in au.solapes(w))
    barra.setParent(w.visor)
    w.visor.layout().itemAt(0).layout().insertWidget(0, barra)
    barra.setGeometry(original)
    barra.show()
    estado(w)

    # LIENZOS: UN LIENZO QUE SE SALE DE SU RECUADRO, Y DOS QUE SE ENCIMAN
    lienzo = w.visor.lateral.lienzo
    w.visor.rejilla.removeWidget(lienzo)
    lienzo.setGeometry(-30, 0, lienzo.width(), lienzo.height())
    assert any("recorte" in h and "se sale" in h for h in au.recortes(w))
    lienzo.setGeometry(w.visor.superior.lienzo.geometry())
    assert any(h.startswith("solape") for h in au.solapes(w))
    estado(w)

    # TABLA DE LA FICHA: UNA CELDA CON UN VALOR QUE NO CABE
    estado(w, pestana=FICHA)
    celda = [e for e in w.visor.ficha.tabla.findChildren(QtWidgets.QLabel) if e.property("rol") == "dato"][0]
    texto = celda.text()
    celda.setText("valor larguísimo que no cabe en su celda de la tabla de datos del robot")
    assert any(h.startswith("texto") for h in au.textos_cortados(w))
    celda.setText(texto)
    estado(w)

    # BOTÓN DE HERRAMIENTA: CHICO Y CON UN TEXTO QUE NO CABE
    boton = QtWidgets.QToolButton(w.visor)
    boton.setObjectName("BOTON_SEMBRADO")
    boton.setText("Herramienta con un texto demasiado largo")
    boton.setGeometry(4, 4, 14, 14)
    boton.show()
    assert any("BOTON_SEMBRADO" in h for h in au.controles_chicos(w))
    boton.deleteLater()
    pasar()


def test_el_auditor_caza_rotulos_encimados_fuera_del_lienzo_y_con_letra_menor_a_la_minima(robot, lazos):
    assert au.LETRA_MINIMA == LETRA_MINIMA == planos.LETRA_MINIMA                  # LA MISMA MÍNIMA EN LOS TRES LUGARES
    d = solo(planos.Lateral(planos.Geometria(robot, lazos)))
    assert au.rotulos_de(d) == []
    t = d.textos[0]
    t.set_fontsize(9 * planos.PT)                                                    # LETRA DE 9 px
    assert any("letra de 9 px" in h for h in au.rotulos_de(d))
    t.set_fontsize(planos.LETRA_MINIMA * planos.PT)
    d.componer()
    d.lienzo.draw()
    d.textos[1].set_position((0, 0))                                                 # DOS RÓTULOS EN EL MISMO SITIO
    d.textos[1].xy = d.textos[0].xy
    d.textos[1].set_position(d.textos[0].get_position())
    d.textos[1].set_ha(d.textos[0].get_ha())
    d.textos[1].set_va(d.textos[0].get_va())
    d.textos[1].xycoords = d.textos[0].xycoords
    assert any("se enciman" in h for h in au.rotulos_de(d))
    d.componer()
    d.lienzo.draw()
    d.textos[0].xy, d.textos[0].xycoords = (0.0, 1.0), "figure fraction"            # PEGADO A LA ESQUINA, AFUERA DEL LIENZO
    d.textos[0].set_position((-80, 40))
    assert any("se sale del lienzo" in h for h in au.rotulos_de(d))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOS RÓTULOS, RECORRIENDO LO QUE PUEDE CAMBIAR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

TAMANOS_DE_DIBUJO = [(247, 195), (320, 240), (480, 300), (620, 400), (880, 540)]
GIROS = [0, 37, 90, 180, 270, 360]                      # 360 = EL MÁXIMO DEL ROBOT (VUELTA COMPLETA)


def _lazos_de_prueba(xmin, xmax, zmin, zmax):
    """Un contorno cualquiera (elipse) con esas medidas: el brazo de 4 GDL tarda 19 s en calcular el suyo."""
    t = np.linspace(0, 2 * np.pi, 120)
    return [np.column_stack([(xmin + xmax) / 2 + (xmax - xmin) / 2 * np.cos(t), (zmin + zmax) / 2 + (zmax - zmin) / 2 * np.sin(t)])]


@pytest.fixture(scope="module")
def lazos_4gdl():
    return _lazos_de_prueba(-330, 630, -430, 830)


@pytest.mark.parametrize("clase", [planos.Lateral, planos.Superior])
@pytest.mark.parametrize("giro", GIROS)
@pytest.mark.parametrize("volumen", [True, False])
def test_f10_7_ningun_rotulo_se_encima_se_sale_ni_baja_de_12_px_en_cada_giro_de_la_base(robot, lazos, clase, giro, volumen):
    geo = planos.Geometria(robot, lazos)
    geo.desde, geo.hasta = 0.0, float(giro)
    fallas = {}
    for tam in TAMANOS_DE_DIBUJO:
        d = solo(clase(geo, servos=(60, 75, 115)), *tam, volumen=volumen)
        hallazgos = au.rotulos_de(d)
        if hallazgos:
            fallas[tam] = hallazgos
        assert d.letras() and min(d.letras()) >= LETRA_MINIMA
    assert not fallas, fallas


@pytest.mark.parametrize("clase", [planos.Lateral, planos.Superior])
@pytest.mark.parametrize("volumen", [True, False])
def test_f10_7_lo_mismo_con_el_brazo_de_4_gdl_de_prueba(robot_4gdl, lazos_4gdl, clase, volumen):
    geo = planos.Geometria(robot_4gdl, lazos_4gdl)
    fallas = {}
    for tam in TAMANOS_DE_DIBUJO:
        d = solo(clase(geo), *tam, volumen=volumen)
        if au.rotulos_de(d):
            fallas[tam] = au.rotulos_de(d)
    assert not fallas, fallas


@pytest.mark.parametrize("cambio", ["limites", "medidas"])
def test_f10_7_lo_mismo_con_limites_o_medidas_cambiados(robot, cambio):
    otro = variante(robot, limites={1: (20.0, 120.0)}) if cambio == "limites" else variante(robot, medidas={1: 640.0, 2: 350.0})
    geo = planos.Geometria(otro, contorno_del_corte(otro))
    fallas = {}
    for clase, tam in itertools.product((planos.Lateral, planos.Superior), TAMANOS_DE_DIBUJO):
        d = solo(clase(geo), *tam)
        if au.rotulos_de(d):
            fallas[(clase.__name__, tam)] = au.rotulos_de(d)
    assert not fallas, fallas


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LA MATRIZ DE ESTADOS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def _estados_de_la_matriz():
    """tamaño × VISIÓN × ACCIONES × pestaña del visor × volumen × pestaña de movimiento. Con el editor de puntos abierto VISIÓN
    siempre está en franja (el editor ocupa su lugar), así que esa combinación no existe."""
    sin_editor = list(itertools.product(TAMANOS, (True, False), (True, False), range(5), (False, True),
                                        (PESTANA_SERVOS, PESTANA_XYZ, PESTANA_PROGRAMA)))
    con_editor = list(itertools.product(TAMANOS, (False,), (True, False), range(5), (False, True), ("editor",)))
    return sin_editor + con_editor


ESTADOS = _estados_de_la_matriz()


def test_la_matriz_tiene_los_estados_que_se_declaran():
    assert len(ESTADOS) == 3 * 2 * 2 * 5 * 2 * 3 + 3 * 1 * 2 * 5 * 2 * 1 == 420


def test_la_matriz_de_estados_el_auditor_no_encuentra_nada(ventana, poses, lazos):
    """420 estados de la ventana real: `auditar(...) == []` en todos (incluye los rótulos de los dibujos)."""
    w = ventana
    fallas, auditados, editor_abierto = {}, 0, False
    for tam, vision, acciones, pestana, volumen, movimiento in ESTADOS:
        con_editor = movimiento == "editor"
        if con_editor and not editor_abierto:
            w.cerrar_editor()
            estado(w, tam, pestana, vision, acciones, volumen, lazos, PESTANA_PROGRAMA)
            abrir(w, "matriz.yaml", *_tres(poses))
            editor_abierto = True
        elif not con_editor and editor_abierto:
            w.cerrar_editor()
            editor_abierto = False
        if con_editor:
            w.resize(*tam)
            w.contraer_acciones(not acciones, recordar=False)
            w.visor.ver(pestana)
            w.visor.poner_volumen(lazos, True) if volumen else w.visor.ocultar_volumen()
            pasar()
        else:
            estado(w, tam, pestana, vision, acciones, volumen, lazos, movimiento)
        assert w.editando == con_editor and (w.visor.principal, w.visor.volumen_visible) == (pestana, volumen)   # EL ESTADO ES EL QUE DICE SER
        assert (w.FONDO_VISION.isVisible(), w.FONDO_ACCIONES.isVisible()) == (vision, acciones) and (w.width(), w.height()) == tam
        hallazgos = au.auditar(w)
        auditados += 1
        if hallazgos:
            clave = (f"{tam[0]}×{tam[1]} · VISIÓN {'abierta' if vision else 'franja'} · ACCIONES {'abierta' if acciones else 'franja'} · "
                     f"visor {NOMBRES[pestana]} · volumen {'a la vista' if volumen else 'oculto'} · movimiento "
                     f"{movimiento if con_editor else ('Servos', 'Punta X Y Z', 'Programa')[movimiento]}")
            fallas[clave] = hallazgos
    w.cerrar_editor()
    assert auditados == len(ESTADOS) == 420
    assert not fallas, "\n" + "\n".join(f"{k}:\n    " + "\n    ".join(v) for k, v in list(fallas.items())[:12])
