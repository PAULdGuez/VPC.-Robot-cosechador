"""El .ui se puede editar en Qt Designer: esta prueba avisa si se renombra o borra un control que el
código usa, o si se vuelve a esconder un botón debajo de otro (GUARDAR estaba bajo COSECHAR)."""
import os
import re

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5 import QtWidgets, uic

from cosechador.ui import ventana

CODIGO = open(ventana.__file__, encoding="utf-8").read()


@pytest.fixture(scope="module")
def ui():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = QtWidgets.QMainWindow()
    uic.loadUi(ventana.RUTA_UI, w)
    w.resize(1440, 833)
    w.show()
    app.processEvents()
    yield w
    w.close()


def test_todo_control_que_usa_el_codigo_existe_en_el_ui(ui):
    usados = set(re.findall(r"self\.([A-Z][A-Z0-9_a-z]*|label_\d+)\b", CODIGO))
    usados -= {"PESTANA_SERVOS", "PESTANA_XYZ", "PESTANA_PROGRAMA"}  # CONSTANTES DEL MÓDULO, NO CONTROLES
    faltan = sorted(n for n in usados if not hasattr(ui, n))
    assert not faltan, f"El .ui ya no tiene: {faltan}"


def test_tres_pestanas_de_movimiento(ui):
    pestanas = ui.PESTANAS_MOVIMIENTO
    assert [pestanas.tabText(i) for i in range(pestanas.count())] == ["Servos", "Punta X Y Z", "Programa"]


def test_ningun_boton_tapa_a_otro(ui):
    botones = [b for b in ui.findChildren(QtWidgets.QPushButton) if b.isVisible()]
    rect = {b.objectName(): b.rect().translated(b.mapTo(ui, b.rect().topLeft())) for b in botones}
    encimados = [(a, b) for i, a in enumerate(rect) for b in list(rect)[i + 1:] if rect[a].intersects(rect[b])]
    assert not encimados, f"Botones encimados: {encimados}"


def _geometria(ui, nombre):
    w = getattr(ui, nombre)
    esquina = w.mapTo(ui, w.rect().topLeft())
    return (esquina.x(), esquina.y(), w.width(), w.height())


@pytest.mark.parametrize("ancho", [1440, 1280, 1512])
def test_u4_contraer_la_vision_no_mueve_el_panel_derecho(ui, ancho):
    app = QtWidgets.QApplication.instance()
    ui.resize(ancho, 833)
    ui.FONDO_VISION.show(); ui.FRANJA_VISION.hide(); app.processEvents()
    derecho_abierta, simulacion_abierta = _geometria(ui, "FONDO_HERRAMIENTAS"), _geometria(ui, "PANTALLA")
    ui.FONDO_VISION.hide(); ui.FRANJA_VISION.show(); app.processEvents()
    derecho_cerrada, simulacion_cerrada = _geometria(ui, "FONDO_HERRAMIENTAS"), _geometria(ui, "PANTALLA")
    assert derecho_cerrada == derecho_abierta              # EL PANEL DERECHO NO CAMBIA NI UN PÍXEL
    assert simulacion_cerrada[2] > simulacion_abierta[2] + 200  # LA SIMULACIÓN GANA EL ESPACIO
    ui.FONDO_VISION.show(); ui.FRANJA_VISION.hide(); app.processEvents()


def test_u5_controles_de_vision_viven_en_su_columna(ui):
    for nombre in ("SLIDER_color_1", "SLIDER_color_2", "CARGA", "GUARDAR"):
        assert ui.FONDO_VISION.isAncestorOf(getattr(ui, nombre)), nombre
