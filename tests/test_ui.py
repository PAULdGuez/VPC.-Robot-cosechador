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
