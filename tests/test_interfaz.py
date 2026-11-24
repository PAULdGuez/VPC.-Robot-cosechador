"""La interfaz, medida al píxel: ningún elemento se encima, se sale, corta su texto ni queda ilegible, en
ninguna combinación de tamaño de ventana × visión abierta/cerrada × pestaña. Usa tests/auditoria_interfaz.py.

El .ui se puede editar en Qt Designer: estas pruebas avisan si algo de eso se rompe.
"""
import os
import re
import xml.etree.ElementTree as ET

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5 import QtCore, QtWidgets, uic

import auditoria_interfaz as au
from cosechador.ui import tema, ventana

CODIGO = open(ventana.__file__, encoding="utf-8").read()
TAMANOS = [(1280, 760), (1440, 860), (1512, 863)]   # LAPTOP CHICA, TAMAÑO POR OMISIÓN, PANTALLA DE PAUL
GEOMETRIA = (au.solapes, au.recortes, au.textos_cortados, au.textos_pegados_al_borde, au.filas_desalineadas)


@pytest.fixture(scope="module")
def ui():
    """La ventana vestida y llena como en la app (sin la escena 3D), con los textos más anchos posibles."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = QtWidgets.QMainWindow()
    uic.loadUi(ventana.RUTA_UI, w)
    tema.aplicar(w)
    for clave, texto in ventana.POSTURAS:
        w.SELECTOR_POSTURA.addItem(texto, clave)
    for texto, niveles in ventana.FILTROS_REGISTRO:
        w.FILTRO_REGISTRO.addItem(texto, niveles)
    w.SELECTOR_PROGRAMA.addItems(["demo-cosecha.yaml", "ejemplo-grabacion.yaml"])
    for chip, texto in ((w.VS_1, "180°"), (w.VS_2, "46°"), (w.VS_3, "46°"), (w.VS_X, "-2050"), (w.VS_Y, "2050"),
                        (w.VS_Z, "1065"), (w.VS_VELOCIDAD, "100 %"), (w.VS_TONO, "255"), (w.VS_SATURACION, "170"),
                        (w.L_CONTADOR, "12")):
        chip.setText(texto)
    w.ESTADO_ROBOT.setText("Cosechador de aguacates (3 GDL)")
    w.ESTADO_CONEXION.setText("<span style='color:#f1c40f'>●</span> ESP32 en /dev/cu.usbserial-0001")
    w.INFO_PROGRAMA.setText("Grabación · 120 muestras · 12.5 s")
    w.BTN_GRABAR.setText(ventana.TEXTO_DETENER_GRABACION)   # Y CON SU ASPECTO DE «GRABANDO» (ROJO)
    w.BTN_GRABAR.setProperty("rol", "grabando")
    w.BTN_GRABAR.style().unpolish(w.BTN_GRABAR)
    w.BTN_GRABAR.style().polish(w.BTN_GRABAR)
    w.LISTA_REGISTRO.setWordWrap(True)
    w.LISTA_REGISTRO.addItems(["17:21:03  ⚠ AVISO · Jog\nNo se mueve: el camino pasa por una zona que el robot no alcanza"] * 4)
    w.BTN_NUEVOS_REGISTRO.hide()
    w.FRANJA_VISION.hide()
    w.FONDO_PUNTOS.hide()        # EL EDITOR DE PUNTOS NACE CERRADO (SUS ESTADOS: test_editor_puntos.py)
    w.resize(1440, 860)
    w.show()
    app.processEvents()
    yield w
    w.close()


def poner_vision(ui):
    def poner(abierta):
        ui.FONDO_VISION.setVisible(abierta)
        ui.FRANJA_VISION.setVisible(not abierta)
    return poner


def _geometria(ui, nombre):
    w = getattr(ui, nombre)
    esquina = w.mapTo(ui, w.rect().topLeft())
    return (esquina.x(), esquina.y(), w.width(), w.height())


def _sin_defectos(ui, revisiones):
    defectos = au.auditar_estados(ui, TAMANOS, poner_vision(ui), ui.PESTANAS_MOVIMIENTO, revisiones)
    poner_vision(ui)(True)
    ui.PESTANAS_MOVIMIENTO.setCurrentIndex(0)
    ui.resize(1440, 860)
    assert not defectos, "\n" + "\n".join(f"{estado}:\n    " + "\n    ".join(h) for estado, h in defectos.items())


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EL .ui Y EL CÓDIGO COINCIDEN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_todo_control_que_usa_el_codigo_existe_en_el_ui(ui):
    usados = set(re.findall(r"self\.([A-Z][A-Z0-9_a-z]*|label_\d+)\b", CODIGO))
    faltan = sorted(n for n in usados if not hasattr(ui, n))
    assert not faltan, f"El .ui ya no tiene: {faltan}"


def test_tres_pestanas_de_movimiento(ui):
    pestanas = ui.PESTANAS_MOVIMIENTO
    assert [pestanas.tabText(i) for i in range(pestanas.count())] == ["Servos", "Punta X Y Z", "Programa"]


def test_u5_controles_de_vision_viven_en_su_columna(ui):
    for nombre in ("SLIDER_TONO", "SLIDER_SATURACION", "CARGA", "GUARDAR"):
        assert ui.FONDO_VISION.isAncestorOf(getattr(ui, nombre)), nombre


@pytest.mark.parametrize("ancho", [1280, 1440, 1512])
def test_u4_contraer_la_vision_no_mueve_el_panel_derecho(ui, ancho):
    app = QtWidgets.QApplication.instance()
    ui.resize(ancho, 860)
    poner_vision(ui)(True); app.processEvents()
    derecho_abierta, simulacion_abierta = _geometria(ui, "FONDO_HERRAMIENTAS"), _geometria(ui, "PANTALLA")
    poner_vision(ui)(False); app.processEvents()
    derecho_cerrada, simulacion_cerrada = _geometria(ui, "FONDO_HERRAMIENTAS"), _geometria(ui, "PANTALLA")
    poner_vision(ui)(True); app.processEvents()
    assert derecho_cerrada == derecho_abierta                   # EL PANEL DERECHO NO CAMBIA NI UN PÍXEL
    assert simulacion_cerrada[2] > simulacion_abierta[2] + 200  # LA SIMULACIÓN GANA EL ESPACIO


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CRITERIOS DE LA COMPACTACIÓN (K) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_k1_la_vista_3d_mide_al_menos_520_px(ui):
    ui.resize(1440, 860)
    QtWidgets.QApplication.processEvents()
    assert ui.PANTALLA.height() >= 520  # ANTES DE COMPACTAR: 435


def test_k2_nada_se_encima_se_sale_ni_corta_su_texto(ui):
    _sin_defectos(ui, GEOMETRIA)


def test_k3_todo_texto_contrasta_con_su_fondo(ui):
    _sin_defectos(ui, (au.contrastes,))


def test_k4_todo_control_mide_al_menos_24_px(ui):
    _sin_defectos(ui, (au.controles_chicos,))


def test_k8_la_ventana_cabe_en_1280_por_720(ui):
    poner_vision(ui)(True)
    QtWidgets.QApplication.processEvents()
    minimo = ui.minimumSizeHint()
    assert (minimo.width(), minimo.height()) <= (1280, 720) and minimo.height() <= 720, (minimo.width(), minimo.height())


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% REGLAS DEL TEMA Y DEL .ui %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_la_letra_de_las_pestanas_no_va_en_el_subcontrol():
    # Qt MIDE LA PESTAÑA CON LA LETRA DE QTabBar: SI ::tab TRAE OTRA MÁS GRANDE, EL TEXTO SALE CORTADO
    for selector, cuerpo in re.findall(r"([^{}]+)\{([^}]*)\}", tema.hoja_de_estilos()):
        if "::tab" in selector and "tab-bar" not in selector:
            assert "font" not in cuerpo, selector.strip()


def test_los_espacios_van_en_rejilla_de_4_px():
    fuera = []
    for propiedad in ET.parse(ventana.RUTA_UI).iter("property"):
        nombre = propiedad.get("name")
        if nombre.endswith("Margin") or nombre.lower().endswith("spacing"):
            valor = int(propiedad.find("number").text)
            if valor % 4:
                fuera.append((nombre, valor))
    assert not fuera, fuera


def test_el_ui_no_trae_estilos_sueltos():
    # LOS ESTILOS VIVEN EN tema.py; UNO SUELTO EN EL .ui LE GANA AL TEMA Y ROMPE LA CONSISTENCIA
    sueltos = [w.get("name") for w in ET.parse(ventana.RUTA_UI).iter("widget")
               if w.find("./property[@name='styleSheet']") is not None]
    assert not sueltos, sueltos


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EL AUDITOR SÍ CAZA (DEFECTOS SEMBRADOS) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_el_auditor_caza_defectos_sembrados(ui):
    app = QtWidgets.QApplication.instance()
    assert au.auditar(ui) == []

    ui.VS_1.setText("-123456789")                                  # TEXTO QUE NO CABE
    assert any("VS_1" in h and h.startswith("texto") for h in au.auditar(ui, (au.textos_cortados,)))
    ui.VS_1.setText("180°")

    ui.VER_VOLUMEN.setStyleSheet("color: #dcdcdc;")                 # GRIS CLARO SOBRE BLANCO
    assert any("VER_VOLUMEN" in h for h in au.auditar(ui, (au.contrastes,)))
    ui.VER_VOLUMEN.setStyleSheet("")

    ui.CHK_TRAYECTORIA.setFixedHeight(16)                           # CONTROL DEMASIADO CHICO
    assert any("CHK_TRAYECTORIA" in h for h in au.auditar(ui, (au.controles_chicos,)))
    ui.CHK_TRAYECTORIA.setMinimumHeight(0)
    ui.CHK_TRAYECTORIA.setMaximumHeight(16777215)

    ui.BTN_COPIAR_REGISTRO.move(ui.BTN_EXPORTAR_REGISTRO.pos() + QtCore.QPoint(-20, 0))  # UN BOTÓN SOBRE OTRO
    assert any("BTN_COPIAR_REGISTRO" in h and "BTN_EXPORTAR_REGISTRO" in h for h in au.solapes(ui))
    ui.FONDO_REGISTRO.layout().invalidate()
    ui.FONDO_REGISTRO.layout().activate()
    app.processEvents()
    assert au.auditar(ui) == []
