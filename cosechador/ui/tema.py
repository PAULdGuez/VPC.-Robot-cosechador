"""Tema visual: UNA sola hoja de estilos para toda la interfaz.

Antes cada control traía su propio estilo dentro del .ui (30 bloques, con duplicados, tres tamaños de
letra para el mismo tipo de botón y dos fuentes de Windows que la Mac no tiene). Ahora:

- Los colores, tamaños y radios viven UNA vez, en TOKENS.
- El .ui solo dice QUÉ es cada control, con la propiedad `rol` (seccion, etiqueta, chip, accion…).
- La hoja de estilos dice CÓMO se ve cada rol.

Reglas que cuida este archivo (las vigila tests/test_interfaz.py):
- Todo control interactivo mide al menos 24 px de alto (WCAG 2.2, criterio 2.5.8).
- Todo texto contrasta al menos 4.5:1 con su fondo (WCAG AA).
- La letra de las pestañas se define en QTabBar, nunca en QTabBar::tab: Qt calcula el ancho de la
  pestaña con la letra del control, y si la del sub-control es más grande el texto sale cortado.
"""
import os
from string import Template

from PyQt5 import QtCore, QtWidgets

ICONOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iconos")

TOKENS = {
    # COLORES
    "tinta": "#1f2d3a", "tinta_suave": "#4a5560", "tinta_apagada": "#6b7680",
    "fondo": "#ffffff", "panel": "#ebebeb", "borde": "#000000", "borde_suave": "#9aa5b1",
    "primario": "#2c3e50", "primario_claro": "#3d5469", "resalte": "#f3f6f9", "seleccion": "#dde6ee",
    "lima": "#c0de5d", "azul": "#101f78", "rojo": "#b11623", "verde": "#1e8449", "amarillo": "#f7db15",
    "pista": "#d3d3d3", "manija": "#d43b2d", "manija_vision": "#0078d7", "apagado": "#b8bec4",
    # LETRA
    "letra": "13px", "letra_chica": "12px", "letra_titulo": "15px",
    # RADIOS
    "radio": "8px", "radio_chico": "6px", "radio_tarjeta": "10px",
    # ICONOS
    "flecha": os.path.join(ICONOS, "flecha-abajo.svg").replace(os.sep, "/"),
}

_HOJA = Template("""
/* %%%%% BASE %%%%% */
QMainWindow, QWidget#centralwidget { background: $fondo; }
QWidget { font-size: $letra; color: $tinta; }
QStatusBar { font-size: $letra_chica; color: $tinta_suave; }
QStatusBar::item { border: none; }

/* %%%%% BARRA DE TÍTULO %%%%% */
QFrame#BARRA_TITULO { background: $primario; border-radius: $radio; }
QLabel#TITULO_APP { color: white; font-size: $letra_titulo; font-weight: bold; }
QLabel[rol="estado"] { color: white; font-size: $letra_chica; }

/* %%%%% PANELES Y TARJETAS %%%%% */
QFrame[rol="panel"] { background: $panel; border-radius: $radio_chico; }
QFrame#FONDO_HERRAMIENTAS { background: $panel; border: 2px solid $borde; }
QFrame[rol="tarjeta"] { border-radius: $radio_tarjeta; }
QFrame#TARJETA_VOLUMEN { background: $lima; }
QFrame#TARJETA_CI { background: $azul; }
QFrame#TARJETA_COSECHA { background: $rojo; }
QFrame#FONDO_REGISTRO { background: $fondo; }
QWidget[rol="pagina"] { background: $fondo; }

/* %%%%% TEXTOS %%%%% */
QLabel[rol="seccion"] { font-size: $letra_titulo; font-weight: bold; color: $primario; }
QLabel[rol="subtitulo"] { font-size: $letra_chica; font-weight: bold; color: $tinta_suave; }
QLabel[rol="nota"] { font-size: $letra_chica; color: $tinta_suave; }
QFrame#TARJETA_VOLUMEN QLabel[rol="nota"] { color: $tinta; }
QFrame#TARJETA_CI QLabel[rol="nota"], QFrame#TARJETA_COSECHA QLabel[rol="nota"] { color: white; }
QLabel[rol="titulo-tarjeta"] { color: white; font-weight: bold; }
QLabel[rol="chip"] { background: $lima; color: $tinta; border-radius: 4px; padding: 0px 6px;
                     min-height: 24px; max-height: 24px; min-width: 40px; }
QLabel[rol="imagen"] { background: $fondo; border: 2px solid $borde; border-radius: 5px; color: $tinta_suave; }

/* %%%%% BOTONES %%%%% */
QPushButton { background: $fondo; color: $tinta; border: 2px solid $borde; border-radius: $radio;
              padding: 0px 12px; min-height: 22px; }
QPushButton:hover { background: $resalte; }
QPushButton:pressed { background: $seleccion; }
QPushButton:focus { border-color: $manija_vision; }
QPushButton[rol="mini"] { font-size: $letra_chica; padding: 0px 8px; min-height: 20px; border-radius: $radio_chico; }
QPushButton[rol="accion"] { font-size: $letra_titulo; min-height: 30px; border-radius: $radio_tarjeta; }
QPushButton[rol="accion"]:hover { background: $amarillo; }
QPushButton[rol="confirmar"] { background: $verde; color: white; font-size: $letra_titulo; min-height: 26px;
                               border-radius: $radio_tarjeta; }
QPushButton[rol="confirmar"]:hover { background: $amarillo; color: $tinta; }
QPushButton[rol="primario"] { background: $primario; color: white; border-color: $primario;
                              font-size: $letra_titulo; min-height: 30px; border-radius: $radio_tarjeta; }
QPushButton[rol="primario"]:hover { background: $primario_claro; border-color: $primario_claro; }
QPushButton[rol="suave"] { background: $primario; color: white; border: none; border-radius: $radio_chico;
                           font-size: $letra_chica; font-weight: bold; padding: 0px 4px; min-height: 24px; }
QPushButton[rol="suave"]:hover { background: $primario_claro; }
QPushButton:disabled, QPushButton[rol="accion"]:disabled, QPushButton[rol="confirmar"]:disabled,
QPushButton[rol="primario"]:disabled { background: #f2f2f2; color: $tinta_apagada; border-color: $apagado; }

/* %%%%% SLIDERS %%%%% */
QSlider { min-height: 24px; }
QSlider::groove:horizontal { border: 1px solid #999999; height: 12px; background: $pista; border-radius: 3px; }
QSlider::handle:horizontal { background: $manija; border: 1px solid #5c5c5c; width: 16px; margin: -4px 0px;
                             border-radius: 8px; }
QSlider[rol="vision"]::handle:horizontal { background: $manija_vision; }
QSlider::handle:horizontal:hover { border: 2px solid $tinta; }
QSlider::handle:horizontal:disabled { background: $apagado; border-color: #999999; }

/* %%%%% SELECTORES, CAMPOS Y CASILLAS %%%%% */
QComboBox { background: $fondo; border: 2px solid $borde; border-radius: $radio; padding: 0px 30px 0px 10px;
            min-height: 22px; }
QComboBox:hover { background: $resalte; }
QComboBox:focus { border-color: $manija_vision; }
QComboBox:disabled { color: $tinta_apagada; border-color: $apagado; background: #f2f2f2; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: center right; width: 26px;
                       border-left: 1px solid #cccccc; }
QComboBox::down-arrow { image: url("$flecha"); width: 12px; height: 8px; }
QComboBox QAbstractItemView { background: $fondo; border: 2px solid $borde; selection-background-color: $primario;
                              selection-color: white; outline: 0; padding: 4px; }
QLineEdit { background: $fondo; border: 1px solid #888888; border-radius: 4px; padding: 0px 6px; min-height: 22px; }
QLineEdit:focus { border: 2px solid $manija_vision; padding: 0px 5px; }
QCheckBox { min-height: 24px; spacing: 6px; }
QFrame#FONDO_SIMULACION QCheckBox { color: $tinta; }

/* %%%%% PESTAÑAS %%%%% */
QTabWidget::pane { border: none; background: $fondo; }
QTabWidget::tab-bar { alignment: left; }
QTabBar { font-size: $letra; }
QTabBar::tab { padding: 4px 16px; background: $pista; color: $tinta; border: 1px solid #999999;
               border-bottom: none; border-top-left-radius: 6px; border-top-right-radius: 6px;
               margin-right: 2px; min-width: 56px; }
QTabBar::tab:selected { background: $primario; color: white; }
QTabBar::tab:!selected:hover { background: #e6e6e6; }
QTabBar::tab:disabled { color: $tinta_apagada; }

/* %%%%% REGISTRO %%%%% */
QListWidget { background: $fondo; border: 1px solid $borde_suave; border-radius: $radio_chico; font-size: $letra_chica; }
QListWidget::item { padding: 3px 4px; border-bottom: 1px solid #eeeeee; }
QListWidget::item:selected { background: $seleccion; color: #000000; }
QTextBrowser { background: #fafafa; border: 1px solid #cccccc; border-radius: $radio_chico; font-size: $letra_chica; }
QProgressBar { border: none; background: rgba(0, 0, 0, 45); border-radius: 3px; min-height: 6px; max-height: 6px; }
QProgressBar::chunk { background: $primario; border-radius: 3px; }
""")

# EN macOS LA BARRA DE DESPLAZAMIENTO NO TOMA EL ESTILO DE LA VENTANA NI EL DE SU LISTA (QUEDA DE 16 px, CON
# UNA LÍNEA NEGRA QUE TAPA TEXTO): SE LE PONE DIRECTO A CADA BARRA. VER aplicar().
_BARRA = Template("""
QScrollBar:vertical { width: 8px; background: transparent; margin: 2px 1px; border: none; }
QScrollBar::handle:vertical { background: #c4c4c4; border: none; border-radius: 3px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: $borde_suave; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; border: none; background: none; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
""")


def hoja_de_estilos():
    return _HOJA.substitute(TOKENS)


def aplicar(ventana):
    """Viste la ventana. Se llama una vez, después de cargar el .ui."""
    ventana.setStyleSheet(hoja_de_estilos())
    # EN macOS EL ESTILO NATIVO LE CALCULA A CADA CONTROL UN RECTÁNGULO DE LAYOUT MÁS CHICO QUE EL QUE SE DIBUJA:
    # DOS BOTONES CON BORDE PROPIO SE ENCIMABAN 2 px AUNQUE EL LAYOUT PIDIERA SEPARACIÓN. CON ESTO EL LAYOUT USA
    # EL TAMAÑO REAL DE CADA CONTROL, AL PÍXEL.
    for control in ventana.findChildren(QtWidgets.QWidget):
        control.setAttribute(QtCore.Qt.WA_LayoutUsesWidgetRect)
    for area in ventana.findChildren(QtWidgets.QAbstractScrollArea):
        area.verticalScrollBar().setStyleSheet(_BARRA.substitute(TOKENS))
