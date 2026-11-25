"""Punto de entrada: python -m cosechador [--robot robots/otro.yaml]"""
import argparse
import sys
import traceback

from PyQt5.QtGui import QSurfaceFormat
from PyQt5.QtWidgets import QApplication, QMessageBox

from cosechador.robot import ErrorConfiguracion, cargar_robot


def preparar_opengl():
    """En macOS VTK necesita un contexto OpenGL 3.2+ ("core profile"); sin esto los shaders
    fallan (version '150' is not supported) y no se dibujan piezas ni volumen.
    Debe llamarse ANTES de crear QApplication."""
    formato = QSurfaceFormat()
    formato.setRenderableType(QSurfaceFormat.OpenGL)
    formato.setVersion(3, 2)
    formato.setProfile(QSurfaceFormat.CoreProfile)
    formato.setDepthBufferSize(24)
    formato.setStencilBufferSize(8)
    formato.setSamples(4)
    QSurfaceFormat.setDefaultFormat(formato)


def mostrar_error(tipo, valor, tb):
    # SIN ESTO, UNA EXCEPCIÓN DENTRO DE UN BOTÓN CIERRA TODA LA APP (PyQt5 >= 5.5)
    traceback.print_exception(tipo, valor, tb)
    from cosechador.ui.ventana import VentanaPrincipal
    for ventana in QApplication.topLevelWidgets():
        if isinstance(ventana, VentanaPrincipal):
            ventana.avisar(f"Error inesperado: {valor}", "error", "Sistema", tipo=tipo.__name__,
                           donde=traceback.format_tb(tb)[-1].strip() if tb else "")


def main(argv=None):
    argumentos = argparse.ArgumentParser(description="Robot cosechador de aguacates")
    argumentos.add_argument("--robot", help="archivo YAML del robot (por omisión: ROBOT_CONFIG o robots/cosechador_3gdl.yaml)")
    opciones, resto = argumentos.parse_known_args(argv)

    sys.excepthook = mostrar_error
    preparar_opengl()
    app = QApplication([sys.argv[0], *resto])
    try:
        robot = cargar_robot(opciones.robot)
    except ErrorConfiguracion as e:
        print(f"Error en la configuración del robot: {e}", file=sys.stderr)
        QMessageBox.critical(None, "Configuración del robot", str(e))
        return 2

    from cosechador.ui.ventana import VentanaPrincipal
    ventana = VentanaPrincipal(robot)
    ventana.show()
    return app.exec_()
