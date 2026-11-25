"""La ficha del robot: la hoja «lado a lado» que se ve en la pestaña Ficha y la misma hoja guardada como imagen.

Los datos (qué se dice del robot y con qué números) se deciden en UN solo lugar, `datos_de_la_ficha`: de ahí leen la tabla
de la pantalla y la hoja guardada. La hoja guardada lleva además la vista 3D, que en pantalla no se repite (ya está en su
pestaña): se dibuja aparte, con la misma escena, en un lienzo fuera de pantalla.
"""
import datetime as dt

import numpy as np
import pyvista as pv
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PyQt5 import QtCore, QtGui, QtWidgets

from cosechador.robot.cinematica import servo_a_dh

from .escena3d import Escena
from .planos import Lateral, Superior, T

ANCHO_HOJA, ALTO_HOJA = 1600, 1000      # px DE LA IMAGEN GUARDADA
MESES = "ene feb mar abr may jun jul ago sep oct nov dic".split()
PIE = "En gris: lo que el robot alcanza por debajo del nivel de su base; el montaje decide si se puede usar."
SUBTITULO = "Volumen de trabajo · medidas en mm, tomadas a la punta de la herramienta"
SIN_VOLUMEN = "Con el volumen de trabajo a la vista, aquí van su alcance, sus alturas y su giro."


def _mm(valor):
    return f"{valor:g} mm"


def datos_de_la_ficha(geo):
    """Lo que dice la ficha: dos grupos de (nombre, valor). Valores cortos: no se parten en dos renglones.

    Construcción: lo que sale del archivo del robot. Alcance: lo que sale del contorno del volumen (si no hay uno a la vista,
    solo hasta dónde llega la punta)."""
    robot = geo.robot
    construccion = [("Articulaciones", str(robot.gdl))]
    for j in robot.articulaciones:
        medidas = [f"altura {_mm(j.d)}"] * bool(j.d) + [f"largo {_mm(abs(j.a))}"] * bool(j.a)
        construccion.append((j.nombre, " · ".join(medidas) or "sin medida"))
    construccion.append(("Herramienta", f"largo {_mm(robot.efector.largo)}"))
    for i, j in enumerate(robot.articulaciones, start=1):
        construccion.append((f"Servo {i} · {j.nombre}", f"{j.servo.minimo:g}° a {j.servo.maximo:g}°"))
    if geo.con_volumen:
        alcance = [("Alcance al frente", _mm(round(geo.frente))), ("Alcance hacia atrás", _mm(round(geo.atras))),
                   ("Altura máxima", _mm(round(geo.zmax))),
                   ("Altura mínima", f"{round(abs(geo.zmin))} mm bajo la base" if geo.zmin < 0 else _mm(round(geo.zmin))),
                   ("Giro de la base", f"{abs(geo.hasta - geo.desde):.0f}°")]
    else:
        alcance = [("Alcance de la punta", _mm(round(geo.alcance)))]
    return construccion, alcance


def fecha_de_hoy(hoy=None):
    hoy = hoy or dt.date.today()
    return f"{hoy.day}-{MESES[hoy.month - 1]}-{hoy.year}"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LA VISTA 3D, FUERA DE PANTALLA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def render_3d(robot, servos, lazos, ancho, alto, numeros=True, densidad=1):
    """La escena 3D (la misma clase que la ventana) como imagen (alto, ancho, 3), con el robot en `servos` y el volumen si hay
    `lazos`. Con `numeros` False la rejilla del volumen va sin números."""
    from cosechador.robot.espacio_trabajo import giro_de_la_base
    pl = pv.Plotter(off_screen=True, window_size=(ancho * densidad, alto * densidad))
    try:
        escena = Escena(pl, robot)
        escena.mostrar_numeros(numeros)
        escena.dibujar(servo_a_dh(robot, servos))
        if lazos:
            escena.mostrar_superficie(lazos, giro_de_la_base(robot))
            escena.encuadrar(zoom=1.0)
        else:
            escena.encuadrar()
        return pl.screenshot(return_img=True)
    finally:
        pl.close()


def _imagen_de(dibujo, ancho, alto):
    """Un dibujo plano como QImage del tamaño pedido (siempre en su versión «grande»: con ejes y nombres)."""
    dibujo.fig.set_size_inches(ancho / 100, alto / 100)
    dibujo.con_titulo = True
    dibujo.componer()
    dibujo.lienzo.draw()
    rgba = np.ascontiguousarray(np.asarray(dibujo.lienzo.buffer_rgba()))
    return QtGui.QImage(rgba.data, rgba.shape[1], rgba.shape[0], 4 * rgba.shape[1], QtGui.QImage.Format_RGBA8888).copy()


def hoja(geo, servos, lazos, volumen_visible):
    """La hoja completa de la ficha (ANCHO_HOJA × ALTO_HOJA): título, lateral, superior, vista 3D y los datos del robot."""
    robot = geo.robot
    imagen = QtGui.QImage(ANCHO_HOJA, ALTO_HOJA, QtGui.QImage.Format_RGB32)
    imagen.fill(QtGui.QColor(T["fondo"]))
    p = QtGui.QPainter(imagen)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.setRenderHint(QtGui.QPainter.TextAntialiasing)
    p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

    def texto(x, y, cadena, px, color, negritas=False, alinear=QtCore.Qt.AlignLeft):
        letra = QtGui.QFont(p.font())
        letra.setPixelSize(px)
        letra.setBold(negritas)
        p.setFont(letra)
        p.setPen(QtGui.QColor(color))
        ancho = p.fontMetrics().horizontalAdvance(cadena)
        p.drawText(QtCore.QPointF(x - ancho if alinear == QtCore.Qt.AlignRight else x, y), cadena)

    texto(40, 54, robot.nombre, 32, T["tinta"], True)
    texto(40, 88, SUBTITULO, 18, T["tinta_suave"])
    for dibujo_cls, rect in ((Lateral, (40, 112, 880, 540)), (Superior, (950, 112, 610, 540))):
        dibujo = dibujo_cls(geo, FigureCanvasAgg, servos)
        dibujo.volumen_visible = bool(volumen_visible)
        p.drawImage(rect[0], rect[1], _imagen_de(dibujo, rect[2], rect[3]))
    p.setPen(QtGui.QColor(T["borde_suave"]))
    p.drawRect(40, 678, 430, 290)
    # SIN NÚMEROS: A 430 × 290 px LOS DE LA REJILLA SE ENCIMAN ENTRE SÍ Y NO SE LEEN
    foto = render_3d(robot, servos, lazos if volumen_visible else None, 430, 290, numeros=False)
    foto = np.ascontiguousarray(foto)
    p.drawImage(40, 678, QtGui.QImage(foto.data, foto.shape[1], foto.shape[0], 3 * foto.shape[1], QtGui.QImage.Format_RGB888).copy())
    texto(500, 700, "Datos del robot", 20, T["tinta"], True)
    construccion, alcance = datos_de_la_ficha(geo)
    for columna, (x0, filas) in enumerate(((500, construccion), (1070, alcance))):
        for k, (nombre, valor) in enumerate(filas):
            y = 736 + 28 * k
            texto(x0, y, nombre, 17, T["tinta_suave"])
            texto(x0 + 480 if columna == 0 else x0 + 490, y, valor, 17, T["tinta"], True, QtCore.Qt.AlignRight)
    if not geo.con_volumen or not volumen_visible:
        texto(1070, 800, "Sin el volumen de trabajo a la vista, no hay más datos de alcance.", 15, T["tinta_suave"])
    texto(40, 988, PIE, 15, T["tinta_suave"])
    texto(1560, 988, f"Generada el {fecha_de_hoy()}", 15, T["tinta_suave"], alinear=QtCore.Qt.AlignRight)
    p.end()
    return imagen


def guardar_hoja(ruta, geo, servos, lazos, volumen_visible):
    """Guarda la hoja como imagen. Devuelve True si quedó escrita."""
    return hoja(geo, servos, lazos, volumen_visible).save(ruta)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LA PESTAÑA «FICHA» (EN PANTALLA) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class PaginaFicha(QtWidgets.QWidget):
    """La hoja «lado a lado»: lateral y superior juntas arriba y la tabla de datos debajo, en dos columnas. La vista 3D no se
    repite aquí (está en su pestaña); la imagen guardada sí la lleva."""

    def __init__(self, geo, padre=None):
        super().__init__(padre)
        self.setObjectName("PAGINA_FICHA")
        self.lateral, self.superior = Lateral(geo), Superior(geo)
        self.dibujos = (self.lateral, self.superior)
        # EL TAMAÑO LO DECIDE EL VISOR, NO EL CONTENIDO: LA FICHA NO AGRANDA EL MÍNIMO DE LA VENTANA
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        rejilla = QtWidgets.QGridLayout(self)
        rejilla.setObjectName("lay_ficha")
        rejilla.setContentsMargins(0, 0, 0, 0)
        rejilla.setSpacing(4)
        for k, (dibujo, peso) in enumerate(zip(self.dibujos, (3, 2))):
            dibujo.lienzo.setMinimumSize(40, 40)
            dibujo.lienzo.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
            rejilla.addWidget(dibujo.lienzo, 0, k)
            rejilla.setColumnStretch(k, peso)
        self.tabla = QtWidgets.QFrame()
        self.tabla.setObjectName("TABLA_FICHA")
        self.lay_tabla = QtWidgets.QGridLayout(self.tabla)
        self.lay_tabla.setContentsMargins(8, 8, 8, 8)
        self.lay_tabla.setHorizontalSpacing(8)
        self.lay_tabla.setVerticalSpacing(2)
        rejilla.addWidget(self.tabla, 1, 0, 1, 2)
        rejilla.setRowStretch(0, 1)
        self.poner_geometria(geo, False)

    def poner_geometria(self, geo, volumen_visible):
        for dibujo in self.dibujos:
            dibujo.poner_geometria(geo, volumen_visible)
        self._llenar_tabla(geo)

    def _llenar_tabla(self, geo):
        lt = self.lay_tabla
        while lt.count():
            viejo = lt.takeAt(0).widget()
            viejo.hide()                   # SI NO, SIGUE EN EL ÁRBOL Y A LA VISTA HASTA QUE QT LO BORRA (Y SE VERÍA ENCIMADO)
            viejo.setParent(None)
            viejo.deleteLater()
        construccion, alcance = datos_de_la_ficha(geo)
        lt.setColumnMinimumWidth(2, 14)
        for columna, (titulo, filas) in enumerate((("CONSTRUCCIÓN", construccion), ("ALCANCE", alcance))):
            c0 = 3 * columna
            encabezado = QtWidgets.QLabel(titulo)
            encabezado.setProperty("rol", "subtitulo")
            lt.addWidget(encabezado, 0, c0, 1, 2)
            for k, (nombre, valor) in enumerate(filas, start=1):
                etiqueta, dato = QtWidgets.QLabel(nombre), QtWidgets.QLabel(valor)
                etiqueta.setProperty("rol", "nota")
                dato.setProperty("rol", "dato")
                dato.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                lt.addWidget(etiqueta, k, c0)
                lt.addWidget(dato, k, c0 + 1)
            lt.setColumnStretch(c0, 1)
        if not geo.con_volumen:
            nota = QtWidgets.QLabel(SIN_VOLUMEN)
            nota.setProperty("rol", "nota")
            nota.setWordWrap(True)
            # LA NOTA SE REPARTE EN LOS RENGLONES QUE LE SOBRAN A «ALCANCE»: UNA SOLA CELDA ESTIRARÍA EL RENGLÓN DE LA OTRA COLUMNA
            lt.addWidget(nota, len(alcance) + 1, 3, max(len(construccion) - len(alcance), 1), 2, QtCore.Qt.AlignTop)
        lt.setRowStretch(max(len(construccion), len(alcance)) + 2, 1)   # LOS RENGLONES NO SE ESTIRAN: EL SOBRANTE VA ABAJO
        for hijo in self.tabla.findChildren(QtWidgets.QWidget):    # LO MISMO QUE HACE tema.aplicar() CON LO QUE YA EXISTE
            hijo.setAttribute(QtCore.Qt.WA_LayoutUsesWidgetRect)
