"""Slider que enseña hasta dónde se puede llegar: lo que la punta no alcanza desde la pose actual sale en rojo.

El slider no sabe nada del robot: la ventana le dice los tramos alcanzables (en las mismas unidades del slider) y él
los pinta encima de su pista y no deja que la persona los salga. Se declara en el .ui como widget propio de QSlider,
así que el tema (cosechador/ui/tema.py) lo viste igual que a los demás.
"""
import math

from PyQt5 import QtCore, QtGui, QtWidgets

from cosechador.robot.cinematica_inversa import TOLERANCIA_TRAMOS_MM
from cosechador.ui import tema


class SliderConAlcance(QtWidgets.QSlider):

    def __init__(self, *args):
        super().__init__(*args)
        self.tramos = None  # [(desde, hasta), ...] ALCANZABLES; None = NO SE SABE: NO SE PINTA NADA NI SE FRENA
        self.color_fuera = QtGui.QColor(tema.TOKENS["rojo"])

    def poner_tramos(self, tramos):
        self.tramos = None if tramos is None else [tuple(t) for t in tramos]
        self.update()

    def acotar(self, valor, desde):
        """El valor más cercano a `valor` que sigue dentro del tramo donde está `desde` (la punta ahora).

        Así el slider se detiene en la frontera: no entra al rojo ni salta al tramo de al lado. Si no hay tramos, o
        `desde` no está en ninguno, el valor pasa tal cual (no hay de dónde frenar).
        """
        for a, b in self.tramos or ():
            if a - TOLERANCIA_TRAMOS_MM <= desde <= b + TOLERANCIA_TRAMOS_MM:
                # EL SLIDER ES ENTERO: LA FRONTERA ES EL ENTERO MÁS CERCANO QUE QUEDA DENTRO DEL TRAMO
                minimo, maximo = math.ceil(a), math.floor(b)
                return valor if minimo > maximo else min(max(valor, minimo), maximo)
        return valor

    def _rojos(self):
        """Los intervalos (en unidades del slider) que quedan fuera de todo tramo."""
        rojos, desde = [], self.minimum()
        for a, b in sorted(self.tramos):
            if a > desde:
                rojos.append((desde, a))
            desde = max(desde, b)
        if desde < self.maximum():
            rojos.append((desde, self.maximum()))
        return rojos

    def paintEvent(self, evento):
        super().paintEvent(evento)
        if not self.tramos or self.maximum() <= self.minimum():
            return
        opcion = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opcion)
        estilo = self.style()
        pista = estilo.subControlRect(QtWidgets.QStyle.CC_Slider, opcion, QtWidgets.QStyle.SC_SliderGroove, self)
        manija = estilo.subControlRect(QtWidgets.QStyle.CC_Slider, opcion, QtWidgets.QStyle.SC_SliderHandle, self)
        pista = QtCore.QRectF(pista).adjusted(1, 1, -1, -1)  # DENTRO DEL BORDE DE LA PISTA
        recorrido = pista.width() + 2 - manija.width()      # LA PERILLA NO RECORRE TODO EL ANCHO: SU CENTRO VA DE MEDIA PERILLA A MEDIA PERILLA

        def x(valor):
            return pista.left() - 1 + manija.width() / 2 + (valor - self.minimum()) / (self.maximum() - self.minimum()) * recorrido

        forma = QtGui.QPainterPath()
        forma.addRoundedRect(pista, 3, 3)
        perilla = QtGui.QPainterPath()
        perilla.addEllipse(QtCore.QRectF(manija))
        pintor = QtGui.QPainter(self)
        pintor.setRenderHint(QtGui.QPainter.Antialiasing)
        pintor.setClipPath(forma.subtracted(perilla))  # EL ROJO VA POR DEBAJO DE LA PERILLA
        for desde, hasta in self._rojos():
            izquierda = pista.left() if desde <= self.minimum() else x(desde)
            derecha = pista.right() if hasta >= self.maximum() else x(hasta)
            pintor.fillRect(QtCore.QRectF(izquierda, pista.top(), derecha - izquierda, pista.height()), self.color_fuera)
        pintor.end()
