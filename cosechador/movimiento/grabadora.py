"""Grabación de movimientos (pura: sin Qt). Bloque 2 de F1.

Se graba lo que DE VERDAD se le manda a los servos, no lo que dicen los sliders: la ventana llama a
`anotar` justo donde envía cada pose a la placa.

Qué significa cada muestra: «a los t segundos, el robot ESTÁ en esta pose». Por eso una pose enviada se
anota en el momento en que termina su paso (envío + duración del paso), y si el robot se quedó quieto
antes de moverse, primero se anota la pose anterior en el momento del envío: así la reproducción
(`trayectorias.plan_grabacion`) ve la pausa como espera y no como un movimiento lentísimo.

La grabación empieza con el primer envío y termina con el último: lo que pasa antes de mover por primera
vez y después de mover por última vez es buscar el botón, no parte del movimiento.
"""
import time

from cosechador.movimiento.programa import Muestra, Programa
from cosechador.movimiento.trayectorias import UMBRAL_QUIETO
from cosechador.robot.cinematica_inversa import salto


class Grabadora:
    """reloj: función que da segundos (por omisión time.monotonic; en pruebas, uno falso)."""

    def __init__(self, reloj=time.monotonic):
        self.reloj = reloj
        self._inicial = None
        self._muestras = None
        self._t0 = None

    @property
    def grabando(self):
        return self._muestras is not None

    def empezar(self, servos):
        """Empieza a grabar desde la pose en la que está el robot."""
        self._inicial = tuple(float(s) for s in servos)
        self._muestras = []
        self._t0 = None

    def anotar(self, servos, duracion_s):
        """Se acaba de mandar esta pose a los servos; su paso dura duracion_s."""
        if not self.grabando:
            return
        ahora = self.reloj()
        if self._t0 is None:  # PRIMER ENVÍO: AQUÍ EMPIEZA LA GRABACIÓN, DESDE LA POSE INICIAL
            self._t0 = ahora
            self._muestras.append(Muestra(0.0, self._inicial))
        t = ahora - self._t0
        anterior = self._muestras[-1]
        if t - anterior.t > duracion_s:  # SE QUEDÓ QUIETO MÁS DE UN PASO: ESA PAUSA TAMBIÉN SE GRABA
            self._muestras.append(Muestra(t, anterior.servos))
        # NUNCA HACIA ATRÁS (EL TEMPORIZADOR REDONDEA A ms): EL ARCHIVO EXIGE TIEMPOS CRECIENTES
        self._muestras.append(Muestra(max(t, anterior.t) + duracion_s, tuple(float(s) for s in servos)))

    @property
    def muestras(self):
        """Lo grabado hasta ahora, o None si el robot no se ha movido (no hay nada que guardar)."""
        muestras = self._muestras
        if not muestras or all(salto(m.servos, muestras[0].servos) < UMBRAL_QUIETO for m in muestras):
            return None
        return tuple(muestras)

    def resumen(self):
        """(cuántas muestras, cuántos segundos) lleva la grabación: para decir cuánto se perdería al descartarla."""
        muestras = self.muestras
        return (len(muestras), muestras[-1].t) if muestras else (0, 0.0)

    def reanudar(self):
        """Se sigue grabando después de una interrupción (el diálogo del nombre): ese rato no es parte de la grabación."""
        if self._muestras and self._t0 is not None:
            self._t0 = self.reloj() - self._muestras[-1].t

    def terminar(self):
        """Deja de grabar. Devuelve las muestras, o None si el robot no se movió (no hay nada que guardar)."""
        muestras = self.muestras
        self._muestras, self._t0 = None, None
        return muestras


def programa_de_grabacion(muestras, nombre, robot):
    return Programa(nombre, "grabacion", robot.nombre, robot.gdl, muestras=tuple(muestras))
