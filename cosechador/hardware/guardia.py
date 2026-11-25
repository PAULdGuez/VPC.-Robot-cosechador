"""Guardia de salida: nada llega a la placa sin que `revisar_paso` (invariantes.py) lo haya dado por bueno.

Envuelve al Enlace y se usa igual que él (`enviar(*servos)`), con una diferencia que no se puede ignorar por
descuido: si la orden no es válida NO devuelve un resultado raro, LEVANTA `OrdenNegada`. Quien no la atrape no
sigue como si nada.

- Una sola decisión, la de invariantes.revisar_paso: aquí no hay reglas propias, solo el estado (lo último que SALIÓ,
  a qué hora y la deuda de ritmo) y el candado.
- GS5: tras negar, queda DETENIDA: toda orden siguiente se niega, con el MISMO motivo y la MISMA clave que la primera,
  hasta que una persona llame `reiniciar()`.
- GS8: la cuenta de lo último que salió es suya; no le pregunta a la app dónde está el robot.
- GS9: si la revisión misma falla (cualquier excepción), no sale nada: se niega con clave `guardia_fallo`.
- Si la placa falla al enviar, tampoco se sabe qué llegó: se detiene con clave `enlace_fallo`.
- La hora viene de `reloj` (inyectable, como en grabadora.py y registro.py); nunca lee la del sistema por su cuenta.
- Sin hilos: se usa desde el hilo de la interfaz, igual que el Enlace.
"""
import logging
import math
import numbers
import time

from cosechador.movimiento.invariantes import Violacion, revisar_paso

_logger = logging.getLogger(__name__)


class OrdenNegada(Exception):
    """La guardia no dejó salir la orden. `violacion` trae clave estable, motivo en español y los datos.

    nueva: True la primera vez (la que causó la detención); False cuando solo es la guardia detenida negando
    de nuevo lo mismo (para que la pantalla avise una sola vez por clave).
    """

    def __init__(self, violacion, nueva=True):
        super().__init__(violacion.motivo)
        self.violacion = violacion
        self.nueva = nueva

    @property
    def clave(self):
        return self.violacion.clave

    @property
    def motivo(self):
        return self.violacion.motivo


class GuardiaDeSalida:
    def __init__(self, enlace, robot, reloj=time.monotonic):
        self._enlace = enlace
        self.robot = robot
        self.reloj = reloj
        self._ultima = None        # LOS SERVOS DE LA ÚLTIMA ORDEN QUE SALIÓ (None: AÚN NO HA SALIDO NINGUNA)
        self._hora_ultima = None   # A QUÉ HORA SALIÓ
        self._deuda_ms = 0.0       # DEUDA DE RITMO (invariantes._revisar_ritmo)
        self._detenida = None      # LA Violacion QUE LA DETUVO (None: FUNCIONANDO)

    # -- lo que la interfaz puede leer ---------------------------------------------------------------------------
    @property
    def nombre(self):
        return self._enlace.nombre

    @property
    def dispositivo(self):
        return self._enlace.dispositivo

    @property
    def detenida(self):
        return self._detenida is not None

    @property
    def violacion(self):
        """Por qué está detenida (None si no lo está)."""
        return self._detenida

    @property
    def ultima(self):
        """Los servos de lo último que salió, o None si todavía no ha salido nada."""
        return self._ultima

    # -- operación -----------------------------------------------------------------------------------------------
    def enviar(self, *servos):
        """Como Enlace.enviar: devuelve (comando, respuestas). Si no es válida levanta OrdenNegada."""
        if self._detenida is not None:
            raise OrdenNegada(self._detenida, nueva=False)
        try:
            ahora = self._leer_reloj()
            dt = None if self._hora_ultima is None else ahora - self._hora_ultima
            revision = revisar_paso(self.robot, servos, self._ultima, dt, self._deuda_ms)
            violacion, saneados, deuda = revision.violacion, revision.servos, revision.deuda_ms
            if violacion is None and not (isinstance(saneados, tuple) and len(saneados) == self.robot.gdl):
                raise RuntimeError(f"revisar_paso dio por buena una orden sin servos saneados: {saneados!r}")
        except Exception as error:  # noqa: BLE001 -- A PROPÓSITO AMPLIO: SI LA GUARDIA FALLA, NO SALE NADA (GS9)
            _logger.exception("La guardia de salida falló al revisar la orden: no sale nada")
            violacion = Violacion("guardia_fallo", "La guardia de salida falló al revisar la orden: no se mandó nada "
                                                   "al robot.", detalle=f"{type(error).__name__}: {error}")
            self._detenida = violacion
            raise OrdenNegada(violacion) from error
        if violacion is not None:
            self._detenida = violacion
            raise OrdenNegada(violacion)
        try:
            respuesta = self._enlace.enviar(*saneados)
        except Exception as error:
            # NO SE SABE SI ALGO LLEGÓ A LA PLACA: SE DETIENE Y SE CONSERVA LA ÚLTIMA ORDEN QUE SÍ SE CONFIRMÓ
            _logger.exception("La placa falló al recibir la orden: la guardia queda detenida")
            violacion = Violacion("enlace_fallo", "La placa no recibió la orden bien: la guardia se detuvo.",
                                  detalle=f"{type(error).__name__}: {error}")
            self._detenida = violacion
            raise OrdenNegada(violacion) from error
        self._ultima, self._hora_ultima, self._deuda_ms = saneados, ahora, deuda
        return respuesta

    def reiniciar(self):
        """Lo único que levanta el candado, y solo una persona lo llama. NO manda nada a la placa (ni mueve el robot
        ni reenvía la orden negada): conserva lo último que salió, su hora y su deuda."""
        self._detenida = None

    def close(self):
        self._enlace.close()

    def _leer_reloj(self):
        ahora = self.reloj()
        if isinstance(ahora, bool) or not isinstance(ahora, numbers.Real) or not math.isfinite(ahora):
            raise ValueError(f"El reloj devolvió {ahora!r}, que no es una hora válida.")
        return float(ahora)
