"""Piezas compartidas por las pruebas de la guardia de salida y de INV1 (no son pruebas)."""
import functools
import os

from hypothesis import HealthCheck, settings

from cosechador.robot import cargar_robot

CARPETA_PRUEBAS = os.path.dirname(os.path.abspath(__file__))
CARPETA_REPO = os.path.dirname(CARPETA_PRUEBAS)
NOMBRES_ROBOT = ("3gdl", "4gdl")

# DETERMINISTA: SIN BASE DE EJEMPLOS Y CON LA SEMILLA DERIVADA DE LA PRUEBA, ASÍ LA SUITE DA LO MISMO SIEMPRE
# Y NO DEJA UNA CARPETA .hypothesis/ CON EJEMPLOS GUARDADOS
PROPIEDADES = settings(derandomize=True, database=None, deadline=None, max_examples=150,
                       suppress_health_check=[HealthCheck.too_slow])


@functools.lru_cache(maxsize=None)
def robot_de(nombre):
    if nombre == "3gdl":
        return cargar_robot()
    return cargar_robot(os.path.join(CARPETA_PRUEBAS, "robots", "brazo_4gdl.yaml"))


class Reloj:
    """Reloj falso: la hora la mueve la prueba."""

    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class EnlaceFalso:
    """Lo único que la guardia le pide al enlace: enviar. Anota lo que de verdad «salió» y a qué hora."""

    nombre = "enlace falso"

    def __init__(self, reloj=None):
        self.reloj = reloj
        self.salidas = []        # [(hora, servos)]
        self.cerrado = False
        self.dispositivo = object()

    def enviar(self, *servos):
        self.salidas.append((self.reloj() if self.reloj else None, servos))
        return "comando", []

    def close(self):
        self.cerrado = True

    @property
    def servos(self):
        return [s for _, s in self.salidas]


def punto_medio(robot):
    return tuple((j.servo.minimo + j.servo.maximo) / 2 for j in robot.articulaciones)
