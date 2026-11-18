"""Volumen de trabajo: los puntos que alcanza el efector recorriendo el rango de cada servo."""
import itertools

import numpy as np

from .cinematica import cinematica_directa, servo_a_dh


def rangos_servo(robot, paso):
    return [np.arange(j.servo.minimo, j.servo.maximo + 1e-9, paso) for j in robot.articulaciones]


def total_puntos(robot, paso=5):
    return int(np.prod([len(r) for r in rangos_servo(robot, paso)]))


def barrer(robot, paso=5, progreso=None):
    """Barre todas las combinaciones de servos (cada `paso` grados).

    progreso(hechos, total) se llama de vez en cuando, para una barra de progreso.
    """
    rangos = rangos_servo(robot, paso)
    total = total_puntos(robot, paso)
    puntos = np.empty((total, 3))
    aviso_cada = max(total // 100, 1)
    for i, servos in enumerate(itertools.product(*rangos)):
        puntos[i] = cinematica_directa(robot, servo_a_dh(robot, servos))
        if progreso is not None and (i + 1) % aviso_cada == 0:
            progreso(i + 1, total)
    if progreso is not None:
        progreso(total, total)
    return puntos
