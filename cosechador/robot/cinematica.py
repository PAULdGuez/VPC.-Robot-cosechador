"""Cinemática directa de cualquier brazo serial descrito con Denavit-Hartenberg, y el mapeo
entre ángulos DH y ángulos de servo. Pura: solo numpy, sin Qt ni PyVista.

Todos los ángulos en grados y las distancias en mm. `q` es la tupla de ángulos DH (theta_i).
"""
import numpy as np


def envolver(angulo, centro=0.0):
    """Lleva un ángulo al intervalo [centro - 180, centro + 180)."""
    return ((angulo - centro + 180) % 360) - 180 + centro


def rot_z(grados):
    c, s = np.cos(np.radians(grados)), np.sin(np.radians(grados))
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)


def rot_x(grados):
    c, s = np.cos(np.radians(grados)), np.sin(np.radians(grados))
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=float)


def traslacion(x=0.0, y=0.0, z=0.0):
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


def matriz_dh(theta, d, a, alpha):
    """Transformación de un eslabón: Rz(theta) · Tz(d) · Tx(a) · Rx(alpha)."""
    return rot_z(theta) @ traslacion(z=d) @ traslacion(x=a) @ rot_x(alpha)


def matrices(robot, q):
    """Matrices acumuladas [T_0, T_01, T_02, ..., T_0n]: la base y el marco de cada articulación."""
    if len(q) != robot.gdl:
        raise ValueError(f"Se esperaban {robot.gdl} ángulos y llegaron {len(q)}.")
    acumulada = [np.eye(4)]
    for theta, j in zip(q, robot.articulaciones):
        acumulada.append(acumulada[-1] @ matriz_dh(theta, j.d, j.a, j.alpha))
    return acumulada


def cinematica_directa(robot, q):
    """Posición (x, y, z) del efector en mm."""
    return matrices(robot, q)[-1][:3, 3]


def pose_efector(robot, q):
    """Matriz 4x4 completa del efector (posición y orientación)."""
    return matrices(robot, q)[-1]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SERVOS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def dh_a_servo(robot, q):
    """Ángulos DH -> ángulos de servo: servo = offset + signo * theta.

    Se envuelve alrededor del CENTRO del rango de cada servo: así el ruido numérico (-1e-10°)
    queda como ~0° y no como 359.9999° (fuera de rango).
    """
    return tuple(envolver(j.servo.offset + j.servo.signo * float(theta), j.servo.centro)
                 for theta, j in zip(q, robot.articulaciones))


def servo_a_dh(robot, servos):
    """Ángulos de servo -> ángulos DH (inversa de dh_a_servo)."""
    return tuple(j.servo.signo * (float(s) - j.servo.offset) for s, j in zip(servos, robot.articulaciones))


def servos_en_rango(robot, servos, tolerancia=1e-6):
    return all(j.servo.minimo - tolerancia <= s <= j.servo.maximo + tolerancia
               for s, j in zip(servos, robot.articulaciones))
