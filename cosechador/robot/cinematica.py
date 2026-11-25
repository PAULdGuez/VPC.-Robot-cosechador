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
    """Posición (x, y, z) en mm de la PUNTA de la herramienta: lo que se pide y se muestra por coordenadas.

    La herramienta prolonga el último eslabón: su punta queda `robot.efector.largo` mm más allá del extremo
    de ese eslabón (pose_efector), sobre su eje X. Sin herramienta (largo 0) son el mismo punto.
    """
    return (pose_efector(robot, q) @ (robot.efector.largo, 0.0, 0.0, 1.0))[:3]


def pose_efector(robot, q):
    """Matriz 4x4 (posición y orientación) del extremo del último eslabón: ahí se atornilla el efector, y la
    cámara si va montada en él. NO es la punta de la herramienta: esa la da cinematica_directa."""
    return matrices(robot, q)[-1]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SERVOS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# UN SERVO RECIBE GRADOS ENTEROS: UNA CENTÉSIMA DE GRADO FUERA DE SU RANGO ES RUIDO NUMÉRICO, NO UN
# LÍMITE REAL. (CON 1e-6 °, EL REDONDEO DE UN PROGRAMA GUARDADO DECLARABA INALCANZABLE UN PUNTO A 350 mm.)
TOLERANCIA_SERVO = 0.01


def _recortar(valor, servo):
    """Si el ángulo se sale del rango por menos de TOLERANCIA_SERVO, se lleva al borde."""
    if servo.minimo - TOLERANCIA_SERVO <= valor < servo.minimo:
        return servo.minimo
    if servo.maximo < valor <= servo.maximo + TOLERANCIA_SERVO:
        return servo.maximo
    return valor


def dh_a_servo(robot, q):
    """Ángulos DH -> ángulos de servo: servo = offset + signo * theta.

    Se envuelve alrededor del CENTRO del rango de cada servo: así el ruido numérico (-1e-10°)
    queda como ~0° y no como 359.9999° (fuera de rango). Lo que se sale por menos de
    TOLERANCIA_SERVO se recorta al borde, para que lo que llega a la placa siempre esté en rango.
    """
    return tuple(_recortar(envolver(j.servo.offset + j.servo.signo * float(theta), j.servo.centro), j.servo)
                 for theta, j in zip(q, robot.articulaciones))


def servo_a_dh(robot, servos):
    """Ángulos de servo -> ángulos DH (inversa de dh_a_servo)."""
    return tuple(j.servo.signo * (float(s) - j.servo.offset) for s, j in zip(servos, robot.articulaciones))


def primer_servo_fuera(robot, servos, tolerancia=TOLERANCIA_SERVO):
    """Índice (desde 0) del primer servo fuera de su rango, o None si todos caben. Un solo lugar decide el rango."""
    for i, (s, j) in enumerate(zip(servos, robot.articulaciones)):
        if not j.servo.minimo - tolerancia <= s <= j.servo.maximo + tolerancia:
            return i
    return None


def servos_en_rango(robot, servos, tolerancia=TOLERANCIA_SERVO):
    return primer_servo_fuera(robot, servos, tolerancia) is None
