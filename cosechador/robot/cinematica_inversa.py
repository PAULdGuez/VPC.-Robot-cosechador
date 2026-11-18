"""Cinemática inversa: posición del efector -> ángulos.

Cada topología tiene su solver; el YAML del robot dice cuál usar en 'cinematica_inversa'.
Hoy existe el analítico 'antropomorfico_3r' (base vertical + hombro + codo). Un robot sin solver
se carga y se dibuja, pero no puede calcular trayectorias (SinSolver).
"""
import numpy as np

from .cinematica import dh_a_servo, servo_a_dh, servos_en_rango


# A MENOS DE ESTA DISTANCIA DEL EJE DE LA BASE, EL GIRO DE LA BASE ES INDIFERENTE (SE USA EL CENTRO DE SU
# RANGO). CON 1e-6 mm, EL HOME REDONDEADO DE UN ARCHIVO (A 0.0005 mm DEL EJE) ELEGÍA EL BRAZO INCLINADO
# HACIA ATRÁS Y EL JOG DESDE AHÍ FALLABA A LOS 350 mm.
SOBRE_EL_EJE_MM = 0.01


class SinSolver(RuntimeError):
    """El robot no declara un solver de cinemática inversa para su topología."""


class FueraDeAlcance(ValueError):
    """El punto está fuera del alcance geométrico del brazo."""


def _antropomorfico_3r(robot, x, y, z):
    """Hasta 4 soluciones (grados DH), en orden de preferencia:
    base mirando al punto con codo arriba / codo abajo, y luego base girada 180°."""
    j1, j2, j3 = robot.articulaciones
    d1, a2, a3 = j1.d, j2.a, j3.a

    # Distancia plana y altura MEDIDA DESDE EL HOMBRO (encima de la base, a d1)
    r = np.hypot(x, y)
    s = z - d1

    # Ley del coseno para el codo; tolerancia para el brazo totalmente estirado (D = 1.0000000001)
    D = (r**2 + s**2 - a2**2 - a3**2) / (2 * a2 * a3)
    if abs(D) > 1 + 1e-9:
        raise FueraDeAlcance("El punto está fuera del alcance del robot.")
    D = float(np.clip(D, -1.0, 1.0))

    if r < SOBRE_EL_EJE_MM:
        # PUNTO SOBRE EL EJE DE LA BASE (P. EJ. HOME): CUALQUIER GIRO SIRVE. SE USA EL CENTRO DEL
        # RANGO DEL SERVO PARA QUE EL ROBOT (Y LA CAMARA) MIRE A LA MITAD DE SU ZONA DE TRABAJO.
        theta1 = servo_a_dh(robot, (j1.servo.centro, j2.servo.centro, j3.servo.centro))[0]
    else:
        theta1 = np.degrees(np.arctan2(y, x))

    soluciones = []
    for t1, r_plano in ((theta1, r), (theta1 + 180, -r)):
        for signo in (-1, 1):  # -1: codo arriba (theta3 negativo), 1: codo abajo
            t3 = np.arctan2(signo * np.sqrt(1 - D**2), D)
            phi = np.arctan2(a3 * np.sin(t3), a2 + a3 * np.cos(t3))
            t2 = np.arctan2(s, r_plano) - phi
            soluciones.append((float(t1), float(np.degrees(t2)), float(np.degrees(t3))))
    return soluciones


SOLVERS = {"antropomorfico_3r": _antropomorfico_3r}


def soluciones(robot, punto):
    """Todas las soluciones DH para llegar al punto. Lanza SinSolver o FueraDeAlcance."""
    solver = SOLVERS.get(robot.cinematica_inversa)
    if solver is None:
        raise SinSolver(f"'{robot.nombre}' no tiene solver de cinemática inversa.")
    return solver(robot, *punto)


def angulos_servo(robot, punto):
    """Ángulos de servo para llegar al punto, o None si el robot real no puede
    (fuera del brazo, o ningún acomodo cabe en el rango de los servos)."""
    try:
        candidatas = soluciones(robot, punto)
    except FueraDeAlcance:
        return None
    for q in candidatas:
        servos = dh_a_servo(robot, q)
        if servos_en_rango(robot, servos):
            return servos
    return None


def alcanzable(robot, punto):
    """True si el robot real puede llegar al punto (alcance + límites de los servos)."""
    return angulos_servo(robot, punto) is not None


def salto(servos_a, servos_b):
    """Lo más que gira cualquier servo para pasar de una pose a otra (grados)."""
    return max(abs(a - b) for a, b in zip(servos_a, servos_b))


def angulos_servo_cercanos(robot, punto, servos_actuales):
    """Como angulos_servo, pero entre las soluciones válidas elige la MÁS CERCANA a la pose actual.

    Mantiene la postura (codo arriba/abajo, base de frente/volteada) mientras se pueda; así un
    movimiento continuo no cambia de solución a medio camino. None si el robot real no puede llegar.
    """
    try:
        candidatas = soluciones(robot, punto)
    except FueraDeAlcance:
        return None
    validas = [s for s in (dh_a_servo(robot, q) for q in candidatas) if servos_en_rango(robot, s)]
    return min(validas, key=lambda s: salto(s, servos_actuales)) if validas else None
