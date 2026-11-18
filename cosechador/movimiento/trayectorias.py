"""Planeación de movimientos. Pura: devuelve la lista de pasos; quien la ejecuta (la interfaz,
con un temporizador) solo manda cada paso a los servos y muestra los avisos.

Un movimiento en línea recta se comprueba COMPLETO antes de aceptarlo: si un solo punto
intermedio es inalcanzable, el movimiento entero se rechaza.
"""
from dataclasses import dataclass

import numpy as np

from cosechador.robot.cinematica import cinematica_directa, servo_a_dh
from cosechador.robot.cinematica_inversa import alcanzable, angulos_servo

PASOS_POR_TRAMO = 20


@dataclass(frozen=True)
class Paso:
    """Un paso del plan: ángulos de servo a los que ir, un aviso para la persona, o ambos."""
    servos: tuple | None = None
    aviso: str | None = None


def posicion(robot, servos):
    return cinematica_directa(robot, servo_a_dh(robot, servos))


def tramo_recto(robot, origen, destino, pasos=PASOS_POR_TRAMO):
    """Servos de cada punto de la recta origen -> destino, o None si alguno es inalcanzable."""
    origen, destino = np.asarray(origen, dtype=float), np.asarray(destino, dtype=float)
    servos = [angulos_servo(robot, origen + (destino - origen) * k) for k in np.linspace(0, 1, pasos)]
    return None if any(s is None for s in servos) else servos


def _texto(punto):
    return str(tuple(int(round(c)) for c in punto))


def _ir(robot, origen, destino, pasos, via_home):
    """Tramo(s) para llegar a destino. Si la recta cruza una zona inalcanzable (p. ej. el eje de la
    base) y via_home, prueba pasar por HOME. Devuelve (servos, posicion_final) o (None, origen)."""
    tramo = tramo_recto(robot, origen, destino, pasos)
    if tramo is not None:
        return tramo, posicion(robot, tramo[-1])
    if via_home:
        a_home = tramo_recto(robot, origen, robot.home, pasos)
        if a_home is not None:
            desde_home = tramo_recto(robot, posicion(robot, a_home[-1]), destino, pasos)
            if desde_home is not None:
                return a_home + desde_home, posicion(robot, desde_home[-1])
    return None, origen


def plan_ir_y_volver(robot, origen, destino, pasos=PASOS_POR_TRAMO):
    """Ir al destino en línea recta y regresar a HOME (botón CALCULAR)."""
    if not alcanzable(robot, destino):
        return [Paso(aviso=f"Punto fuera del volumen de trabajo: {_texto(destino)}")]
    ida, llegada = _ir(robot, origen, destino, pasos, via_home=False)
    if ida is None:
        return [Paso(aviso=f"El camino hacia {_texto(destino)} pasa por una zona que el robot no alcanza.")]
    regreso, _ = _ir(robot, llegada, robot.home, pasos, via_home=False)
    plan = [Paso(servos=s) for s in ida] + [Paso(servos=s) for s in (regreso or [])]
    return plan + [Paso(aviso=f"Punto {_texto(destino)} alcanzado; de regreso en HOME.")]


def plan_cosecha(robot, origen, puntos, pasos=PASOS_POR_TRAMO):
    """Visitar cada punto (pasando por HOME si hace falta) y terminar en HOME (botón COSECHAR).

    Devuelve (plan, cosechados).
    """
    plan, cosechados, actual = [], 0, np.asarray(origen, dtype=float)
    for i, punto in enumerate(puntos, start=1):
        plan.append(Paso(aviso=f"Cosechando {i}/{len(puntos)}: {_texto(punto)}"))
        if not alcanzable(robot, punto):
            plan.append(Paso(aviso=f"Punto fuera del volumen de trabajo: {_texto(punto)}"))
            continue
        tramo, llegada = _ir(robot, actual, punto, pasos, via_home=True)
        if tramo is None:
            plan.append(Paso(aviso=f"El camino hacia {_texto(punto)} pasa por una zona que el robot no alcanza."))
            continue
        plan.extend(Paso(servos=s) for s in tramo)
        cosechados, actual = cosechados + 1, llegada
    regreso, _ = _ir(robot, actual, robot.home, pasos, via_home=False)
    plan.extend(Paso(servos=s) for s in (regreso or []))
    plan.append(Paso(aviso=f"Cosecha terminada: {cosechados} de {len(puntos)} puntos alcanzados."))
    return plan, cosechados
