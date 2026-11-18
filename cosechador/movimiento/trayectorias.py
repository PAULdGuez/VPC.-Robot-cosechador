"""Planeación de movimientos. Pura: devuelve la lista de pasos; quien la ejecuta (la interfaz,
con un temporizador) solo manda cada paso a los servos y muestra los avisos.

Dos reglas para cada movimiento en línea recta:
1. ALCANCE: todo punto del camino debe ser alcanzable.
2. CONTINUIDAD: entre dos pasos seguidos ningún servo gira más de `movimiento.salto_max_por_paso`
   (lo que el servo alcanza a girar en un paso). Si un tramo pide más, se parte a la mitad, una y
   otra vez: si así desaparece, solo era velocidad. Si el salto sigue con los puntos a < 0.5 mm, el
   robot tiene que CAMBIAR DE POSTURA (p. ej. voltear la base 180° para cruzar y = 0, o girarla
   sobre su propio eje): según `al_cambiar_postura`, se hace "despacio" (servos poco a poco,
   avisando) o se "niega".
"""
import math
from dataclasses import dataclass

import numpy as np

from cosechador.robot.cinematica import cinematica_directa, servo_a_dh
from cosechador.robot.cinematica_inversa import alcanzable, angulos_servo_cercanos, salto

PASOS_POR_TRAMO = 20          # MUESTRAS MÍNIMAS DE CADA RECTA (ANTES DE PARTIR POR CONTINUIDAD)
DISTANCIA_MINIMA_MM = 0.5     # POR DEBAJO DE ESTO, UN SALTO GRANDE YA NO ES VELOCIDAD: ES CAMBIO DE POSTURA


@dataclass(frozen=True)
class Paso:
    """Un paso del plan: ángulos de servo a los que ir, un aviso para la persona, o ambos."""
    servos: tuple | None = None
    aviso: str | None = None


class _NoSePuede(Exception):
    def __init__(self, motivo):
        super().__init__(motivo)
        self.motivo = motivo


def posicion(robot, servos):
    return cinematica_directa(robot, servo_a_dh(robot, servos))


def servos_del_plan(plan):
    return [p.servos for p in plan if p.servos is not None]


def _texto(punto):
    return str(tuple(int(round(c)) for c in punto))


def _cambio_de_postura(robot, servos_a, servos_b, cerca_de, postura):
    # SE GIRA EN ESPACIO DE SERVOS A LA MITAD DEL SALTO MÁXIMO POR PASO: DESPACIO Y CONTINUO
    n = max(2, math.ceil(salto(servos_a, servos_b) / (robot.movimiento.salto_max_por_paso / 2)))
    a, b = np.asarray(servos_a, float), np.asarray(servos_b, float)
    intermedios = [tuple(a + (b - a) * k) for k in np.linspace(0, 1, n + 1)[1:]]
    # ¿LA PUNTA SE MUEVE MIENTRAS TANTO? SI NO (P. EJ. GIRAR LA BASE CON LA PUNTA SOBRE SU EJE, COMO EN
    # HOME) ES UN GIRO EN EL LUGAR: INOFENSIVO, SE PERMITE SIEMPRE. SI SÍ, EL BRAZO BARRE EL ESPACIO.
    desvio = max(np.linalg.norm(posicion(robot, s) - cerca_de) for s in intermedios)
    pasos = [Paso(servos=s) for s in intermedios]
    if desvio < 1.0:
        return [Paso(aviso=f"Giro en el lugar cerca de {_texto(cerca_de)} (la punta no se mueve).")] + pasos
    if postura == "negar":
        raise _NoSePuede(f"Para pasar por {_texto(cerca_de)} el robot tendría que cambiar de postura "
                         f"(el brazo barrería hasta {desvio:.0f} mm fuera de la línea) y el modo es «negar».")
    aviso = (f"Cambio de postura cerca de {_texto(cerca_de)}: los servos giran despacio "
             f"(la punta sale hasta {desvio:.0f} mm de la línea recta).")
    return [Paso(aviso=aviso)] + pasos


def _refinar(robot, pa, sa, pb, postura):
    """Pasos para ir de pa (con servos sa) a pb respetando el salto máximo por paso."""
    sb = angulos_servo_cercanos(robot, pb, sa)
    if sb is None:
        raise _NoSePuede(f"pasa por {_texto(pb)}, una zona que el robot no alcanza")
    if salto(sa, sb) <= robot.movimiento.salto_max_por_paso:
        return [Paso(servos=sb)]
    if np.linalg.norm(pb - pa) < DISTANCIA_MINIMA_MM:
        return _cambio_de_postura(robot, sa, sb, pb, postura)
    medio = (pa + pb) / 2
    izquierda = _refinar(robot, pa, sa, medio, postura)
    return izquierda + _refinar(robot, medio, servos_del_plan(izquierda)[-1], pb, postura)


def tramo_continuo(robot, servos_inicio, destino, postura=None, pasos=PASOS_POR_TRAMO):
    """Recta desde la pose actual hasta destino. Devuelve (pasos, None) o (None, motivo)."""
    postura = postura or robot.movimiento.al_cambiar_postura
    origen = posicion(robot, servos_inicio)
    destino = np.asarray(destino, dtype=float)
    plan, actual_p, actual_s = [], origen, tuple(servos_inicio)
    try:
        for k in np.linspace(0, 1, pasos)[1:]:
            siguiente = origen + (destino - origen) * k
            parte = _refinar(robot, actual_p, actual_s, siguiente, postura)
            plan.extend(parte)
            actual_p, actual_s = siguiente, servos_del_plan(parte)[-1]
    except _NoSePuede as e:
        return None, e.motivo
    return plan, None


def tramo_recto(robot, origen, destino, pasos=PASOS_POR_TRAMO):
    """Servos de la recta origen -> destino (desde la postura de angulos_servo en el origen), o None."""
    from cosechador.robot.cinematica_inversa import angulos_servo
    inicio = angulos_servo(robot, origen)
    if inicio is None:
        return None
    plan, _ = tramo_continuo(robot, inicio, destino, pasos=pasos)
    return None if plan is None else [inicio] + servos_del_plan(plan)


def _ir(robot, servos, destino, postura, via_home):
    """Pasos para llegar a destino desde `servos`. Si la recta no sirve y via_home, prueba pasar por
    HOME. Devuelve (pasos, servos_finales, motivo)."""
    plan, motivo = tramo_continuo(robot, servos, destino, postura)
    if plan is not None:
        return plan, servos_del_plan(plan)[-1] if servos_del_plan(plan) else servos, None
    if via_home:
        a_home, _ = tramo_continuo(robot, servos, robot.home, postura)
        if a_home is not None:
            en_home = servos_del_plan(a_home)[-1] if servos_del_plan(a_home) else servos
            desde_home, _ = tramo_continuo(robot, en_home, destino, postura)
            if desde_home is not None:
                return a_home + desde_home, servos_del_plan(desde_home)[-1], None
    return None, servos, motivo


def plan_ir_y_volver(robot, servos, destino, postura=None):
    """Ir al destino en línea recta y regresar a HOME (botón CALCULAR)."""
    if not alcanzable(robot, destino):
        return [Paso(aviso=f"Punto fuera del volumen de trabajo: {_texto(destino)}")]
    ida, llegada, motivo = _ir(robot, servos, destino, postura, via_home=False)
    if ida is None:
        return [Paso(aviso=f"El camino hacia {_texto(destino)} {motivo}"
                           if not motivo.startswith("Para") else motivo)]
    regreso, _, _ = _ir(robot, llegada, robot.home, postura, via_home=False)
    return ida + (regreso or []) + [Paso(aviso=f"Punto {_texto(destino)} alcanzado; de regreso en HOME.")]


def plan_cosecha(robot, servos, puntos, postura=None):
    """Visitar cada punto (pasando por HOME si hace falta) y terminar en HOME (botón COSECHAR).

    Devuelve (plan, cosechados).
    """
    plan, cosechados, actual = [], 0, tuple(servos)
    for i, punto in enumerate(puntos, start=1):
        plan.append(Paso(aviso=f"Cosechando {i}/{len(puntos)}: {_texto(punto)}"))
        if not alcanzable(robot, punto):
            plan.append(Paso(aviso=f"Punto fuera del volumen de trabajo: {_texto(punto)}"))
            continue
        tramo, llegada, motivo = _ir(robot, actual, punto, postura, via_home=True)
        if tramo is None:
            plan.append(Paso(aviso=f"El camino hacia {_texto(punto)} {motivo}"
                                   if not motivo.startswith("Para") else motivo))
            continue
        plan.extend(tramo)
        cosechados, actual = cosechados + 1, llegada
    regreso, _, _ = _ir(robot, actual, robot.home, postura, via_home=False)
    plan.extend(regreso or [])
    plan.append(Paso(aviso=f"Cosecha terminada: {cosechados} de {len(puntos)} puntos alcanzados."))
    return plan, cosechados
