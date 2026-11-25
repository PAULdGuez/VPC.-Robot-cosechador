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
    """Un paso del plan: ángulos de servo a los que ir, un aviso para la persona, o ambos.

    ms: cuánto dura este paso (a velocidad general 100 %). None = el ms_por_paso del robot.
    Un paso sin servos pero con ms es una ESPERA (p. ej. una pausa de una grabación).
    """
    servos: tuple | None = None
    aviso: str | None = None
    ms: float | None = None
    nivel: str = "info"           # DEL AVISO: info | ok | aviso | error (ver cosechador/registro.py)


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
    return [Paso(aviso=aviso, nivel="aviso")] + pasos


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
    """Pasos para llegar a destino desde `servos`. Si la recta no sirve y via_home, prueba pasar por el punto de
    giro de la base (donde girarla no mueve la punta), no por la punta de HOME. Devuelve (pasos, servos_finales, motivo)."""
    plan, motivo = tramo_continuo(robot, servos, destino, postura)
    if plan is not None:
        return plan, servos_del_plan(plan)[-1] if servos_del_plan(plan) else servos, None
    if via_home:
        a_home, _ = tramo_continuo(robot, servos, robot.punto_de_giro, postura)
        if a_home is not None:
            en_home = servos_del_plan(a_home)[-1] if servos_del_plan(a_home) else servos
            desde_home, _ = tramo_continuo(robot, en_home, destino, postura)
            if desde_home is not None:
                return a_home + desde_home, servos_del_plan(desde_home)[-1], None
    return None, servos, motivo


def plan_ir_y_volver(robot, servos, destino, postura=None):
    """Ir al destino en línea recta y regresar a HOME, a la postura de descanso (botón CALCULAR)."""
    if not alcanzable(robot, destino):
        return [Paso(aviso=f"Punto fuera del volumen de trabajo: {_texto(destino)}", nivel="aviso")]
    ida, llegada, motivo = _ir(robot, servos, destino, postura, via_home=False)
    if ida is None:
        return [Paso(aviso=f"El camino hacia {_texto(destino)} {motivo}"
                           if not motivo.startswith("Para") else motivo, nivel="aviso")]
    regreso, _, _ = _ir(robot, llegada, robot.home, postura, via_home=False)
    return ida + (regreso or []) + [Paso(aviso=f"Punto {_texto(destino)} alcanzado; de regreso en HOME.", nivel="ok")]


def plan_cosecha(robot, servos, puntos, postura=None):
    """Visitar cada punto (pasando por HOME si hace falta) y terminar en HOME (botón COSECHAR).

    Devuelve (plan, cosechados).
    """
    plan, cosechados, actual = [], 0, tuple(servos)
    for i, punto in enumerate(puntos, start=1):
        plan.append(Paso(aviso=f"Cosechando {i}/{len(puntos)}: {_texto(punto)}"))
        if not alcanzable(robot, punto):
            plan.append(Paso(aviso=f"Punto fuera del volumen de trabajo: {_texto(punto)}", nivel="aviso"))
            continue
        tramo, llegada, motivo = _ir(robot, actual, punto, postura, via_home=True)
        if tramo is None:
            plan.append(Paso(aviso=f"El camino hacia {_texto(punto)} {motivo}"
                                   if not motivo.startswith("Para") else motivo, nivel="aviso"))
            continue
        plan.extend(tramo)
        cosechados, actual = cosechados + 1, llegada
    regreso, _, _ = _ir(robot, actual, robot.home, postura, via_home=False)
    plan.extend(regreso or [])
    plan.append(Paso(aviso=f"Cosecha terminada: {cosechados} de {len(puntos)} puntos alcanzados.",
                     nivel="ok" if cosechados == len(puntos) else "aviso"))
    return plan, cosechados


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MOVIMIENTO ARTICULAR Y PROGRAMAS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def tramo_articular(robot, servos_a, servos_b, ms=None):
    """Interpola los servos de una pose a otra, sin que ninguno gire más de salto_max_por_paso.

    Siempre es posible entre dos poses válidas: el rango de cada servo es un intervalo y la recta
    entre dos puntos de un intervalo se queda dentro. La punta NO va en línea recta.
    """
    a, b = np.asarray(servos_a, float), np.asarray(servos_b, float)
    n = max(1, math.ceil(salto(a, b) / robot.movimiento.salto_max_por_paso - 1e-9))
    return [Paso(servos=tuple(a + (b - a) * k), ms=ms) for k in np.linspace(0, 1, n + 1)[1:]]


def _ms_a_velocidad(robot, velocidad):
    return robot.movimiento.ms_por_paso * 100 / velocidad


def tramo_a_punto(robot, servos, punto, postura=None):
    """Cómo se llega a UN punto de un programa desde `servos`: con SU tipo (articular o lineal) y SU velocidad.

    Devuelve (pasos, None), o (None, motivo) si el punto es lineal y la recta no se puede recorrer. Lo usan
    plan_puntos (el programa completo) y el paso a paso del editor: un solo lugar decide cómo se llega.
    """
    ms = _ms_a_velocidad(robot, punto.velocidad)
    if punto.movimiento == "articular":
        return tramo_articular(robot, servos, punto.servos, ms), None
    tramo, motivo = tramo_continuo(robot, servos, posicion(robot, punto.servos), postura)
    if tramo is None:
        return None, motivo if motivo.startswith("Para") else f"el camino {motivo}"
    return [Paso(servos=p.servos, aviso=p.aviso, ms=ms) for p in tramo], None


def plan_puntos(robot, servos, programa, postura=None):
    """Plan para recorrer un programa por puntos. Devuelve (plan, problemas).

    problemas: [(número de punto, motivo)]. Si hay alguno, plan es None: el programa NO arranca.
    """
    plan, problemas, actual = [], [], tuple(servos)
    for i, punto in enumerate(programa.puntos, start=1):
        tramo, motivo = tramo_a_punto(robot, actual, punto, postura)
        if tramo is None:
            problemas.append((i, f"Punto {i} ({punto.movimiento}): {motivo}"))
            actual = punto.servos  # SE SIGUE REVISANDO DESDE DONDE DEBIÓ QUEDAR, PARA LISTAR TODOS LOS PROBLEMAS
            continue
        plan.append(Paso(aviso=f"Punto {i}/{len(programa.puntos)} ({punto.movimiento}, {punto.velocidad} %)"))
        plan.extend(tramo)
        if punto.movimiento == "articular":  # EN ARTICULAR SE LLEGA EXACTO A LOS SERVOS GUARDADOS
            actual = punto.servos
        elif servos_del_plan(tramo):         # EN LINEAL, A LOS QUE DIO LA CINEMÁTICA INVERSA PARA ESA XYZ
            actual = servos_del_plan(tramo)[-1]
    if problemas:
        return None, problemas
    return plan + [Paso(aviso=f"Programa «{programa.nombre}» terminado.", nivel="ok")], []


UMBRAL_QUIETO = 0.05   # GRADOS: POR DEBAJO, ENTRE DOS MUESTRAS NO HUBO MOVIMIENTO (ES PAUSA)


def plan_grabacion(robot, servos, programa, quitar_pausas=False):
    """Plan para reproducir una grabación respetando sus tiempos. Devuelve (plan, problemas).

    Primero va (articular) a la primera pose. Entre muestras, si el servo tendría que girar más de
    salto_max_por_paso, se parte en pasos; si la grabación fue más rápida de lo que aguanta el servo,
    se reproduce a lo más rápido seguro (más lento que lo grabado) y se avisa al final.
    """
    muestras = programa.muestras
    plan = [Paso(aviso=f"Yendo al inicio de «{programa.nombre}»")]
    plan.extend(tramo_articular(robot, servos, muestras[0].servos))
    plan.append(Paso(aviso=f"Reproduciendo «{programa.nombre}» ({programa.duracion:.1f} s grabados)"))
    ms_min, retraso = robot.movimiento.ms_por_paso, 0.0
    for anterior, siguiente in zip(muestras, muestras[1:]):
        dt_ms = (siguiente.t - anterior.t) * 1000
        giro = salto(anterior.servos, siguiente.servos)
        if giro < UMBRAL_QUIETO:
            if not quitar_pausas and dt_ms > 0:
                plan.append(Paso(ms=dt_ms))  # ESPERA: SE QUEDÓ QUIETO ESE TIEMPO
            continue
        pasos = tramo_articular(robot, anterior.servos, siguiente.servos)
        ms = dt_ms / len(pasos)
        if ms < ms_min:  # MÁS RÁPIDO DE LO QUE AGUANTA EL SERVO
            retraso += (ms_min - ms) * len(pasos)
            ms = ms_min
        plan.extend(Paso(servos=p.servos, ms=ms) for p in pasos)
    final = f"Grabación «{programa.nombre}» terminada."
    if retraso > 1:
        final += f" Tardó {retraso / 1000:.1f} s más que lo grabado: hubo movimientos más rápidos que lo que aguanta el servo."
    return plan + [Paso(aviso=final, nivel="aviso" if retraso > 1 else "ok")], []


def duracion_ms(robot, plan):
    """Duración total del plan a velocidad general 100 %."""
    return sum(p.ms if p.ms is not None else (robot.movimiento.ms_por_paso if p.servos is not None else 0)
               for p in plan)
