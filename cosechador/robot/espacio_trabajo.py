"""Volumen de trabajo: los puntos que alcanza el efector recorriendo el rango de cada servo.

Para dibujarlo como superficie basta su CORTE: lo que el efector alcanza con la base quieta. El contorno de
ese corte, girado lo que gira la base, es la superficie del volumen (el giro se hace en ui/escena3d.py).
"""
import itertools

import numpy as np
from scipy.spatial import Delaunay, QhullError

from .cinematica import cinematica_directa, rot_z, servo_a_dh

PASO_CORTE = 2  # GRADOS ENTRE DOS MUESTRAS DEL CORTE (COMO MUCHO)
# EL CORTE SOLO SE PUEDE GIRAR SI ES PLANO: CON LA BASE QUIETA, EL EFECTOR NO SE SALE DEL PLANO VERTICAL QUE PASA
# POR EL EJE DE LA BASE. SE MIDE, NO SE SUPONE; MENOS DE ESTO ES RUIDO NUMÉRICO (EL COSECHADOR DA 1e-13 mm)
EN_EL_PLANO_MM = 0.01


def rangos_servo(robot, paso):
    """Los ángulos de cada servo que se barren: de su mínimo a su máximo, cada `paso` grados como mucho.

    Los dos límites entran siempre: si el rango no es múltiplo del paso, el paso se acorta lo necesario.
    """
    return [np.linspace(j.servo.minimo, j.servo.maximo,
                        int(np.ceil((j.servo.maximo - j.servo.minimo) / paso - 1e-9)) + 1)
            for j in robot.articulaciones]


def total_puntos(robot, paso=5):
    return int(np.prod([len(r) for r in rangos_servo(robot, paso)]))


def _recorrer(robot, rangos, progreso=None):
    """El efector en todas las combinaciones de esos ángulos de servo (en el orden de itertools.product)."""
    puntos = np.empty((*(len(r) for r in rangos), 3)).reshape(-1, 3)
    total = len(puntos)
    aviso_cada = max(total // 100, 1)
    for i, servos in enumerate(itertools.product(*rangos)):
        puntos[i] = cinematica_directa(robot, servo_a_dh(robot, servos))
        if progreso is not None and (i + 1) % aviso_cada == 0:
            progreso(i + 1, total)
    if progreso is not None:
        progreso(total, total)
    return puntos


def barrer(robot, paso=5, progreso=None):
    """Barre todas las combinaciones de servos (cada `paso` grados como mucho; ver rangos_servo).

    progreso(hechos, total) se llama de vez en cuando, para una barra de progreso.
    """
    return _recorrer(robot, rangos_servo(robot, paso), progreso)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EL CORTE Y SU CONTORNO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def giro_de_la_base(robot):
    """(desde, hasta): el ángulo DH de la primera articulación en los dos límites de su servo, en grados.

    La primera articulación gira sobre el eje Z de la base: de `desde` a `hasta` es cuánto, y hacia dónde,
    gira el corte para llenar el volumen de trabajo.
    """
    minimos, maximos = zip(*((j.servo.minimo, j.servo.maximo) for j in robot.articulaciones))
    return servo_a_dh(robot, minimos)[0], servo_a_dh(robot, maximos)[0]


def corte(robot, paso=PASO_CORTE, progreso=None):
    """Lo que alcanza el efector con la base quieta, visto en el plano vertical del frente de la base.

    Devuelve los puntos (alcance, altura) en mm, acomodados con un eje por cada articulación después de la
    primera (los ángulos de rangos_servo). El alcance se mide desde el eje de la base hacia su frente, y es
    negativo si el punto queda detrás. Devuelve None si el efector se sale de ese plano: entonces el corte
    girado NO es el volumen de trabajo.
    """
    rangos = rangos_servo(robot, paso)
    rangos[0] = rangos[0][:1]  # LA BASE, QUIETA EN SU MÍNIMO
    puntos = _recorrer(robot, rangos, progreso)
    # SE DESHACE EL GIRO DE LA BASE: X QUEDA SOBRE SU FRENTE Y EN Y QUEDA LO QUE EL EFECTOR SE SALE DEL PLANO
    alcance, fuera, altura = (puntos @ rot_z(giro_de_la_base(robot)[0])[:3, :3]).T
    if np.abs(fuera).max() > EN_EL_PLANO_MM:
        return None
    return np.column_stack([alcance, altura]).reshape(*(len(r) for r in rangos[1:]), 2)


def contorno(puntos, lado_maximo):
    """Contorno de una nube de puntos del plano (n, 2): lista de lazos cerrados, cada uno (m, 2).

    La nube se triangula (Delaunay) y se descartan los triángulos con algún lado más largo que `lado_maximo`:
    son los que saltan sobre un hueco o una entrada de la nube. El contorno son los lados que quedan con un
    triángulo de un solo lado. Cada lazo deja lo de adentro a su izquierda (los de los huecos giran al revés).
    Si no queda ningún triángulo (puntos alineados, o menos de tres), no hay contorno: lista vacía.
    """
    puntos = np.asarray(puntos, dtype=float)
    try:
        triangulos = Delaunay(puntos)
    except QhullError:
        return []
    esquinas = puntos[triangulos.simplices]
    dentro = np.linalg.norm(esquinas - np.roll(esquinas, 1, axis=1), axis=2).max(axis=1) <= lado_maximo
    # scipy: EL VECINO k ES EL TRIÁNGULO DE ENFRENTE DEL VÉRTICE k (-1 SI NO HAY), Y LOS VÉRTICES VAN EN SENTIDO
    # ANTIHORARIO: EL LADO DE ENFRENTE DEL VÉRTICE k VA DEL k+1 AL k+2, CON EL TRIÁNGULO A SU IZQUIERDA
    vecinos = triangulos.neighbors
    triangulo, k = np.nonzero(dentro[:, None] & ((vecinos < 0) | ~dentro[vecinos]))
    siguientes = {}
    for desde, hasta in zip(triangulos.simplices[triangulo, (k + 1) % 3], triangulos.simplices[triangulo, (k + 2) % 3]):
        siguientes.setdefault(int(desde), []).append(int(hasta))
    lazos = []
    while siguientes:  # SE ENCADENAN LOS LADOS; DONDE DOS LAZOS SE TOCAN EN UN VÉRTICE, ESE VÉRTICE TIENE DOS SALIDAS
        inicio = vertice = next(iter(siguientes))
        lazo = []
        while not lazo or vertice != inicio:
            lazo.append(vertice)
            vertice = siguientes[lazo[-1]].pop()
            if not siguientes[lazo[-1]]:
                del siguientes[lazo[-1]]
        lazos.append(puntos[lazo])
    return lazos


def contorno_del_corte(robot, paso=PASO_CORTE, progreso=None):
    """Contorno de lo que alcanza el efector con la base quieta: lazos de puntos (alcance, altura) en mm.

    Tal cual, es el corte 2D del volumen de trabajo; girado lo que gira la base (giro_de_la_base), su
    superficie. None si el corte no sirve para eso: el efector se sale del plano (ver corte) o lo que
    alcanza no encierra área (un punto o un arco).
    """
    puntos = corte(robot, paso, progreso)
    if puntos is None:
        return None
    # DOS MUESTRAS VECINAS DEL BARRIDO DISTAN, COMO MUCHO, UN SALTO POR CADA ARTICULACIÓN QUE CAMBIA. UN LADO MÁS
    # LARGO QUE LA SUMA DE ESOS SALTOS NO UNE VECINAS: CRUZA POR DONDE EL ROBOT NO LLEGA
    lado_maximo = sum(np.linalg.norm(np.diff(puntos, axis=eje), axis=-1).max() for eje in range(puntos.ndim - 1))
    return contorno(puntos.reshape(-1, 2), lado_maximo) or None
