"""F1, bloque 3: el editor de puntos de un programa (B3.1–B3.11). Cada prueba lleva el número de su criterio.

La ventana es la de la app, con la misma receta de test_grabacion_ventana.py (se reutiliza su fixture): no se
abre nada a la vista, la carpeta de programas y los ajustes son temporales, todo lo que sale hacia la placa se
cuenta en `w.enviados` y los diálogos se contestan desde la prueba.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import yaml

from cosechador.movimiento import programa as prog
from cosechador.movimiento.trayectorias import posicion, servos_del_plan, tramo_a_punto, tramo_articular
from cosechador.robot.cinematica_inversa import angulos_servo


# LA RECTA DE A A B CRUZA EL EJE DE LA BASE: NO SE PUEDE RECORRER (EL MISMO CASO DE test_trayectorias.py)
A, B = (-1656, 0, 630), (1656, 0, 630)
# DE IDA A VUELTA LA RECTA SÍ SE PUEDE, Y EN ARTICULAR LA PUNTA SE ALEJA MUCHO DE ELLA: SIRVE PARA DISTINGUIR EL TIPO
IDA, VUELTA = (800, 300, 900), (-800, 300, 900)


@pytest.fixture(scope="module")
def home_s(robot):
    return angulos_servo(robot, robot.home)


def lejos_de_la_recta(robot, servos, desde, hasta):
    """Cuántos mm se aleja la punta, en cada pose, de la recta desde → hasta."""
    a, b = np.asarray(desde, float), np.asarray(hasta, float)
    u = (b - a) / np.linalg.norm(b - a)
    return [float(np.linalg.norm((p - a) - np.dot(p - a, u) * u)) for p in (posicion(robot, s) for s in servos)]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% B3.6: EL ARCHIVO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_b3_6_una_postura_con_numeros_de_numpy_se_guarda_como_numeros_normales(robot, home_s, tmp_path):
    # ASÍ QUEDAN LOS SERVOS DESPUÉS DE CUALQUIER MOVIMIENTO: tramo_articular LOS CALCULA CON NUMPY
    servos = servos_del_plan(tramo_articular(robot, home_s, (100, 60, 60)))[-1]
    assert type(servos[0]) is np.float64                               # SI ESTO CAMBIA, LA PRUEBA YA NO PRUEBA NADA
    por_puntos = prog.Programa("numpy", "puntos", robot.nombre, robot.gdl, puntos=(prog.Punto(servos, "lineal", 55),))
    grabacion = prog.Programa("numpy", "grabacion", robot.nombre, robot.gdl,
                              muestras=(prog.Muestra(0.0, home_s), prog.Muestra(1.0, servos)))
    for programa in (por_puntos, grabacion):
        ruta = str(tmp_path / f"{programa.tipo}.yaml")
        prog.guardar(programa, ruta)
        with open(ruta, encoding="utf-8") as f:
            crudo = yaml.safe_load(f)                                   # safe_load NO ACEPTA ETIQUETAS DE PYTHON NI DE NUMPY
        guardados = crudo.get("puntos") or crudo.get("muestras")
        assert all(type(s) is float for s in guardados[-1]["servos"])
        leido = prog.cargar(ruta, robot)
        assert (leido.puntos or leido.muestras)[-1].servos == pytest.approx(servos, abs=1e-6)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% B3.8: CÓMO SE LLEGA A UN PUNTO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_b3_8_a_un_punto_se_llega_con_su_tipo_y_su_velocidad(robot):
    """La trampa de UR: «ir al punto» iba siempre en articular, aunque el punto fuera lineal."""
    inicio, destino, ms = angulos_servo(robot, IDA), angulos_servo(robot, VUELTA), robot.movimiento.ms_por_paso
    recta, motivo = tramo_a_punto(robot, inicio, prog.Punto(destino, "lineal", 50))
    assert motivo is None
    assert max(lejos_de_la_recta(robot, servos_del_plan(recta), IDA, VUELTA)) < 0.5
    assert posicion(robot, servos_del_plan(recta)[-1]) == pytest.approx(VUELTA, abs=0.5)
    assert {p.ms for p in recta} == {ms * 2}                             # AL 50 %, CADA PASO DURA EL DOBLE

    giro, motivo = tramo_a_punto(robot, inicio, prog.Punto(destino, "articular", 20))
    assert motivo is None and servos_del_plan(giro)[-1] == pytest.approx(destino)
    assert max(lejos_de_la_recta(robot, servos_del_plan(giro), IDA, VUELTA)) > 50   # LA PUNTA NO VA EN RECTA
    assert {p.ms for p in giro} == {ms * 5}                              # AL 20 %, CINCO VECES MÁS

    nada, motivo = tramo_a_punto(robot, angulos_servo(robot, A), prog.Punto(angulos_servo(robot, B), "lineal", 100))
    assert nada is None and motivo.startswith("el camino pasa por") and "no alcanza" in motivo
