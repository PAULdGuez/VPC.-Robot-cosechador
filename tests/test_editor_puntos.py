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
from cosechador.movimiento.trayectorias import servos_del_plan, tramo_articular
from cosechador.robot.cinematica_inversa import angulos_servo


@pytest.fixture(scope="module")
def home_s(robot):
    return angulos_servo(robot, robot.home)


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
