import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import pyvista as pv

from cinematica import A2, A3, D1, cinematica, servo_a_dh
from modelo3d import PIEZAS, cargar_pieza, pieza_provisional, poses_piezas


@pytest.fixture(scope="module")
def robot():
    return cinematica()


def en_base(T, p):
    return (T @ np.append(p, 1.0))[:3]


@pytest.mark.parametrize("servos", [(90, 46, 46), (0, 20, 0), (150, 120, 170), (45, 90, 90)])
def test_cada_pieza_empieza_donde_acaba_la_anterior(robot, servos):
    dh = servo_a_dh(*servos)
    _, A0_1, A0_2, A0_3 = robot.matrices(*dh)
    poses = poses_piezas(robot, *dh)
    # LA PUNTA DE LA COLUMNA ES EL HOMBRO
    assert np.allclose(en_base(poses["columna"], (0, 0, D1)), A0_1[:3, 3])
    # EL ORIGEN DEL BRAZO ES EL HOMBRO Y SU PUNTA (A2 SOBRE +X) ES EL CODO
    assert np.allclose(en_base(poses["brazo"], (0, 0, 0)), A0_1[:3, 3])
    assert np.allclose(en_base(poses["brazo"], (A2, 0, 0)), A0_2[:3, 3])
    # LA PUNTA DEL ANTEBRAZO (A3 SOBRE +X) ES EL EFECTOR, Y AHI VA LA HERRAMIENTA
    assert np.allclose(en_base(poses["antebrazo"], (A3, 0, 0)), A0_3[:3, 3])
    assert np.allclose(poses["efector"][:3, 3], robot.CD(*dh))


def test_las_piezas_provisionales_respetan_sus_medidas():
    xmin, xmax, *_ = pieza_provisional("brazo").bounds
    assert xmin < 0 < A2 < xmax  # CUBRE DEL HOMBRO AL CODO (CON LOS CUBOS DE ARTICULACION)
    *_, zmin, zmax = pieza_provisional("columna").bounds
    assert zmin < D1 < zmax


def test_un_stl_reemplaza_a_la_pieza_provisional(tmp_path):
    pv.Cube(center=(500, 0, 0), x_length=1000, y_length=50, z_length=50).triangulate().save(tmp_path / "brazo.stl")
    malla, origen = cargar_pieza("brazo", carpeta=str(tmp_path))
    assert origen.endswith("brazo.stl")
    assert malla.bounds[1] == pytest.approx(1000)
    _, origen_otra = cargar_pieza("antebrazo", carpeta=str(tmp_path))
    assert origen_otra == "provisional"


def test_la_simulacion_dibuja_todas_las_piezas(robot):
    robot.COSECHADOR(*servo_a_dh(90, 46, 46))
    assert set(robot.actores_piezas) == set(PIEZAS)
    T = robot.actores_piezas["efector"].user_matrix
    assert np.allclose(T[:3, 3], robot.CD(*servo_a_dh(90, 46, 46)))
