"""Escena 3D: cada pieza debe acabar donde empieza la siguiente, para cualquier robot del YAML."""
import dataclasses

import numpy as np
import pytest
import pyvista as pv

from cosechador.robot.cinematica import matrices, servo_a_dh
from cosechador.ui.escena3d import Escena, cargar_pieza, pieza_provisional, piezas_del_robot, poses_piezas


def en_base(T, p):
    return (T @ np.append(p, 1.0))[:3]


@pytest.mark.parametrize("servos", [(90, 46, 46), (0, 20, 0), (150, 120, 170), (45, 90, 90)])
def test_cada_pieza_empieza_donde_acaba_la_anterior(robot, servos):
    q = servo_a_dh(robot, servos)
    acumuladas = matrices(robot, q)
    poses = poses_piezas(robot, q)
    for i, j in enumerate(robot.articulaciones):
        # LA PIEZA i ARRANCA EN SU EJE Y ACABA (d SOBRE Z, a SOBRE +X) EN EL EJE DE LA SIGUIENTE
        assert np.allclose(en_base(poses[i + 1], (0, 0, 0)), acumuladas[i][:3, 3])
        assert np.allclose(en_base(poses[i + 1], (j.a, 0, j.d)), acumuladas[i + 1][:3, 3])
    assert np.allclose(poses[-1], acumuladas[-1])  # EL EFECTOR VA EN LA PUNTA


def test_las_piezas_provisionales_respetan_sus_medidas(robot):
    d1, a2 = robot.articulaciones[0].d, robot.articulaciones[1].a
    xmin, xmax, *_ = pieza_provisional(robot, 2).bounds  # BRAZO
    assert xmin < 0 < a2 < xmax
    *_, zmin, zmax = pieza_provisional(robot, 1).bounds  # COLUMNA
    assert zmin < d1 < zmax


def test_un_stl_reemplaza_a_la_pieza_provisional(robot, tmp_path):
    pv.Cube(center=(500, 0, 0), x_length=1000, y_length=50, z_length=50).triangulate().save(tmp_path / "brazo.stl")
    con_carpeta = dataclasses.replace(robot, carpeta_modelos=str(tmp_path))
    _, brazo = piezas_del_robot(con_carpeta)[2]
    malla, origen = cargar_pieza(con_carpeta, 2, brazo)
    assert origen.endswith("brazo.stl")
    assert malla.bounds[1] == pytest.approx(1000)
    _, antebrazo = piezas_del_robot(con_carpeta)[3]
    assert cargar_pieza(con_carpeta, 3, antebrazo)[1] == "provisional"


def test_la_escena_dibuja_todas_las_piezas(robot):
    escena = Escena(pv.Plotter(off_screen=True), robot)
    q = servo_a_dh(robot, (90, 46, 46))
    escena.dibujar(q)
    assert len(escena.actores) == robot.gdl + 2
    assert np.allclose(escena.actores[-1].user_matrix[:3, 3], matrices(robot, q)[-1][:3, 3])


def test_c4_otro_robot_se_dibuja_con_piezas_provisionales(robot_4gdl):
    escena = Escena(pv.Plotter(off_screen=True), robot_4gdl)
    q = (30, 20, -40, 10)
    escena.dibujar(q)
    assert len(escena.actores) == 6
    assert set(escena.origen_piezas.values()) == {"provisional"}
    for i, j in enumerate(robot_4gdl.articulaciones):
        assert np.allclose(en_base(poses_piezas(robot_4gdl, q)[i + 1], (j.a, 0, j.d)),
                           matrices(robot_4gdl, q)[i + 1][:3, 3])


def test_u1_plan_y_recorrido_de_la_punta(robot):
    escena = Escena(pv.Plotter(off_screen=True), robot)
    escena.mostrar_plan([(0, 0, 1065), (300, 0, 1065), (300, 300, 1065)])
    assert escena._plan.n_points == 3 and escena._actor_plan.GetVisibility()
    escena.reiniciar_recorrido((0, 0, 1065))
    for x in (0.1, 50, 50.2, 100):            # 0.1 Y 50.2 ESTÁN A < 0.5 mm DEL ANTERIOR: NO SE AGREGAN
        escena.agregar_al_recorrido((x, 0, 1065))
    assert [p[0] for p in escena.puntos_recorrido] == [0, 50, 100]
    assert escena._recorrido.n_points == 3 and escena._actor_recorrido.GetVisibility()
    escena.mostrar_trayectoria(False)
    assert not escena._actor_plan.GetVisibility() and not escena._actor_recorrido.GetVisibility()
    escena.mostrar_trayectoria(True)
    escena.limpiar_trayectoria((100, 0, 1065))
    assert len(escena.puntos_recorrido) == 1
    assert not escena._actor_plan.GetVisibility()  # SIN PLAN NO HAY LÍNEA QUE MOSTRAR


def test_u2_ejes_en_el_origen(robot):
    plotter = pv.Plotter(off_screen=True)
    antes = len(plotter.renderer.actors)
    Escena(plotter, robot)
    # PISO + PIEZAS + 3 FLECHAS + 3 ETIQUETAS (MÍNIMO)
    assert len(plotter.renderer.actors) - antes >= 1 + robot.gdl + 2 + 6
