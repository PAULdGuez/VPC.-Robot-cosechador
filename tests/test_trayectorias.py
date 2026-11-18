"""Planeación de movimientos: se prueba sin ventana, solo con el plan que se le entrega a la interfaz."""
import numpy as np
import pytest

from cosechador.movimiento.trayectorias import (PASOS_POR_TRAMO, plan_cosecha, plan_ir_y_volver,
                                                posicion, tramo_recto)
from cosechador.robot.cinematica_inversa import alcanzable


def servos_del_plan(plan):
    return [p.servos for p in plan if p.servos is not None]


def avisos_del_plan(plan):
    return [p.aviso for p in plan if p.aviso is not None]


def test_cosecha_de_demostracion_completa(robot):
    plan, cosechados = plan_cosecha(robot, robot.home, robot.puntos_demo)
    assert cosechados == len(robot.puntos_demo) == 9
    assert avisos_del_plan(plan)[-1] == "Cosecha terminada: 9 de 9 puntos alcanzados."
    # TERMINA EN HOME
    assert posicion(robot, servos_del_plan(plan)[-1]) == pytest.approx(robot.home, abs=1e-6)


def test_cada_punto_de_la_cosecha_se_visita(robot):
    plan, _ = plan_cosecha(robot, robot.home, robot.puntos_demo)
    visitadas = [posicion(robot, s) for s in servos_del_plan(plan)]
    for punto in robot.puntos_demo:
        assert min(np.linalg.norm(v - punto) for v in visitadas) < 1e-6


def test_la_recta_que_cruza_el_eje_pasa_por_home(robot):
    # DE (-1656, 0, 630) A (1656, 0, 630) LA RECTA PASA POR EL EJE DE LA BASE: INALCANZABLE
    assert tramo_recto(robot, (-1656, 0, 630), (1656, 0, 630)) is None
    plan, cosechados = plan_cosecha(robot, (-1656, 0, 630), [(1656, 0, 630)])
    assert cosechados == 1
    visitadas = [posicion(robot, s) for s in servos_del_plan(plan)]
    assert min(np.linalg.norm(v - robot.home) for v in visitadas) < 1e-6


def test_todo_paso_del_plan_es_ejecutable(robot):
    plan, _ = plan_cosecha(robot, robot.home, robot.puntos_demo)
    for servos in servos_del_plan(plan):
        assert alcanzable(robot, posicion(robot, servos))


def test_punto_inalcanzable_se_salta_con_aviso(robot):
    plan, cosechados = plan_cosecha(robot, robot.home, [(5000, 0, 0), (430, 0, 1718)])
    assert cosechados == 1
    assert "Punto fuera del volumen de trabajo: (5000, 0, 0)" in avisos_del_plan(plan)


def test_ir_y_volver(robot):
    plan = plan_ir_y_volver(robot, robot.home, (1200, 300, 900))
    servos = servos_del_plan(plan)
    assert len(servos) == 2 * PASOS_POR_TRAMO
    assert posicion(robot, servos[PASOS_POR_TRAMO - 1]) == pytest.approx((1200, 300, 900), abs=1e-6)
    assert posicion(robot, servos[-1]) == pytest.approx(robot.home, abs=1e-6)


def test_ir_a_un_punto_fuera_no_mueve_nada(robot):
    plan = plan_ir_y_volver(robot, robot.home, (5000, 300, 900))
    assert servos_del_plan(plan) == []
    assert avisos_del_plan(plan) == ["Punto fuera del volumen de trabajo: (5000, 300, 900)"]
