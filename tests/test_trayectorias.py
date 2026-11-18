"""Planeación de movimientos: se prueba sin ventana, solo con el plan que se le entrega a la interfaz."""
import numpy as np
import pytest

from cosechador.movimiento.trayectorias import (plan_cosecha, plan_ir_y_volver, posicion, servos_del_plan,
                                                tramo_continuo, tramo_recto)
from cosechador.robot.cinematica_inversa import alcanzable, angulos_servo, salto


@pytest.fixture(scope="module")
def home_s(robot):
    return angulos_servo(robot, robot.home)


def avisos_del_plan(plan):
    return [p.aviso for p in plan if p.aviso is not None]


def salto_maximo(inicio, plan):
    s = [inicio] + servos_del_plan(plan)
    return max(salto(s[i], s[i + 1]) for i in range(len(s) - 1))


@pytest.mark.parametrize("modo", ["despacio", "negar"])
def test_cosecha_de_demostracion_completa(robot, home_s, modo):
    plan, cosechados = plan_cosecha(robot, home_s, robot.puntos_demo, modo)
    assert cosechados == len(robot.puntos_demo) == 9
    assert avisos_del_plan(plan)[-1] == "Cosecha terminada: 9 de 9 puntos alcanzados."
    assert posicion(robot, servos_del_plan(plan)[-1]) == pytest.approx(robot.home, abs=1e-6)


def test_cada_punto_de_la_cosecha_se_visita(robot, home_s):
    plan, _ = plan_cosecha(robot, home_s, robot.puntos_demo)
    visitadas = [posicion(robot, s) for s in servos_del_plan(plan)]
    for punto in robot.puntos_demo:
        assert min(np.linalg.norm(v - punto) for v in visitadas) < 1e-6


def test_la_recta_que_cruza_el_eje_pasa_por_home(robot):
    # DE (-1656, 0, 630) A (1656, 0, 630) LA RECTA PASA DEMASIADO CERCA DEL EJE DE LA BASE: INALCANZABLE
    assert tramo_recto(robot, (-1656, 0, 630), (1656, 0, 630)) is None
    inicio = angulos_servo(robot, (-1656, 0, 630))
    plan, cosechados = plan_cosecha(robot, inicio, [(1656, 0, 630)])
    assert cosechados == 1
    visitadas = [posicion(robot, s) for s in servos_del_plan(plan)]
    assert min(np.linalg.norm(v - robot.home) for v in visitadas) < 1e-6


def test_todo_paso_del_plan_es_ejecutable(robot, home_s):
    plan, _ = plan_cosecha(robot, home_s, robot.puntos_demo)
    for servos in servos_del_plan(plan):
        assert alcanzable(robot, posicion(robot, servos))


def test_punto_inalcanzable_se_salta_con_aviso(robot, home_s):
    plan, cosechados = plan_cosecha(robot, home_s, [(5000, 0, 0), (430, 0, 1718)])
    assert cosechados == 1
    assert "Punto fuera del volumen de trabajo: (5000, 0, 0)" in avisos_del_plan(plan)


def test_ir_y_volver(robot, home_s):
    destino = (1200, 300, 900)
    plan = plan_ir_y_volver(robot, home_s, destino)
    visitadas = [posicion(robot, s) for s in servos_del_plan(plan)]
    assert min(np.linalg.norm(v - destino) for v in visitadas) < 1e-6
    assert visitadas[-1] == pytest.approx(robot.home, abs=1e-6)
    assert avisos_del_plan(plan)[-1] == "Punto (1200, 300, 900) alcanzado; de regreso en HOME."


def test_ir_a_un_punto_fuera_no_mueve_nada(robot, home_s):
    plan = plan_ir_y_volver(robot, home_s, (5000, 300, 900))
    assert servos_del_plan(plan) == []
    assert avisos_del_plan(plan) == ["Punto fuera del volumen de trabajo: (5000, 300, 900)"]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONTINUIDAD (BUG DEL 18-NOV: SALTOS DE HASTA 180°) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@pytest.mark.parametrize("modo", ["despacio", "negar"])
def test_c1_ningun_plan_salta_mas_que_el_servo(robot, home_s, modo):
    maximo = robot.movimiento.salto_max_por_paso
    plan, _ = plan_cosecha(robot, home_s, robot.puntos_demo, modo)
    assert salto_maximo(home_s, plan) <= maximo + 1e-9          # ANTES: 180°
    plan = plan_ir_y_volver(robot, home_s, (1200, 300, 900), modo)
    assert salto_maximo(home_s, plan) <= maximo + 1e-9          # ANTES: 76°


def test_c1_rectas_aleatorias_continuas(robot):
    maximo = robot.movimiento.salto_max_por_paso
    rng = np.random.default_rng(0)
    resueltas = 0
    for _ in range(300):
        a = rng.uniform([-1500, 0, 300], [1500, 1500, 1900])
        b = a.copy()
        b[rng.integers(3)] += rng.choice([-1, 1]) * 400
        inicio = angulos_servo(robot, a)
        if inicio is None:
            continue
        plan, _ = tramo_continuo(robot, inicio, b, "despacio")
        if plan is None:
            continue
        resueltas += 1
        assert salto_maximo(inicio, plan) <= maximo + 1e-9
        assert posicion(robot, servos_del_plan(plan)[-1]) == pytest.approx(b, abs=1e-6)
    assert resueltas > 100


def test_c2_cruzar_y_cero_respeta_el_modo(robot):
    # DE (-1400, 150, 1300) A (-1400, -150, 1300): TODO ES ALCANZABLE, PERO LA BASE (0-180°) TIENE QUE
    # VOLTEARSE PARA CRUZAR y = 0 (EL BRAZO PASA A INCLINARSE HACIA ATRAS)
    inicio = angulos_servo(robot, (-1400, 150, 1300))
    plan, motivo = tramo_continuo(robot, inicio, (-1400, -150, 1300), "negar")
    assert plan is None and "«negar»" in motivo
    plan, motivo = tramo_continuo(robot, inicio, (-1400, -150, 1300), "despacio")
    assert motivo is None
    assert any("Cambio de postura" in a for a in avisos_del_plan(plan))
    assert salto_maximo(inicio, plan) <= robot.movimiento.salto_max_por_paso + 1e-9


def test_c2_girar_en_el_lugar_siempre_se_permite(robot, home_s):
    # EN HOME LA PUNTA ESTA SOBRE EL EJE DE LA BASE: GIRARLA NO MUEVE LA PUNTA, AUN EN MODO «negar»
    plan, motivo = tramo_continuo(robot, home_s, (300, 0, 1065), "negar")
    assert motivo is None
    assert any("Giro en el lugar" in a for a in avisos_del_plan(plan))
