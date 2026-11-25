"""tramos_alcanzables: los tramos de un eje a los que llega la punta, medidos contra `alcanzable` (S1, la función sola)."""
import numpy as np
import pytest

from cosechador.robot.cinematica_inversa import TOLERANCIA_TRAMOS_MM, alcanzable, tramos_alcanzables

PUNTO_DE_PAUL = (-697, 718, 1862)  # LA POSE DE LA CAPTURA DEL 24-NOV


def con_el_eje(punto, eje, valor):
    q = list(punto)
    q[eje] = valor
    return q


@pytest.mark.parametrize("eje, rango, esperado", [(0, (-2170, 2170), (-710, 710)), (1, (-2170, 2170), (-735, 735)),
                                                  (2, (0, 2170), (0, 1870))])
def test_la_pose_de_paul_da_un_tramo_por_eje_con_las_cifras_de_la_nota(robot, eje, rango, esperado):
    tramos = tramos_alcanzables(robot, PUNTO_DE_PAUL, eje, *rango)
    assert len(tramos) == 1
    assert tramos[0] == pytest.approx(esperado, abs=5)  # LA NOTA MIDIÓ CADA 5 mm: ESTO AFINA LA FRONTERA


def test_con_un_hueco_junto_a_la_base_salen_dos_tramos(robot):
    tramos = tramos_alcanzables(robot, (-600, 0, 500), 0, -2170, 2170)
    assert len(tramos) == 2 and tramos[0][1] < tramos[1][0]
    hueco = (tramos[0][1] + tramos[1][0]) / 2
    assert not alcanzable(robot, con_el_eje((-600, 0, 500), 0, hueco))


def test_todo_lo_de_dentro_se_alcanza_y_lo_de_fuera_a_mas_de_la_tolerancia_no(robot):
    for punto in (PUNTO_DE_PAUL, (-600, 0, 500), (0, 120, 1065), (800, 300, 900)):
        for eje, rango in enumerate(((-2170, 2170), (-2170, 2170), (0, 2170))):
            tramos = tramos_alcanzables(robot, punto, eje, *rango)
            for a, b in tramos:
                assert alcanzable(robot, con_el_eje(punto, eje, a)) and alcanzable(robot, con_el_eje(punto, eje, b))
                assert all(alcanzable(robot, con_el_eje(punto, eje, v)) for v in np.linspace(a, b, 25))
                for fuera in (a - 1.01 * TOLERANCIA_TRAMOS_MM, b + 1.01 * TOLERANCIA_TRAMOS_MM):
                    if rango[0] <= fuera <= rango[1]:
                        assert not alcanzable(robot, con_el_eje(punto, eje, fuera))


def test_un_punto_fuera_de_todo_tramo_no_se_inventa_uno(robot):
    assert tramos_alcanzables(robot, (5000, 0, 0), 0, 3000, 4000) == []
