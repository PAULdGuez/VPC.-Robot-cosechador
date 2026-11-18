import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from cinematica import (A2, A3, D1, HOME, LIMITES_SERVO, cinematica, dh_a_servo,
                        servo_a_dh)


@pytest.fixture(scope="module")
def robot():
    return cinematica()


def test_home_es_alcanzable_y_coincide_con_los_sliders(robot):
    # HOME ESTA SOBRE EL EJE DE LA BASE: LA BASE VA AL CENTRO DE SU RANGO (90°) PARA QUE LA CAMARA
    # MIRE A LA MITAD DE LA ZONA DE TRABAJO; BRAZO Y ANTEBRAZO COMO LOS SLIDERS ORIGINALES (45, 45)
    s1, s2, s3 = robot.angulos_servo(*HOME)
    assert (round(s1), round(s2), round(s3)) == (90, 46, 46)


@pytest.mark.parametrize("servos", [(0, 20, 0), (90, 45, 45), (180, 180, 180),
                                    (30, 100, 120), (150, 60, 10)])
def test_ida_y_vuelta_servo_dh(servos):
    assert np.allclose(dh_a_servo(*servo_a_dh(*servos)), servos)


def test_ci_es_la_inversa_de_cd(robot):
    # TODO PUNTO GENERADO CON ANGULOS VALIDOS DEBE REGRESAR A ESE MISMO PUNTO
    rng = np.random.default_rng(7)
    probados = 0
    for _ in range(500):
        servos = [rng.uniform(lo, hi) for lo, hi in LIMITES_SERVO]
        punto = robot.CD(*servo_a_dh(*servos))
        try:
            t = robot.CI(*punto)
        except ValueError:
            continue
        assert np.allclose(robot.CD(*t), punto, atol=1e-6)
        probados += 1
    assert probados > 400


def test_ci_usa_la_altura_de_la_base(robot):
    # BRAZO ESTIRADO HACIA ARRIBA: z = D1 + A2 + A3. CON LA CI VIEJA (SIN D1) ESTE PUNTO TRONABA.
    assert robot.CI(0.001, 0, D1 + A2 + A3 - 1) is not None
    with pytest.raises(ValueError):
        robot.CI(0, 0, D1 + A2 + A3 + 10)


def test_punto_fuera_de_alcance(robot):
    assert robot.angulos_servo(5000, 0, 0) is None
    assert not robot.punto_en_volumen(5000, 0, 0)


def test_punto_que_pide_un_servo_fuera_de_rango(robot):
    # y < 0 PEDIRIA GIRAR LA BASE A ANGULOS NEGATIVOS; EL SERVO 1 SOLO VA DE 0 A 180
    assert robot.angulos_servo(500, -500, 800) is None
    assert robot.angulos_servo(500, 500, 800) is not None


def test_no_hace_falta_calcular_el_volumen_antes(robot):
    # ANTES TRONABA SI SE PREGUNTABA ALCANCE SIN HABER PRESIONADO "VER VOLUMEN"
    assert robot.punto_en_volumen(*HOME)


def test_volumen_respeta_rango_de_servos(robot):
    class Barra:
        def setRange(self, a, b): self.max = b
        def setValue(self, v): self.valor = v
    barra = Barra()
    puntos = robot.volumen_trabajo(progresbar=barra, paso=10)
    assert barra.valor == barra.max == len(puntos)
    # TODO PUNTO DEL VOLUMEN DEBE SER ALCANZABLE SEGUN LA CI
    for p in puntos:
        assert robot.punto_en_volumen(*p)
