import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from cinematica import HOME, cinematica, servo_a_dh
from vision import (DIAMETRO_AGUACATE_MM, camara_a_pixel, detectar_aguacates, matriz_camara,
                    pedunculo_en_base, pixel_a_camara)

ANCHO, ALTO = 1280, 720
K = matriz_camara(ANCHO, ALTO)


@pytest.fixture(scope="module")
def robot():
    return cinematica()


@pytest.fixture(scope="module")
def T_home(robot):
    return robot.matrices(*servo_a_dh(*robot.angulos_servo(*HOME)))[-1]


def foto_con_aguacate(centro_camara_mm):
    """Dibuja un aguacate de 80 mm en la posicion 3D dada (marco de la camara)."""
    img = np.full((ALTO, ANCHO, 3), 40, np.uint8)
    u, v = camara_a_pixel(centro_camara_mm, K)
    radio_px = K[0, 0] * (DIAMETRO_AGUACATE_MM / 2) / centro_camara_mm[2]
    # ALTO 1.3 VECES EL ANCHO, COMO UN AGUACATE
    cv2.ellipse(img, (int(round(u)), int(round(v))), (int(round(radio_px)), int(round(radio_px * 1.3))),
                0, 0, 360, (40, 170, 40), -1)
    return img


def detectar_uno(img):
    _, detecciones, _, _ = detectar_aguacates(img)
    assert len(detecciones) == 1
    return detecciones[0]


def test_ida_y_vuelta_pixel_camara():
    p = np.array([120.0, -80.0, 900.0])
    assert np.allclose(pixel_a_camara(*camara_a_pixel(p, K), 900.0, K), p)


@pytest.mark.parametrize("distancia", [500, 800, 1200])
def test_estima_la_profundidad_por_el_tamano(distancia, T_home):
    det = detectar_uno(foto_con_aguacate(np.array([0.0, 0.0, distancia])))
    p = pedunculo_en_base(det, ANCHO, ALTO, T_home)
    # EN HOME LA CAMARA ESTA EN (0, 0, 1065) MIRANDO A +Y: LA Y DEL PEDUNCULO ES LA DISTANCIA.
    # 5 %: LA PRECISION REALISTA DE ESTIMAR DISTANCIA POR TAMAÑO (1-2 PX DE BORDE EN ~50 PX)
    assert p[1] == pytest.approx(distancia, rel=0.05)


def test_aguacate_al_centro_queda_enfrente_y_arriba_del_fruto(T_home):
    det = detectar_uno(foto_con_aguacate(np.array([0.0, 0.0, 800.0])))
    x, y, z = pedunculo_en_base(det, ANCHO, ALTO, T_home)
    assert abs(x) < 5                      # NI A LA IZQUIERDA NI A LA DERECHA
    assert HOME[2] + 40 < z < HOME[2] + 120  # EL CORTE VA ENCIMA DEL FRUTO, NO EN SU CENTRO


def test_los_ejes_de_la_imagen_corresponden_al_mundo(T_home):
    centro = pedunculo_en_base(detectar_uno(foto_con_aguacate(np.array([0.0, 0.0, 800.0]))), ANCHO, ALTO, T_home)
    derecha = pedunculo_en_base(detectar_uno(foto_con_aguacate(np.array([200.0, 0.0, 800.0]))), ANCHO, ALTO, T_home)
    arriba = pedunculo_en_base(detectar_uno(foto_con_aguacate(np.array([0.0, -150.0, 800.0]))), ANCHO, ALTO, T_home)
    # MIRANDO HACIA +Y CON Z ARRIBA, LA DERECHA DE LA FOTO ES +X DEL ROBOT Y ARRIBA ES +Z
    assert derecha[0] == pytest.approx(centro[0] + 200, abs=10)
    assert arriba[2] == pytest.approx(centro[2] + 150, abs=10)


@pytest.mark.parametrize("lado_mm", [-300.0, 0.0, 300.0])
def test_un_aguacate_enfrente_se_puede_cosechar(robot, T_home, lado_mm):
    # ANTES, NINGUNA DETECCION QUEDABA AL ALCANCE; Y CON LA BASE EN 0° LA MITAD DERECHA DE LA FOTO TAMPOCO
    det = detectar_uno(foto_con_aguacate(np.array([lado_mm, 50.0, 800.0])))
    assert robot.punto_en_volumen(*pedunculo_en_base(det, ANCHO, ALTO, T_home))


def test_imagen_sin_aguacates():
    _, detecciones, _, _ = detectar_aguacates(np.full((ALTO, ANCHO, 3), 40, np.uint8))
    assert detecciones == []
