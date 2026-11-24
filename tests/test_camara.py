"""Vision: deteccion + camara, con aguacates sinteticos dibujados en posiciones 3D conocidas."""
import cv2
import numpy as np
import pytest

from cosechador.robot.cinematica import servo_a_dh
from cosechador.robot.cinematica_inversa import alcanzable, angulos_servo
from cosechador.robot.config import con_herramienta
from cosechador.vision.camara import camara_a_pixel, matriz_camara, pedunculo_en_base, pixel_a_camara
from cosechador.vision.deteccion import detectar_aguacates

ANCHO, ALTO = 1280, 720


@pytest.fixture(scope="module")
def K(robot):
    return matriz_camara(robot.camara, ANCHO, ALTO)


@pytest.fixture(scope="module")
def q_de_frente(robot):
    # LA POSTURA QUE DEJA LA CÁMARA (EL EXTREMO DEL ANTEBRAZO) EN EL PUNTO DE HOME, MIRANDO DE FRENTE: LA DE HOME SIN
    # HERRAMIENTA. CON HERRAMIENTA, A ESE PUNTO VA LA PUNTA: EN HOME LA CÁMARA QUEDA MÁS ATRÁS Y MIRANDO HACIA ARRIBA.
    return servo_a_dh(robot, angulos_servo(con_herramienta(robot, 0), robot.home))


@pytest.fixture(scope="module")
def foto(robot, K):
    def dibujar(centro_camara_mm):
        return foto_con_aguacate(centro_camara_mm, K, robot.camara.diametro_objeto_mm)
    return dibujar


@pytest.fixture(scope="module")
def detectar_uno(robot):
    def detectar(img):
        _, detecciones, _, _ = detectar_aguacates(img, robot.vision, 30, 90)
        assert len(detecciones) == 1
        return detecciones[0]
    return detectar


@pytest.fixture(scope="module")
def en_base(robot, q_de_frente):
    def convertir(det):
        return pedunculo_en_base(det, ANCHO, ALTO, robot, q_de_frente)
    return convertir


def foto_con_aguacate(centro_camara_mm, K, diametro):
    """Dibuja un aguacate del diametro dado en la posicion 3D indicada (marco de la camara)."""
    img = np.full((ALTO, ANCHO, 3), 40, np.uint8)
    u, v = camara_a_pixel(centro_camara_mm, K)
    radio_px = K[0, 0] * (diametro / 2) / centro_camara_mm[2]
    # ALTO 1.3 VECES EL ANCHO, COMO UN AGUACATE
    cv2.ellipse(img, (int(round(u)), int(round(v))), (int(round(radio_px)), int(round(radio_px * 1.3))),
                0, 0, 360, (40, 170, 40), -1)
    return img


def test_ida_y_vuelta_pixel_camara(K):
    p = np.array([120.0, -80.0, 900.0])
    assert np.allclose(pixel_a_camara(*camara_a_pixel(p, K), 900.0, K), p)


@pytest.mark.parametrize("distancia", [500, 800, 1200])
def test_estima_la_profundidad_por_el_tamano(distancia, foto, detectar_uno, en_base):
    p = en_base(detectar_uno(foto(np.array([0.0, 0.0, distancia]))))
    # CON q_de_frente LA CAMARA ESTA EN (0, 0, 1065) MIRANDO A +Y: LA Y DEL PEDUNCULO ES LA DISTANCIA.
    # 5 %: LA PRECISION REALISTA DE ESTIMAR DISTANCIA POR TAMAÑO (1-2 PX DE BORDE EN ~50 PX)
    assert p[1] == pytest.approx(distancia, rel=0.05)


def test_aguacate_al_centro_queda_enfrente_y_arriba_del_fruto(robot, foto, detectar_uno, en_base):
    x, y, z = en_base(detectar_uno(foto(np.array([0.0, 0.0, 800.0]))))
    assert abs(x) < 5                      # NI A LA IZQUIERDA NI A LA DERECHA
    assert robot.home[2] + 40 < z < robot.home[2] + 120  # EL CORTE VA ENCIMA DEL FRUTO, NO EN SU CENTRO


def test_los_ejes_de_la_imagen_corresponden_al_mundo(foto, detectar_uno, en_base):
    centro = en_base(detectar_uno(foto(np.array([0.0, 0.0, 800.0]))))
    derecha = en_base(detectar_uno(foto(np.array([200.0, 0.0, 800.0]))))
    arriba = en_base(detectar_uno(foto(np.array([0.0, -150.0, 800.0]))))
    # MIRANDO HACIA +Y CON Z ARRIBA, LA DERECHA DE LA FOTO ES +X DEL ROBOT Y ARRIBA ES +Z
    assert derecha[0] == pytest.approx(centro[0] + 200, abs=10)
    assert arriba[2] == pytest.approx(centro[2] + 150, abs=10)


@pytest.mark.parametrize("lado_mm", [-300.0, 0.0, 300.0])
def test_un_aguacate_enfrente_se_puede_cosechar(robot, foto, detectar_uno, en_base, lado_mm):
    # ANTES, NINGUNA DETECCION QUEDABA AL ALCANCE; Y CON LA BASE EN 0° LA MITAD DERECHA DE LA FOTO TAMPOCO
    assert alcanzable(robot, en_base(detectar_uno(foto(np.array([lado_mm, 50.0, 800.0])))))


def test_imagen_sin_aguacates(robot):
    _, detecciones, _, _ = detectar_aguacates(np.full((ALTO, ANCHO, 3), 40, np.uint8), robot.vision)
    assert detecciones == []


def test_camara_fija_no_depende_del_brazo(robot_4gdl):
    # CON MONTAJE "fija" LA POSE DE LA CAMARA ES LA DEL YAML, SE MUEVA COMO SE MUEVA EL BRAZO
    det = {"caja": (600, 300, 81, 104), "pedunculo_px": (640.0, 360.0)}
    a = pedunculo_en_base(det, ANCHO, ALTO, robot_4gdl, (0, 0, 0, 0))
    b = pedunculo_en_base(det, ANCHO, ALTO, robot_4gdl, (90, 45, 30, 10))
    assert np.allclose(a, b)
