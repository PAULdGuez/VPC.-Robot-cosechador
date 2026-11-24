"""Punta de herramienta (criterios P1 a P8) y los remates del volumen de trabajo (V7 y V8).

La herramienta es una extensión fija al final del antebrazo, alineada con él; su largo vive en los datos del
robot (`efector.largo`). Lo que se pide por coordenadas lo alcanza la PUNTA; la cámara sigue atornillada al
extremo del antebrazo.

Dónde queda la punta se mide aquí por otro camino que el del producto: trigonometría a mano (`punta_a_mano`),
sin las matrices de cosechador/robot/cinematica.py. Los números de «hoy» (P5 y P7) se congelaron el 24-nov-2026
con el código de fc3f2f2, antes de que existiera la herramienta.
"""
import itertools
import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import pyvista as pv
import yaml
from PyQt5 import QtCore, QtWidgets
from vtkmodules.vtkRenderingCore import vtkCoordinate

import auditoria_interfaz as au
from cosechador import registro as reg
from cosechador.movimiento import programa as prog
from cosechador.movimiento.trayectorias import (plan_cosecha, plan_ir_y_volver, posicion, servos_del_plan,
                                                tramo_continuo)
from cosechador.robot import ErrorConfiguracion, config, robot_desde_dict
from cosechador.robot.cinematica import servo_a_dh
from cosechador.robot.cinematica_inversa import angulos_servo
from cosechador.robot.config import ROBOT_POR_OMISION
from cosechador.robot.espacio_trabajo import contorno_del_corte, giro_de_la_base, rangos_servo
from cosechador.ui.escena3d import Escena
from cosechador.vision.camara import matriz_camara, pedunculo_en_base, pose_camara
from test_volumen import (MARGEN_MM, bases_al_azar, dentro, distancia, en_el_corte, frente, fuera_por, lados,
                          superficie_dibujada, tajada)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LO QUE HACE EL CÓDIGO DE HOY (fc3f2f2, SIN HERRAMIENTA) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CONGELADO EL 24-NOV-2026, ANTES DE TOCAR EL PRODUCTO. NO SE RECALCULA: ES LA VARA CON LA QUE SE MIDEN P5 Y P7.
HOY_HOME_SERVOS = (90.0, 45.573038, 45.643155)
HOY_DIRECTA = [  # SERVOS -> POSICIÓN (mm)
    ((90, 46, 46), (0.0, 5.34163, 1069.3398)),
    ((60, 110, 100), (515.692785, 893.206105, 1168.138896)),
    ((25, 135, 60), (805.055183, 375.403397, 380.958703)),
    ((0, 20, 0), (-281.907786, 0.0, 452.606043)),
    ((150, 120, 170), (-822.681981, 474.975663, 1752.256514)),
    ((63.402724, 99.964707, 72.677616), (356.000003, 711.000007, 1013.999996)),
]
HOY_INVERSA = [  # PUNTO (mm) -> SERVOS; None = NO SE ALCANZA
    ((0, 0, 1065), (90.0, 45.573038, 45.643155)),
    ((356, 711, 1014), (63.402724, 99.964707, 72.677616)),
    ((1656, 0, 630), (0.0, 162.96388, 161.901841)),
    ((430, 0, 1718), (0.0, 80.934695, 113.860872)),
    ((-370, 0, 1368), (180.0, 70.960851, 76.922514)),
    ((5, -4, 1057), (141.340192, 45.057026, 44.99071)),
    ((2000, 0, 2000), None),
]
HOY_CORTE = {"lazos": 1, "vertices": 354, "frente": 1700.0, "atras": -1597.477, "arriba": 2050.0, "abajo": -350.0,
             "area_mm2": 3909147.2, "suma": 333651.62}
DETECCION = {"caja": (900, 500, 121, 150), "pedunculo_px": (960.0, 480.0)}  # EN UNA FOTO DE 1920 x 1080
HOY_CAMARA = [  # SERVOS -> DÓNDE ESTÁ LA CÁMARA -> DÓNDE DICE LA VISIÓN QUE ESTÁ EL FRUTO DE `DETECCION`
    ((90, 46, 46), (0.0, 5.34163, 1069.3398), (0.0, 805.3416, 1109.3398)),
    ((60, 110, 100), (515.692785, 893.206105, 1168.138896), (913.0888, 1581.5163, 1068.6127)),
    ((25, 135, 60), (805.055183, 375.403397, 380.958703), (1027.728, 479.2374, -381.4292)),
]
HOY_COSECHA = {"pasos": 336, "cosechados": 9, "suma": 64474.395, "ultimo": (90.0, 45.573, 45.6432)}
HOY_CALCULAR = {"pasos": 58, "suma": 10529.144, "aviso": "Punto (356, 711, 1014) alcanzado; de regreso en HOME."}
HOY_SLIDERS_XYZ = [(-2050, 2050), (-2050, 2050), (0, 2050)]
HOY_VENTANA_MINIMA = (1243, 620)  # minimumSizeHint DE LA VENTANA RECIÉN CREADA, SIN MOSTRAR
# programas/demo-cosecha.yaml DESDE HOME: LO QUE SALE HACIA LA PLACA (46 PASOS) Y LO QUE ELLA RECIBE (GRADOS ENTEROS).
# PASOS 0-3: ARTICULAR AL PUNTO 1 · 4-22: LINEAL AL PUNTO 2 · 23-41: LINEAL AL PUNTO 3 · 42-45: ARTICULAR AL PUNTO 4
HOY_DEMO = [
    (83.3507, 58.4337, 54.4955), (76.7014, 71.2943, 63.3478), (70.052, 84.1549, 72.2001), (63.4027, 97.0155, 81.0524),
    (63.4027, 97.1319, 80.5798), (63.4027, 97.2524, 80.1107), (63.4027, 97.3773, 79.6452), (63.4027, 97.5064, 79.1833),
    (63.4027, 97.6398, 78.725), (63.4027, 97.7775, 78.2702), (63.4027, 97.9195, 77.819), (63.4027, 98.0659, 77.3713),
    (63.4027, 98.2166, 76.9271), (63.4027, 98.3716, 76.4864), (63.4027, 98.531, 76.0493), (63.4027, 98.6948, 75.6156),
    (63.4027, 98.863, 75.1854), (63.4027, 99.0355, 74.7587), (63.4027, 99.2125, 74.3355), (63.4027, 99.3939, 73.9158),
    (63.4027, 99.5797, 73.4996), (63.4027, 99.77, 73.0869), (63.4027, 99.9647, 72.6776), (63.4027, 99.77, 73.0869),
    (63.4027, 99.5797, 73.4996), (63.4027, 99.3939, 73.9158), (63.4027, 99.2125, 74.3355), (63.4027, 99.0355, 74.7587),
    (63.4027, 98.863, 75.1854), (63.4027, 98.6948, 75.6156), (63.4027, 98.531, 76.0493), (63.4027, 98.3716, 76.4864),
    (63.4027, 98.2166, 76.9271), (63.4027, 98.0659, 77.3713), (63.4027, 97.9195, 77.819), (63.4027, 97.7775, 78.2702),
    (63.4027, 97.6398, 78.725), (63.4027, 97.5064, 79.1833), (63.4027, 97.3773, 79.6452), (63.4027, 97.2524, 80.1107),
    (63.4027, 97.1319, 80.5798), (63.4027, 97.0155, 81.0524), (70.052, 84.1549, 72.2001), (76.7014, 71.2943, 63.3478),
    (83.3507, 58.4337, 54.4955), (90.0, 45.573, 45.6432),
]
HOY_DEMO_PLACA = [
    (83, 58, 54), (77, 71, 63), (70, 84, 72), (63, 97, 81), (63, 97, 81), (63, 97, 80), (63, 97, 80), (63, 98, 79),
    (63, 98, 79), (63, 98, 78), (63, 98, 78), (63, 98, 77), (63, 98, 77), (63, 98, 76), (63, 99, 76), (63, 99, 76),
    (63, 99, 75), (63, 99, 75), (63, 99, 74), (63, 99, 74), (63, 100, 73), (63, 100, 73), (63, 100, 73), (63, 100, 73),
    (63, 100, 73), (63, 99, 74), (63, 99, 74), (63, 99, 75), (63, 99, 75), (63, 99, 76), (63, 99, 76), (63, 98, 76),
    (63, 98, 77), (63, 98, 77), (63, 98, 78), (63, 98, 78), (63, 98, 79), (63, 98, 79), (63, 97, 80), (63, 97, 80),
    (63, 97, 81), (63, 97, 81), (70, 84, 72), (77, 71, 63), (83, 58, 54), (90, 46, 46),
]
DEMO_ARTICULARES, DEMO_LINEALES, DEMO_ENSENADAS = [0, 1, 2, 3, 42, 43, 44, 45], list(range(4, 42)), [3, 22, 41, 45]
# programas/ejemplo-grabacion.yaml DESDE HOME
HOY_GRABACION = [
    (90.0, 45.573, 45.6432), (103.3333, 52.2397, 55.6432), (116.6667, 58.9064, 65.6432), (130.0, 65.573, 75.6432),
    (116.0, 61.573, 69.6432), (102.0, 57.573, 63.6432), (88.0, 53.573, 57.6432), (74.0, 49.573, 51.6432),
    (60.0, 45.573, 45.6432), (75.0, 45.573, 45.6432), (90.0, 45.573, 45.6432),
]
HOY_GRABACION_PLACA = [(90, 46, 46), (103, 52, 56), (117, 59, 66), (130, 66, 76), (116, 62, 70), (102, 58, 64),
                       (88, 54, 58), (74, 50, 52), (60, 46, 46), (75, 46, 46), (90, 46, 46)]
HOY_4GDL_DIRECTA = [  # EL BRAZO DE PRUEBA DE 4 ARTICULACIONES (NO TRAE HERRAMIENTA)
    ((90, 90, 90, 90), (0.0, 630.0, 200.0)),
    ((30, 120, 40, 150), (440.480119, 254.311315, 185.710344)),
    ((170, 10, 95, 60), (-170.733911, 30.104995, -393.492325)),
]
HOY_4GDL_CORTE = {"vertices": 127, "frente": 630.0, "atras": -330.0, "arriba": 830.0, "abajo": -430.0, "suma": 42776.678}

# EN programas/demo-cosecha.yaml, CON LA HERRAMIENTA DE 120 mm, LAS POSTURAS INTERMEDIAS DE LOS TRAMOS LINEALES SE
# APARTAN DE LAS DE HOY, COMO MUCHO, ESTO (GRADOS DE SERVO). MEDIDO EL 24-NOV: 0.0344° EN EL PEOR PASO (0.0144° DE
# MEDIA); YA REDONDEADO A GRADOS ENTEROS, A LA PLACA LE LLEGA 1° DISTINTO EN 4 DE LOS 46 PASOS.
COTA_TRAMO_LINEAL = 0.05
# AL SALIR DE HOME (LA PUNTA SOBRE EL EJE DE LA BASE) LA BASE GIRA Y LA PUNTA SE MUEVE UNAS DÉCIMAS DE MILÍMETRO:
# 0.04 mm CON CALCULAR Y 0.11 mm CON LOS SLIDERS, IGUAL CON HERRAMIENTA QUE SIN ELLA (MEDIDO EL 24-NOV).
RECTA_MM = 0.5


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INSTRUMENTOS DE MEDICIÓN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def punta_a_mano(robot, servos, largo=None):
    """Dónde queda la punta de la herramienta (mm), por trigonometría y sin la cinemática del producto.

    Vale para los brazos de este repo: columna vertical que gira, y el resto de los eslabones en el plano del
    frente de la base. La herramienta prolonga el último eslabón. `largo=0` da el extremo del antebrazo.
    """
    largo = robot.efector.largo if largo is None else largo
    base, *resto = robot.articulaciones
    q = [j.servo.signo * (float(s) - j.servo.offset) for s, j in zip(servos, robot.articulaciones)]
    alcance, altura, inclinacion = 0.0, base.d, 0.0
    for angulo, j in zip(q[1:], resto):
        inclinacion += math.radians(angulo)
        alcance += j.a * math.cos(inclinacion)
        altura += j.a * math.sin(inclinacion)
    alcance += largo * math.cos(inclinacion)
    altura += largo * math.sin(inclinacion)
    return np.array([alcance * math.cos(math.radians(q[0])), alcance * math.sin(math.radians(q[0])), altura])


def puntas_a_mano(robot, servo_base, rng, largo=None, paso=5, al_azar=1500):
    """La punta, a mano, con la base en ese ángulo y los demás servos en todo su rango: cada `paso` grados (con los
    dos límites) y además al azar."""
    resto = robot.articulaciones[1:]
    servos = list(itertools.product(*rangos_servo(robot, paso)[1:]))
    servos += [tuple(rng.uniform(j.servo.minimo, j.servo.maximo) for j in resto) for _ in range(al_azar)]
    return np.array([punta_a_mano(robot, (servo_base, *s), largo) for s in servos])


def lejos_de_la_recta(puntos, desde, hasta):
    """Cuánto se aparta (mm), como mucho, un grupo de puntos del segmento que va de `desde` a `hasta`."""
    puntos, desde, hasta = np.asarray(puntos, float), np.asarray(desde, float), np.asarray(hasta, float)
    a_lo_largo = np.clip((puntos - desde) @ (hasta - desde) / ((hasta - desde) @ (hasta - desde)), 0, 1)
    return np.linalg.norm(puntos - (desde + a_lo_largo[:, None] * (hasta - desde)), axis=1).max()


def herramienta_dibujada(escena):
    """(x mínima, x máxima) de la pieza del efector en su propio marco, y sus puntos ya puestos en el mundo."""
    actor = escena.actores[-1]
    malla = actor.mapper.dataset
    en_el_mundo = (np.asarray(actor.user_matrix) @ np.column_stack([malla.points, np.ones(malla.n_points)]).T).T[:, :3]
    return malla.bounds[:2], en_el_mundo


def foto(escena):
    escena.plotter.render()
    return escena.plotter.screenshot(return_img=True).astype(int)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ROBOTS Y VENTANAS DE PRUEBA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@pytest.fixture
def datos_yaml():
    with open(ROBOT_POR_OMISION, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(params=["largo cero", "archivo sin largo"])
def como_hoy(request, robot, datos_yaml):
    """El cosechador sin herramienta, de las dos maneras de decirlo."""
    if request.param == "largo cero":
        return config.con_herramienta(robot, 0)
    datos_yaml["efector"].pop("largo", None)
    return robot_desde_dict(datos_yaml, carpeta_modelos=robot.carpeta_modelos)


@pytest.fixture
def abrir(tmp_path):
    """Abre la ventana de la app (QT_QPA_PLATFORM=offscreen: nada a la vista). Los ajustes de la persona viven en
    un archivo de la prueba: volver a abrir con los mismos ajustes es cerrar la app y abrirla otra vez."""
    from cosechador.ui import ventana
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    (tmp_path / "programas").mkdir()
    abiertas = []

    def abrir(robot, programas=None):
        ajustes = QtCore.QSettings(str(tmp_path / "ajustes.ini"), QtCore.QSettings.IniFormat)
        w = ventana.VentanaPrincipal(robot, ajustes=ajustes, carpeta_programas=programas or str(tmp_path / "programas"),
                                     registro=reg.Registro(carpeta=None))
        w.enviados = []  # LO QUE SALE HACIA LA PLACA, ANTES DE QUE ELLA LO REDONDEE A GRADOS ENTEROS
        enviar = w.enlace.enviar
        w.enlace.enviar = lambda *servos: (w.enviados.append(tuple(float(s) for s in servos)), enviar(*servos))[1]
        abiertas.append(w)
        return w

    def cerrar(w):
        abiertas.remove(w)
        w.close()
        app.processEvents()

    abrir.cerrar = cerrar
    yield abrir
    for w in abiertas:
        w.close()
    app.processEvents()


def correr(w):
    """Ejecuta lo que haya en la cola, paso por paso, sin esperar al temporizador."""
    w.temporizador_movimiento.stop()
    while w.cola:
        w._siguiente_paso()
    w._siguiente_paso()  # COLA VACÍA: EL MOVIMIENTO TERMINA (Y SE REACTIVAN LOS CONTROLES)


def ir_a(w, punto):
    """Escribe el punto en X, Y, Z y presiona CALCULAR."""
    for campo, valor in zip((w.TE_X, w.TE_Y, w.TE_Z), punto):
        campo.setText(str(valor))
    w.CALCULAR.click()
    correr(w)


def reproducir(w, archivo, desde):
    """Reproduce un programa del repo saliendo de esa pose: en w.enviados queda lo que salió hacia la placa."""
    w.poner_servos(desde)
    w.enviados.clear()
    w.enlace.dispositivo.historial.clear()
    w.SELECTOR_PROGRAMA.setCurrentIndex(w.SELECTOR_PROGRAMA.findData(archivo))
    w.BTN_REPRODUCIR.click()
    correr(w)


def ultimo_aviso(w):
    return w.registro.filtrar(reg.NIVELES)[-1]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EL INSTRUMENTO, CONTRA LO QUE YA SE SABE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_la_medicion_a_mano_da_lo_de_hoy_cuando_no_hay_herramienta(robot, robot_4gdl):
    for servos, punto in HOY_DIRECTA:
        assert punta_a_mano(robot, servos, largo=0) == pytest.approx(punto, abs=1e-5)
    for servos, punto in HOY_4GDL_DIRECTA:
        assert punta_a_mano(robot_4gdl, servos, largo=0) == pytest.approx(punto, abs=1e-5)
    # Y CON HERRAMIENTA, LA PUNTA QUEDA ESE LARGO MÁS ALLÁ, SOBRE LA DIRECCIÓN DEL ANTEBRAZO
    extremo, punta = punta_a_mano(robot, (60, 110, 100), largo=0), punta_a_mano(robot, (60, 110, 100), largo=120)
    codo = np.array([0.0, 0.0, robot.articulaciones[0].d]) + 1000 * np.array(
        [math.cos(math.radians(70)) * math.cos(math.radians(60)), math.cos(math.radians(70)) * math.sin(math.radians(60)),
         math.sin(math.radians(70))])
    assert np.linalg.norm(punta - extremo) == pytest.approx(120)
    assert np.cross(punta - extremo, extremo - codo) == pytest.approx((0, 0, 0), abs=1e-6)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% P1: EL LARGO VIVE EN LOS DATOS Y ASÍ SE DIBUJA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_p1_el_largo_de_la_herramienta_vive_en_los_datos_del_robot(datos_yaml, robot, robot_4gdl):
    assert datos_yaml["efector"]["largo"] == 120 and robot.efector.largo == 120  # EL ARCHIVO DEL COSECHADOR (PROVISIONAL)
    datos_yaml["efector"]["largo"] = 87.5
    assert robot_desde_dict(datos_yaml).efector.largo == 87.5
    del datos_yaml["efector"]["largo"]  # UN ARCHIVO QUE NO LO DICE: SIN HERRAMIENTA
    assert robot_desde_dict(datos_yaml).efector.largo == 0
    assert robot_4gdl.efector.largo == 0  # EL BRAZO DE PRUEBA NI SIQUIERA TRAE LA SECCIÓN
    # EL COLOR Y LA PIEZA DEL EFECTOR SIGUEN SALIENDO DEL MISMO LUGAR
    assert (robot.efector.archivo, robot.efector.color) == ("efector.stl", "#c9d1d9")


@pytest.mark.parametrize("largo", [-1, "largo", True, None, [120], float("nan"), float("inf"), 1e6])
def test_p1_un_largo_que_no_es_una_medida_se_rechaza_diciendo_cual_campo(datos_yaml, largo):
    datos_yaml["efector"]["largo"] = largo
    with pytest.raises(ErrorConfiguracion, match="efector.largo"):
        robot_desde_dict(datos_yaml)


@pytest.mark.parametrize("largo", [120, 87.5, 300])
def test_p1_la_herramienta_se_dibuja_con_su_largo(robot, largo):
    con = config.con_herramienta(robot, largo)
    escena = Escena(pv.Plotter(off_screen=True), con)
    servos = (60, 110, 100)
    escena.dibujar(servo_a_dh(con, servos))
    (desde, hasta), puntos = herramienta_dibujada(escena)
    # EN SU PROPIO MARCO NACE EN EL EXTREMO DEL ANTEBRAZO (X = 0) Y ACABA A SU LARGO...
    assert (desde, hasta) == pytest.approx((0, largo), abs=1e-6)
    # ...Y EN EL MUNDO, SU PUNTO MÁS LEJANO DEL ANTEBRAZO ES LA PUNTA MEDIDA A MANO
    extremo = punta_a_mano(con, servos, largo=0)
    lejano = puntos[np.argmax(np.linalg.norm(puntos - extremo, axis=1))]
    assert lejano == pytest.approx(punta_a_mano(con, servos), abs=1e-3)


def test_p1_sin_herramienta_no_se_dibuja_nada_que_sobresalga(robot):
    # LARGO CERO: LA PIEZA DEL EFECTOR NO MIDE NADA A LO LARGO DEL ANTEBRAZO (NI EL CONO DE ADORNO QUE HABÍA ANTES)
    sin = config.con_herramienta(robot, 0)
    escena = Escena(pv.Plotter(off_screen=True), sin)
    escena.dibujar(servo_a_dh(sin, (60, 110, 100)))
    (desde, hasta), _ = herramienta_dibujada(escena)
    assert (desde, hasta) == pytest.approx((0, 0), abs=1e-9)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% P2: A UN PUNTO PEDIDO POR COORDENADAS LLEGA LA PUNTA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@pytest.mark.parametrize("largo", [120, 300])
def test_p2_la_inversa_lleva_la_punta_al_punto(robot, largo):
    con = config.con_herramienta(robot, largo)
    rng = np.random.default_rng(2)
    probados = 0
    for _ in range(400):
        punto = punta_a_mano(con, [rng.uniform(j.servo.minimo, j.servo.maximo) for j in con.articulaciones])
        servos = angulos_servo(con, punto)
        if servos is None:
            continue
        assert punta_a_mano(con, servos) == pytest.approx(punto, abs=1e-6)                   # LLEGA LA PUNTA...
        assert np.linalg.norm(punta_a_mano(con, servos, largo=0) - punto) == pytest.approx(largo, abs=1e-6)  # ...NO EL ANTEBRAZO
        probados += 1
    assert probados > 300


def test_p2_calcular_lleva_la_punta_al_punto_escrito(robot, abrir):
    w = abrir(robot)
    destino = (356, 711, 1014)
    salida = punta_a_mano(robot, w.servos)
    ir_a(w, destino)
    puntas = np.array([punta_a_mano(robot, s) for s in w.enviados])
    llegada = int(np.argmin(np.linalg.norm(puntas - destino, axis=1)))
    assert puntas[llegada] == pytest.approx(destino, abs=1e-3)
    assert np.linalg.norm(punta_a_mano(robot, w.enviados[llegada], largo=0) - destino) == pytest.approx(120, abs=1e-3)
    assert lejos_de_la_recta(puntas[:llegada + 1], salida, destino) < RECTA_MM  # Y LA PUNTA FUE EN LÍNEA RECTA
    assert ultimo_aviso(w).mensaje == "Punto (356, 711, 1014) alcanzado; de regreso en HOME."
    assert punta_a_mano(robot, w.servos) == pytest.approx(robot.home, abs=1e-3)  # HOME TAMBIÉN ES DE LA PUNTA


def test_p2_la_pestana_punta_xyz_mueve_la_punta_y_la_muestra(robot, abrir):
    w = abrir(robot)
    # LOS SLIDERS LLEGAN HASTA DONDE LLEGA LA PUNTA: EL ALCANCE DEL BRAZO (2050) MÁS LA HERRAMIENTA (120)
    assert [(s.minimum(), s.maximum()) for s in w.sliders_xyz] == [(-2170, 2170), (-2170, 2170), (0, 2170)]
    salida = punta_a_mano(robot, w.servos)
    w.PESTANAS_MOVIMIENTO.setCurrentIndex(1)
    w.SLIDER_X.setValue(300)  # DE HOME (0, 0, 1065) A (300, 0, 1065)
    correr(w)
    puntas = np.array([punta_a_mano(robot, s) for s in w.enviados])
    assert puntas[-1] == pytest.approx((300, 0, 1065), abs=1e-3)
    assert lejos_de_la_recta(puntas, salida, (300, 0, 1065)) < RECTA_MM
    assert [valor.text() for valor in w.valores_xyz] == ["300", "0", "1065"]  # LAS COORDENADAS QUE SE VEN SON LAS DE LA PUNTA
    assert w.posicion == pytest.approx(puntas[-1], abs=1e-6)


@pytest.mark.parametrize("largo", [0, 60, 120, 200, 300])
def test_p2_al_salir_de_home_la_base_voltea_hacia_el_punto(robot, largo):
    # EN HOME LA PUNTA ESTÁ SOBRE EL EJE DE LA BASE. PARA IR A UN LADO, LA BASE PUEDE VOLTEAR HACIA EL PUNTO O DARLE LA
    # ESPALDA (Y EL BRAZO DOBLARSE HACIA ATRÁS): LOS DOS GIROS MIDEN 90°. QUE NO LO DECIDA EL RUIDO NUMÉRICO: CON
    # HERRAMIENTAS DE 60, 120 Y 200 mm LE DABA LA ESPALDA Y A LOS 121-278 mm TENÍA QUE CAMBIAR DE POSTURA.
    con = config.con_herramienta(robot, largo)
    home = angulos_servo(con, con.home)
    for destino, base in (((300, 0, 1065), 0), ((-300, 0, 1065), 180)):
        plan, motivo = tramo_continuo(con, home, destino, "negar")
        assert motivo is None, (largo, destino, motivo)
        llegada = servos_del_plan(plan)[-1]
        assert llegada[0] == pytest.approx(base, abs=1e-6)
        assert punta_a_mano(con, llegada) == pytest.approx(destino, abs=1e-6)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% P3: LA LÍNEA NARANJA ES DE LA PUNTA, Y LA PUNTA NO SALE DEL VOLUMEN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_p3_la_linea_naranja_la_dibuja_la_punta(robot, abrir):
    w = abrir(robot)
    salida = punta_a_mano(robot, w.servos)
    ir_a(w, (356, 711, 1014))
    estela = np.array(w.escena.puntos_recorrido)
    puntas = np.array([salida] + [punta_a_mano(robot, s) for s in w.enviados])
    assert len(estela) > 30
    # CADA PUNTO DE LA LÍNEA ES UN LUGAR DONDE ESTUVO LA PUNTA, Y LA LÍNEA PASA POR EL PUNTO PEDIDO
    assert np.linalg.norm(estela[:, None] - puntas[None], axis=2).min(axis=1).max() < 1e-6
    assert np.linalg.norm(estela - (356, 711, 1014), axis=1).min() < 1e-3


def test_p3_la_punta_nunca_queda_fuera_del_volumen_dibujado(robot, abrir):
    w = abrir(robot)
    w.VER_VOLUMEN.click()
    rng = np.random.default_rng(24)
    for servo_base in bases_al_azar(robot, rng):
        puntas = en_el_corte(robot, servo_base, puntas_a_mano(robot, servo_base, rng))
        assert fuera_por(tajada(superficie_dibujada(w), robot, servo_base), puntas).max() <= MARGEN_MM
    # EL VOLUMEN ES EL DE LA PUNTA: LLEGA 120 mm MÁS LEJOS QUE EL DEL EXTREMO DEL ANTEBRAZO (EL DE HOY)
    arriba = superficie_dibujada(w).bounds[5]
    assert arriba == pytest.approx(HOY_CORTE["arriba"] + 120, abs=0.01)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% P4: COSECHAR LLEVA LA PUNTA AL FRUTO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_p4_cosechar_lleva_la_punta_a_cada_fruto(robot, abrir):
    w = abrir(robot)
    frutos = [(356, 711, 1014), (430, 0, 1718), (-370, 0, 1368)]
    w.puntos_cosecha = list(frutos)
    w.COSECHAR.click()
    correr(w)
    puntas = np.array([punta_a_mano(robot, s) for s in w.enviados])
    for fruto in frutos:
        llegada = int(np.argmin(np.linalg.norm(puntas - fruto, axis=1)))
        assert puntas[llegada] == pytest.approx(fruto, abs=1e-3), fruto
        assert np.linalg.norm(punta_a_mano(robot, w.enviados[llegada], largo=0) - fruto) == pytest.approx(120, abs=1e-3)
    assert ultimo_aviso(w).mensaje == "Cosecha terminada: 3 de 3 puntos alcanzados."


def test_p4_los_frutos_de_demostracion_siguen_al_alcance_de_la_punta(robot):
    plan, cosechados = plan_cosecha(robot, angulos_servo(robot, robot.home), robot.puntos_demo)
    assert cosechados == len(robot.puntos_demo) == HOY_COSECHA["cosechados"]
    puntas = np.array([punta_a_mano(robot, s) for s in servos_del_plan(plan)])
    for fruto in robot.puntos_demo:
        assert np.linalg.norm(puntas - fruto, axis=1).min() < 1e-3, fruto


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% P5: LOS PROGRAMAS GUARDADOS SE REPRODUCEN IGUAL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_p5_el_programa_por_puntos_pasa_por_las_mismas_poses(robot, abrir):
    w = abrir(robot, programas=prog.CARPETA_PROGRAMAS)
    reproducir(w, "demo-cosecha.yaml", desde=HOY_HOME_SERVOS)
    mandado, hoy = np.array(w.enviados), np.array(HOY_DEMO)
    assert mandado.shape == hoy.shape  # LOS MISMOS 46 PASOS: NINGÚN TRAMO DEJÓ DE PODERSE
    assert ultimo_aviso(w).mensaje == "Programa «Demo de cosecha» terminado."
    # LAS POSES ENSEÑADAS (LO QUE GUARDA EL ARCHIVO: ÁNGULOS DE SERVO) SE ALCANZAN TAL CUAL...
    ensenadas = prog.cargar(os.path.join(prog.CARPETA_PROGRAMAS, "demo-cosecha.yaml"), robot).puntos
    assert mandado[DEMO_ENSENADAS] == pytest.approx(np.array([p.servos for p in ensenadas]), abs=1e-6)
    # ...LOS TRAMOS ARTICULARES SON LOS DE HOY...
    assert mandado[DEMO_ARTICULARES] == pytest.approx(hoy[DEMO_ARTICULARES], abs=1e-4)
    # ...Y EN LOS LINEALES LO QUE VA RECTO ES AHORA LA PUNTA: LAS POSTURAS INTERMEDIAS SE APARTAN POCO DE LAS DE HOY
    for tramo in (range(3, 23), range(22, 42)):
        puntas = [punta_a_mano(robot, mandado[i]) for i in tramo]
        assert lejos_de_la_recta(puntas, puntas[0], puntas[-1]) < 0.01
    assert np.abs(mandado - hoy).max() <= COTA_TRAMO_LINEAL
    # A LA PLACA (GRADOS ENTEROS) LE LLEGA, COMO MUCHO, 1° DISTINTO DE LO DE HOY
    assert np.abs(np.array(w.enlace.dispositivo.historial) - np.array(HOY_DEMO_PLACA)).max() <= 1


def test_p5_la_grabacion_se_reproduce_identica(robot, abrir):
    w = abrir(robot, programas=prog.CARPETA_PROGRAMAS)
    reproducir(w, "ejemplo-grabacion.yaml", desde=HOY_HOME_SERVOS)
    assert np.array(w.enviados) == pytest.approx(np.array(HOY_GRABACION), abs=1e-4)
    assert w.enlace.dispositivo.historial == HOY_GRABACION_PLACA


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% P6: LA CÁMARA NO CAMBIA DE LUGAR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@pytest.mark.parametrize("largo", [0, 120, 300])
def test_p6_la_camara_sigue_en_el_extremo_del_antebrazo_y_mide_lo_mismo(robot, largo):
    con = config.con_herramienta(robot, largo)
    for servos, donde_esta, fruto in HOY_CAMARA:
        q = servo_a_dh(con, servos)
        assert pose_camara(con, q)[:3, 3] == pytest.approx(donde_esta, abs=1e-5)
        assert pose_camara(con, q)[:3, 3] == pytest.approx(punta_a_mano(con, servos, largo=0), abs=1e-6)
        assert pedunculo_en_base(DETECCION, 1920, 1080, con, q) == pytest.approx(fruto, abs=1e-3)


def test_p6_lo_que_mide_la_vision_cae_en_el_fruto_y_ahi_llega_la_punta(robot):
    # UN FRUTO PUESTO A MANO: 600 mm DELANTE DEL EXTREMO DEL ANTEBRAZO, SOBRE SU EJE (A DONDE MIRA LA CÁMARA)
    servos = (60, 110, 100)
    extremo = punta_a_mano(robot, servos, largo=0)
    fruto = extremo + 600 * (punta_a_mano(robot, servos, largo=1) - extremo)
    # ASÍ LO VE LA CÁMARA: AL CENTRO DE LA FOTO Y DEL ANCHO QUE LE TOCA A ESA DISTANCIA
    K = matriz_camara(robot.camara, 1920, 1080)
    ancho_px = K[0, 0] * robot.camara.diametro_objeto_mm / 600 + 1
    deteccion = {"caja": (0, 0, ancho_px, ancho_px), "pedunculo_px": (960.0, 540.0)}
    medido = pedunculo_en_base(deteccion, 1920, 1080, robot, servo_a_dh(robot, servos))
    assert medido == pytest.approx(fruto, abs=1e-6)                                          # LA VISIÓN LO UBICA BIEN...
    assert punta_a_mano(robot, angulos_servo(robot, medido)) == pytest.approx(fruto, abs=1e-6)  # ...Y AHÍ LLEGA LA PUNTA


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% P7: CON LARGO CERO, TODO QUEDA COMO HOY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_p7_la_directa_y_la_inversa_dan_lo_de_hoy(como_hoy):
    assert como_hoy.efector.largo == 0
    for servos, punto in HOY_DIRECTA:
        assert posicion(como_hoy, servos) == pytest.approx(punto, abs=1e-5)
    for punto, servos in HOY_INVERSA:
        if servos is None:
            assert angulos_servo(como_hoy, punto) is None
        else:
            assert angulos_servo(como_hoy, punto) == pytest.approx(servos, abs=1e-5)


def test_p7_el_volumen_es_el_de_hoy(como_hoy):
    lazos = contorno_del_corte(como_hoy)
    todos = np.concatenate(lazos)
    area = sum(0.5 * abs(np.sum(l[:, 0] * np.roll(l[:, 1], -1) - np.roll(l[:, 0], -1) * l[:, 1])) for l in lazos)
    medido = {"lazos": len(lazos), "vertices": len(todos), "frente": todos[:, 0].max(), "atras": todos[:, 0].min(),
              "arriba": todos[:, 1].max(), "abajo": todos[:, 1].min(), "area_mm2": area, "suma": todos.sum()}
    assert medido == pytest.approx(HOY_CORTE, abs=0.06)


def test_p7_la_camara_mide_lo_de_hoy(como_hoy):
    for servos, donde_esta, fruto in HOY_CAMARA:
        q = servo_a_dh(como_hoy, servos)
        assert pose_camara(como_hoy, q)[:3, 3] == pytest.approx(donde_esta, abs=1e-5)
        assert pedunculo_en_base(DETECCION, 1920, 1080, como_hoy, q) == pytest.approx(fruto, abs=1e-3)


def test_p7_cosechar_calcular_y_la_recta_planean_lo_de_hoy(como_hoy):
    home = angulos_servo(como_hoy, como_hoy.home)
    plan, cosechados = plan_cosecha(como_hoy, home, como_hoy.puntos_demo)
    servos = np.array(servos_del_plan(plan))
    assert (len(servos), cosechados) == (HOY_COSECHA["pasos"], HOY_COSECHA["cosechados"])
    assert servos.sum() == pytest.approx(HOY_COSECHA["suma"], abs=2e-3)
    assert servos[-1] == pytest.approx(HOY_COSECHA["ultimo"], abs=1e-4)

    plan = plan_ir_y_volver(como_hoy, home, (356, 711, 1014))
    servos = np.array(servos_del_plan(plan))
    assert (len(servos), plan[-1].aviso) == (HOY_CALCULAR["pasos"], HOY_CALCULAR["aviso"])
    assert servos.sum() == pytest.approx(HOY_CALCULAR["suma"], abs=2e-3)

    recta, motivo = tramo_continuo(como_hoy, (63.402724, 97.015548, 81.052423), (356, 711, 1014))
    assert motivo is None
    assert np.array(servos_del_plan(recta)) == pytest.approx(np.array(HOY_DEMO[4:23]), abs=1e-4)


def test_p7_la_ventana_arranca_y_reproduce_como_hoy(como_hoy, abrir):
    w = abrir(como_hoy, programas=prog.CARPETA_PROGRAMAS)
    assert w.servos == pytest.approx(HOY_HOME_SERVOS, abs=1e-5)
    assert w.posicion == pytest.approx((0, 0, 1065), abs=1e-4)
    assert [(s.minimum(), s.maximum()) for s in w.sliders_xyz] == HOY_SLIDERS_XYZ
    reproducir(w, "demo-cosecha.yaml", desde=w.servos)
    assert np.array(w.enviados) == pytest.approx(np.array(HOY_DEMO), abs=1e-4)
    assert w.enlace.dispositivo.historial == HOY_DEMO_PLACA


def test_p7_el_brazo_de_4_articulaciones_queda_como_hoy(robot_4gdl):
    for servos, punto in HOY_4GDL_DIRECTA:
        assert posicion(robot_4gdl, servos) == pytest.approx(punto, abs=1e-5)
    (lazo,) = contorno_del_corte(robot_4gdl, paso=10)
    medido = {"vertices": len(lazo), "frente": lazo[:, 0].max(), "atras": lazo[:, 0].min(),
              "arriba": lazo[:, 1].max(), "abajo": lazo[:, 1].min(), "suma": lazo.sum()}
    assert medido == pytest.approx(HOY_4GDL_CORTE, abs=2e-3)
    # Y SI A ESE BRAZO (SIN SOLVER DE INVERSA) SE LE PONE HERRAMIENTA, SU DIRECTA TAMBIÉN DA LA PUNTA
    con = config.con_herramienta(robot_4gdl, 55)
    for servos, _ in HOY_4GDL_DIRECTA:
        assert posicion(con, servos) == pytest.approx(punta_a_mano(con, servos), abs=1e-6)
        assert np.linalg.norm(posicion(con, servos) - posicion(robot_4gdl, servos)) == pytest.approx(55)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% V8: EL BOTÓN DEL VOLUMEN TAMBIÉN LO OCULTA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_v8_el_boton_del_volumen_tambien_lo_oculta(robot, abrir):
    w = abrir(robot)
    actores = lambda: len(w.escena.plotter.renderer.actors)
    sin_volumen, rotulo = actores(), w.VER_VOLUMEN.text()
    assert rotulo == "VOLUMEN DE TRABAJO"

    w.VER_VOLUMEN.click()
    con_volumen = actores()
    assert len(w.escena.actores_volumen) == 2 and con_volumen > sin_volumen
    assert w.VER_VOLUMEN.text() == "OCULTAR VOLUMEN"
    assert w.PROCESO.value() == w.PROCESO.maximum() > 0

    w.VER_VOLUMEN.click()
    assert w.escena.actores_volumen == [] and actores() == sin_volumen  # NI LA SUPERFICIE, NI SU CONTORNO, NI LA REJILLA
    assert w.VER_VOLUMEN.text() == rotulo and w.PROCESO.value() == 0

    w.VER_VOLUMEN.click()  # Y SE VUELVE A VER, SIN ENCIMARSE
    assert len(w.escena.actores_volumen) == 2 and actores() == con_volumen
    assert w.VER_VOLUMEN.text() == "OCULTAR VOLUMEN"


def test_v8_al_ocultarlo_la_escena_se_ve_como_antes(robot):
    escena = Escena(pv.Plotter(off_screen=True, window_size=(640, 480)), robot)
    escena.dibujar(servo_a_dh(robot, (60, 110, 100)))
    escena.vista("iso")
    escena.encuadrar(zoom=1.0)
    antes = foto(escena)
    escena.mostrar_superficie(contorno_del_corte(robot), giro_de_la_base(robot))
    distintos = lambda imagen: int((np.abs(imagen - antes).sum(axis=2) > 30).sum())
    assert distintos(foto(escena)) > 5000  # CON EL VOLUMEN, LA IMAGEN CAMBIA
    escena.ocultar_volumen()
    assert distintos(foto(escena)) == 0    # Y AL OCULTARLO QUEDA IGUAL QUE ANTES, PIXEL POR PIXEL
