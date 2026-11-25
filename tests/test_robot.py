"""Modelo del robot: configuración, cinemática directa e inversa, servos y volumen de trabajo."""
import copy
import subprocess
import sys

import numpy as np
import pytest
import yaml

from cosechador.robot import ErrorConfiguracion, cargar_robot, robot_desde_dict
from cosechador.robot.cinematica import (TOLERANCIA_SERVO, cinematica_directa, dh_a_servo, matrices, servo_a_dh,
                                         servos_en_rango)
from cosechador.robot.cinematica_inversa import (FueraDeAlcance, SinSolver, alcanzable, angulos_servo,
                                                 angulos_servo_cercanos, soluciones)
from cosechador.robot.config import ROBOT_POR_OMISION
from cosechador.robot.espacio_trabajo import barrer, rangos_servo, total_puntos


@pytest.fixture
def datos_yaml():
    with open(ROBOT_POR_OMISION, encoding="utf-8") as f:
        return yaml.safe_load(f)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CINEMATICA (COMPORTAMIENTO DE SIEMPRE) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_home_es_alcanzable_y_coincide_con_los_sliders(robot):
    # HOME ES UNA POSTURA: EL EXTREMO DEL ANTEBRAZO ESTA SOBRE EL EJE DE LA BASE (LA BASE VA AL CENTRO DE SU RANGO, 90°),
    # CON BRAZO Y ANTEBRAZO COMO LOS SLIDERS ORIGINALES (45, 45) QUE ERAN (90, 46, 46), TENGA O NO HERRAMIENTA
    assert tuple(round(s) for s in robot.postura_home) == (90, 46, 46)
    assert alcanzable(robot, robot.home) and cinematica_directa(robot, servo_a_dh(robot, robot.postura_home)) == pytest.approx(robot.home, abs=1e-6)


@pytest.mark.parametrize("servos", [(0, 20, 0), (90, 45, 45), (180, 180, 180), (30, 100, 120), (150, 60, 10)])
def test_ida_y_vuelta_servo_dh(robot, servos):
    assert np.allclose(dh_a_servo(robot, servo_a_dh(robot, servos)), servos)


def test_ci_es_la_inversa_de_cd(robot):
    rng = np.random.default_rng(7)
    probados = 0
    for _ in range(500):
        servos = [rng.uniform(j.servo.minimo, j.servo.maximo) for j in robot.articulaciones]
        punto = cinematica_directa(robot, servo_a_dh(robot, servos))
        servos_ci = angulos_servo(robot, punto)
        if servos_ci is None:
            continue
        assert np.allclose(cinematica_directa(robot, servo_a_dh(robot, servos_ci)), punto, atol=1e-6)
        probados += 1
    assert probados > 400


def test_ci_usa_la_altura_de_la_base(robot):
    d1, a2, a3 = robot.articulaciones[0].d, robot.articulaciones[1].a, robot.articulaciones[2].a
    estirado = d1 + a2 + a3 + robot.efector.largo  # HASTA DONDE LLEGA LA PUNTA DE LA HERRAMIENTA
    assert soluciones(robot, (0.001, 0, estirado - 1))
    with pytest.raises(FueraDeAlcance):
        soluciones(robot, (0, 0, estirado + 10))


def test_un_empate_de_acomodos_no_lo_decide_el_ruido(robot):
    # A UN LADO DE HOME, LA BASE A 0° Y A 180° QUEDAN IGUAL DE LEJOS (90°): GANA LA QUE MIRA AL PUNTO, AUNQUE EL
    # RUIDO NUMÉRICO PONGA EL PUNTO UN PELO DEL OTRO LADO DE y = 0. (SI NO, EL BRAZO SALE DE HOME DE ESPALDAS AL
    # PUNTO Y A LOS POCOS CENTÍMETROS TIENE QUE CAMBIAR DE POSTURA.)
    home = angulos_servo(robot, robot.punto_de_giro)  # EL PUNTO SOBRE EL EJE, DONDE LA BASE A 0° Y A 180° EMPATAN
    for ruido in (-1e-13, 0.0, 1e-13):
        assert angulos_servo_cercanos(robot, (0.4, ruido, 1065), home)[0] == pytest.approx(0, abs=1e-6)
        assert angulos_servo_cercanos(robot, (-0.4, ruido, 1065), home)[0] == pytest.approx(180, abs=1e-6)


def test_punto_fuera_de_alcance(robot):
    assert angulos_servo(robot, (5000, 0, 0)) is None
    assert not alcanzable(robot, (5000, 0, 0))


def test_punto_que_pide_un_servo_fuera_de_rango(robot):
    assert angulos_servo(robot, (500, -500, 800)) is None
    assert angulos_servo(robot, (500, 500, 800)) is not None


def test_ruido_de_punto_flotante_en_la_base_no_saca_de_rango(robot):
    home_real = cinematica_directa(robot, servo_a_dh(robot, angulos_servo(robot, robot.home)))
    destino = np.array([1656.0, 0.0, 630.0])
    for k in np.linspace(0, 1, 20):
        assert angulos_servo(robot, home_real + (destino - home_real) * k) is not None
    assert dh_a_servo(robot, (-1e-10, 90, -90))[0] == pytest.approx(0, abs=1e-6)


def test_volumen_respeta_rango_de_servos(robot):
    avances = []
    puntos = barrer(robot, paso=10, progreso=lambda h, t: avances.append((h, t)))
    assert avances[-1] == (len(puntos), len(puntos)) == (total_puntos(robot, 10),) * 2
    assert all(alcanzable(robot, p) for p in puntos)


def test_el_barrido_llega_a_los_dos_limites_de_cada_servo(datos_yaml):
    # UN RANGO QUE NO ES MÚLTIPLO DEL PASO (HOMBRO DE 20° A 175°, CADA 10°): EL BARRIDO SE QUEDABA EN 170° Y
    # EL VOLUMEN DIBUJADO NO LLEGABA A DONDE SÍ LLEGA EL ROBOT
    datos_yaml["articulaciones"][1]["servo"]["max"] = 175
    recortado = robot_desde_dict(datos_yaml)
    hombro = rangos_servo(recortado, 10)[1]
    assert (hombro[0], hombro[-1]) == (20, 175)
    assert np.diff(hombro).max() <= 10
    en_el_limite = cinematica_directa(recortado, servo_a_dh(recortado, (0, 175, 180)))
    assert np.linalg.norm(barrer(recortado, paso=10) - en_el_limite, axis=1).min() == pytest.approx(0, abs=1e-6)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CRITERIOS DEL REFACTOR A + B %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_c2_medida_del_yaml_llega_a_todo(datos_yaml):
    # SE ALARGA EL ANTEBRAZO EN EL YAML: CINEMATICA DIRECTA, INVERSA Y VOLUMEN LO USAN SIN TOCAR CODIGO
    datos_yaml["articulaciones"][2]["dh"]["a"] = 900
    datos_yaml["home_mm"] = [0, 0, 1200]  # CON EL ANTEBRAZO MÁS LARGO, LA PUNTA YA NO ALCANZA EL PUNTO DE GIRO A 1065 (P12)
    largo = robot_desde_dict(datos_yaml)
    estirado = cinematica_directa(largo, (0, 0, 0))
    assert estirado == pytest.approx((2020, 0, 350))  # 1000 + 900 + LOS 120 mm DE LA HERRAMIENTA
    assert alcanzable(largo, (1850, 10, 400))
    assert not alcanzable(cargar_robot(), (1850, 10, 400))
    assert np.max(np.linalg.norm(barrer(largo, paso=30) - (0, 0, 350), axis=1)) == pytest.approx(2020, rel=1e-6)


def test_c3_mapeo_servos_igual_al_original(robot):
    # LAS FORMULAS QUE ANTES ESTABAN ESCRITAS EN EL CODIGO: s1 = t1, s2 = 180 - t2, s3 = t3 + 180
    def original(t1, t2, t3):
        return ((t1 + 90) % 360) - 90, 180 - t2, (t3 + 180) % 360
    rng = np.random.default_rng(3)
    for _ in range(2000):
        q = (rng.uniform(-10, 190), rng.uniform(-10, 170), rng.uniform(-190, 10))
        nuevo, viejo = dh_a_servo(robot, q), original(*q)
        if servos_en_rango(robot, viejo):
            # IGUALES, SALVO EL RECORTE AL BORDE (≤ TOLERANCIA_SERVO) DE LO QUE SE SALÍA POR RUIDO
            assert np.allclose(nuevo, viejo, atol=TOLERANCIA_SERVO + 1e-9)


def test_c4_otro_robot_se_carga_y_tiene_cinematica(robot_4gdl):
    assert robot_4gdl.gdl == 4
    # TODO ESTIRADO HACIA ADELANTE: x = 300 + 250 + 80, z = d1
    estirado = cinematica_directa(robot_4gdl, (0, 0, 0, 0))
    assert estirado == pytest.approx((630, 0, 200))
    assert len(matrices(robot_4gdl, (10, 20, 30, 40))) == 5
    assert total_puntos(robot_4gdl, 45) == 5 ** 4
    # SIN SOLVER DECLARADO: LO DICE, NO TRUENA A MEDIAS
    with pytest.raises(SinSolver):
        angulos_servo(robot_4gdl, (300, 0, 300))


@pytest.mark.parametrize("romper, mensaje", [
    (lambda d: d["articulaciones"][1]["servo"].update({"min": 200}), "articulaciones[1].servo"),
    (lambda d: d["articulaciones"][0]["dh"].pop("d"), "Falta el campo 'd' en articulaciones[0].dh"),
    (lambda d: d["articulaciones"][2]["servo"].update({"signo": 2}), "'signo'"),
    (lambda d: d.update({"home_mm": [0, 0]}), "home_mm"),
    (lambda d: d["camara"].update({"montaje": "techo"}), "camara.montaje"),
    (lambda d: d["camara"]["matriz_montaje"][0].__setitem__(0, 2), "rotación válida"),
    (lambda d: d.update({"cinematica_inversa": "magia"}), "cinematica_inversa"),
    (lambda d: d["articulaciones"][1]["dh"].update({"alpha": 90}), "antropomorfico_3r"),
    (lambda d: d["articulaciones"][0]["dh"].update({"a": "mucho"}), "debe ser un número"),
    (lambda d: d["movimiento"].update({"al_cambiar_postura": "brincar"}), "movimiento.al_cambiar_postura"),
    (lambda d: d["movimiento"].update({"salto_max_por_paso": 0}), "mayores que 0"),
])
def test_c5_yaml_invalido_dice_que_esta_mal(datos_yaml, romper, mensaje):
    datos = copy.deepcopy(datos_yaml)
    romper(datos)
    with pytest.raises(ErrorConfiguracion, match=mensaje.replace("[", r"\[").replace("]", r"\]").replace(".", r"\.").replace("(", r"\(")):
        robot_desde_dict(datos)


def test_c5_archivo_que_no_existe(tmp_path):
    with pytest.raises(ErrorConfiguracion, match="No existe"):
        cargar_robot(str(tmp_path / "nada.yaml"))


def test_c7_robot_no_importa_qt_ni_pyvista():
    # SE CORRE EN UN PROCESO APARTE PARA QUE NO CUENTE LO QUE OTRAS PRUEBAS YA IMPORTARON
    codigo = ("import sys; import cosechador.robot.cinematica_inversa, cosechador.robot.espacio_trabajo; "
              "malos = [m for m in ('PyQt5', 'pyvista', 'vtk', 'cv2') if m in sys.modules]; "
              "print(malos); sys.exit(1 if malos else 0)")
    salida = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True,
                            cwd=__file__.rsplit("/tests/", 1)[0])
    assert salida.returncode == 0, salida.stdout + salida.stderr


def test_movimiento_por_omision_si_el_yaml_no_lo_trae(robot_4gdl):
    assert robot_4gdl.movimiento.salto_max_por_paso == 15
    assert robot_4gdl.movimiento.al_cambiar_postura == "despacio"


def test_home_redondeado_de_un_archivo_sigue_sobre_el_eje(robot):
    # 18-NOV: EL HOME DE UN PROGRAMA GUARDADO (SERVOS CON 4 DECIMALES) QUEDA A 0.0005 mm DEL EJE; CON
    # TOLERANCIAS DE 1e-6 EL JOG HACIA +X FALLABA A LOS 350 mm EN LUGAR DE A LOS 1545 mm
    # (24-NOV: CON LA HERRAMIENTA DE 120 mm, EL PUNTO SOBRE EL EJE ES (90, 35.9155, 44.9249) Y LA PUNTA LLEGA HASTA LOS 1675 mm)
    from cosechador.movimiento.trayectorias import tramo_continuo
    redondeado = tuple(round(s, 4) for s in angulos_servo(robot, robot.punto_de_giro))
    assert redondeado == (90.0, 35.9155, 44.9249)
    plan, motivo = tramo_continuo(robot, redondeado, (2050, 0, 1065), "despacio", pasos=411)
    assert "(1675, 0, 1065)" in motivo or "(1674, 0, 1065)" in motivo
    assert servos_en_rango(robot, dh_a_servo(robot, (-6e-5, 90, -90)))
    assert dh_a_servo(robot, (-6e-5, 90, -90))[0] == 0.0  # SE RECORTA AL BORDE: A LA PLACA LLEGA EN RANGO
