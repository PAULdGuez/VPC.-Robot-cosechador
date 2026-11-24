"""VER VOLUMEN: el volumen de trabajo dibujado como superficie translúcida (criterios V1 a V4).

El contorno del corte se calcula en cosechador/robot/espacio_trabajo.py y se gira en cosechador/ui/escena3d.py.
Aquí lo DIBUJADO se compara con lo que el robot alcanza de verdad, medido por otro camino que el que arma el
contorno: cinemática directa con otros ángulos (otro paso de barrido y ángulos al azar) y cinemática inversa.
"""
import dataclasses
import itertools
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import pyvista as pv
import yaml
from PyQt5 import QtCore, QtWidgets

from cosechador import registro as reg
from cosechador.robot import robot_desde_dict
from cosechador.robot.cinematica import cinematica_directa, matrices, servo_a_dh
from cosechador.robot.cinematica_inversa import alcanzable
from cosechador.robot.config import ROBOT_POR_OMISION
from cosechador.robot.espacio_trabajo import barrer, contorno_del_corte, giro_de_la_base, rangos_servo
from cosechador.ui.escena3d import Escena

MARGEN_MM = 15          # V2: LO QUE EL ROBOT ALCANZA QUEDA DENTRO DE LA SUPERFICIE O, COMO MUCHO, A ESTO
# CONTROL INVERSO: LO MÁS HONDO QUE UN PUNTO INALCANZABLE PUEDE QUEDAR DENTRO DE LA SUPERFICIE. MEDIDO EL 24-NOV
# (41 PUNTOS POR LADO DEL CONTORNO, ENTRANDO DE 0.1 EN 0.1 mm): 2.9 mm, DONDE UN LADO LARGO CRUZA UNA ENTRADA
PROMETE_DE_MAS_MM = 5
PASO_DE_CONTROL = 5     # EL CONTORNO SE ARMA CADA 2°; LA MUESTRA QUE LO JUZGA, CADA 5° Y CON ÁNGULOS AL AZAR


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INSTRUMENTOS DE MEDICIÓN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def dentro(segmentos, puntos):
    """Qué puntos (n, 2) encierra un contorno dado por sus segmentos (m, 2, 2). Regla par-impar."""
    x, y = puntos[:, :1], puntos[:, 1:]
    (x1, y1), (x2, y2) = segmentos[:, 0].T, segmentos[:, 1].T
    cruza = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_del_cruce = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return np.sum(cruza & (x < x_del_cruce), axis=1) % 2 == 1


def distancia(segmentos, puntos):
    """Distancia (mm) de cada punto al segmento más cercano del contorno."""
    a, ab = segmentos[:, 0], segmentos[:, 1] - segmentos[:, 0]
    t = np.clip(((puntos[:, None] - a) * ab).sum(axis=2) / np.maximum((ab * ab).sum(axis=1), 1e-30), 0, 1)
    return np.linalg.norm(puntos[:, None] - (a + t[..., None] * ab), axis=2).min(axis=1)


def fuera_por(segmentos, puntos):
    """Cuántos mm se sale cada punto del contorno (0 si queda dentro)."""
    if len(segmentos) == 0:
        return np.full(len(puntos), np.inf)
    return np.where(dentro(segmentos, puntos), 0.0, distancia(segmentos, puntos))


def dentro_por(segmentos, puntos):
    """Cuántos mm se mete cada punto en el contorno (0 si queda fuera)."""
    if len(segmentos) == 0:
        return np.zeros(len(puntos))
    return np.where(dentro(segmentos, puntos), distancia(segmentos, puntos), 0.0)


def lados(lazos):
    """Los lazos de un contorno, como segmentos (m, 2, 2)."""
    return np.concatenate([np.stack([lazo, np.roll(lazo, -1, axis=0)], axis=1) for lazo in lazos])


def frente(robot, servo_base):
    """Hacia dónde mira la base con ese ángulo de servo: el eje X de su marco (horizontal, unitario)."""
    q = servo_a_dh(robot, (servo_base, *(j.servo.centro for j in robot.articulaciones[1:])))
    return matrices(robot, q)[1][:3, 0]


def en_el_corte(robot, servo_base, puntos):
    """Puntos (x, y, z) -> (alcance sobre el frente de la base, altura): como se ven en el corte."""
    puntos = np.asarray(puntos, dtype=float)
    return np.column_stack([puntos @ frente(robot, servo_base), puntos[:, 2]])


def tajada(superficie, robot, servo_base):
    """La superficie DIBUJADA, cortada por el plano vertical de la base en ese ángulo: segmentos (m, 2, 2)."""
    hacia = frente(robot, servo_base)
    corte = superficie.slice(normal=np.cross((0, 0, 1), hacia), origin=(0, 0, 0))
    if corte.n_cells == 0:
        return np.empty((0, 2, 2))
    extremos = corte.points[corte.lines.reshape(-1, 3)[:, 1:]]
    return np.stack([extremos @ hacia, extremos[..., 2]], axis=-1)


def muestra_independiente(robot, servo_base, rng, paso=PASO_DE_CONTROL, al_azar=1500):
    """Puntos (x, y, z) que el robot alcanza con la base en ese ángulo, por otro camino que el del contorno:
    otro paso de barrido (con los dos límites de cada servo) y además ángulos al azar dentro de los límites."""
    resto = robot.articulaciones[1:]
    servos = list(itertools.product(*rangos_servo(robot, paso)[1:]))
    servos += [tuple(rng.uniform(j.servo.minimo, j.servo.maximo) for j in resto) for _ in range(al_azar)]
    return np.array([cinematica_directa(robot, servo_a_dh(robot, (servo_base, *s))) for s in servos])


def bases_al_azar(robot, rng, cuantas=3):
    """Ángulos de servo de la base al azar: caen ENTRE los gajos de la superficie, no sobre sus aristas."""
    base = robot.articulaciones[0].servo
    return rng.uniform(base.minimo, base.maximo, cuantas)


def huella(escena, actores):
    """Cuántos pixeles de la imagen cambian al esconder esos actores: lo que de ellos SE VE."""
    def imagen():
        escena.plotter.render()
        return escena.plotter.screenshot(return_img=True).astype(int)
    con = imagen()
    for actor in actores:
        actor.SetVisibility(False)
    sin = imagen()
    for actor in actores:
        actor.SetVisibility(True)
    return int((np.abs(con - sin).sum(axis=2) > 30).sum())


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ROBOTS Y VENTANAS DE PRUEBA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@pytest.fixture
def datos_yaml():
    with open(ROBOT_POR_OMISION, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def abrir_ventana(tmp_path_factory):
    """Abre la ventana de la app con el robot que se le pase (QT_QPA_PLATFORM=offscreen: nada a la vista)."""
    from cosechador.ui.ventana import VentanaPrincipal
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    abiertas = []

    def abrir(robot):
        carpeta = tmp_path_factory.mktemp("ventana")
        (carpeta / "programas").mkdir()
        ajustes = QtCore.QSettings(str(carpeta / "ajustes.ini"), QtCore.QSettings.IniFormat)
        abiertas.append(VentanaPrincipal(robot, ajustes=ajustes, carpeta_programas=str(carpeta / "programas"),
                                         registro=reg.Registro(carpeta=None)))
        return abiertas[-1]

    yield abrir
    for w in abiertas:
        w.close()
    app.processEvents()


@pytest.fixture(scope="module")
def con_volumen(robot, abrir_ventana):
    """La ventana del cosechador después de presionar VER VOLUMEN una vez."""
    w = abrir_ventana(robot)
    w.VER_VOLUMEN.click()
    return w


def superficie_dibujada(w):
    """La malla de la superficie del volumen que quedó en la escena de la ventana."""
    return w.escena.actores_volumen[0].mapper.dataset


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% V1: SUPERFICIE TRANSLÚCIDA CON CONTORNO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_v1_superficie_translucida_con_contorno_y_se_ve_a_traves(robot):
    escena = Escena(pv.Plotter(off_screen=True, window_size=(640, 480)), robot)
    # EL ROBOT EN UNA POSE CUALQUIERA, CON ALGO DE TRAYECTORIA: LO YA RECORRIDO Y LO QUE FALTA
    viene_de, pose, va_a = np.array([(90, 46, 46), (60, 110, 100), (25, 135, 60)], dtype=float)
    punta = lambda servos: cinematica_directa(robot, servo_a_dh(robot, servos))
    escena.dibujar(servo_a_dh(robot, pose))
    escena.reiniciar_recorrido(punta(viene_de))
    for k in np.linspace(0, 1, 30)[1:]:
        escena.agregar_al_recorrido(punta(viene_de + (pose - viene_de) * k))
    escena.mostrar_plan([punta(pose + (va_a - pose) * k) for k in np.linspace(0, 1, 30)])
    trayectoria = [escena._actor_plan, escena._actor_recorrido]

    escena.mostrar_superficie(contorno_del_corte(robot), giro_de_la_base(robot))
    escena.vista("iso")
    escena.encuadrar(zoom=1.0)  # COMO QUEDA LA CÁMARA DESPUÉS DE VER VOLUMEN

    superficie, borde = escena.actores_volumen
    assert superficie.mapper.dataset.area > 0 and borde.mapper.dataset.n_lines > 0
    assert superficie.GetVisibility() and borde.GetVisibility()
    assert all(actor.GetVisibility() for actor in escena.actores + trayectoria)

    # A TRAVÉS DE LA SUPERFICIE SE SIGUE VIENDO (CASI) TODO LO QUE SE VEÍA SIN ELLA, DESDE LA MISMA CÁMARA
    se_ve = huella(escena, escena.actores), huella(escena, trayectoria)
    for actor in escena.actores_volumen:
        actor.SetVisibility(False)
    se_veia = huella(escena, escena.actores), huella(escena, trayectoria)
    assert min(se_veia) > 100
    assert se_ve[0] >= 0.8 * se_veia[0], f"robot: se ven {se_ve[0]} pixeles de {se_veia[0]}"
    assert se_ve[1] >= 0.8 * se_veia[1], f"trayectoria: se ven {se_ve[1]} pixeles de {se_veia[1]}"
    assert 0 < superficie.prop.opacity < 1


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% V2: LO ALCANZABLE QUEDA DENTRO (Y NO SE PROMETE DE MÁS) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_v2_todo_lo_que_el_robot_alcanza_queda_dentro_de_la_superficie(robot, con_volumen):
    rng = np.random.default_rng(24)
    # EL CORTE, CON LA BASE EN UNO DE SUS LÍMITES...
    en_el_limite = robot.articulaciones[0].servo.maximo
    alcanzado = en_el_corte(robot, en_el_limite, muestra_independiente(robot, en_el_limite, rng))
    assert fuera_por(lados(contorno_del_corte(robot)), alcanzado).max() <= MARGEN_MM
    # ...Y LA SUPERFICIE QUE QUEDÓ DIBUJADA AL PRESIONAR EL BOTÓN, CON LA BASE DONDE SEA
    for servo_base in bases_al_azar(robot, rng):
        alcanzado = en_el_corte(robot, servo_base, muestra_independiente(robot, servo_base, rng))
        assert fuera_por(tajada(superficie_dibujada(con_volumen), robot, servo_base), alcanzado).max() <= MARGEN_MM


def test_v2_la_superficie_no_promete_de_mas(robot):
    lazos = contorno_del_corte(robot)
    contorno = lados(lazos)
    # LA CINEMÁTICA INVERSA DICE SI UN PUNTO DEL CORTE SE ALCANZA (CON LA BASE AL CENTRO, PARA QUE NO LO ALCANCE GIRANDO)
    centro = robot.articulaciones[0].servo.centro
    se_alcanza = lambda puntos: np.array([alcanzable(robot, alcance * frente(robot, centro) + (0, 0, altura))
                                          for alcance, altura in puntos])
    assert se_alcanza(np.concatenate(lazos)).all()  # CADA VÉRTICE DEL CONTORNO ES UN PUNTO AL QUE EL ROBOT LLEGA

    # PUNTOS AL AZAR PEGADOS AL CONTORNO (DONDE SE PUEDE PROMETER DE MÁS) Y EN TODA LA CAJA (POR SI TAPA UN HUECO)
    rng = np.random.default_rng(11)
    lado, a_lo_largo = contorno[rng.integers(len(contorno), size=6000)], rng.random((6000, 1))
    giro, radio = rng.uniform(0, 2 * np.pi, 6000), 4 * PROMETE_DE_MAS_MM * np.sqrt(rng.random(6000))
    pegados = lado[:, 0] + a_lo_largo * (lado[:, 1] - lado[:, 0]) + np.column_stack([radio * np.cos(giro), radio * np.sin(giro)])
    en_la_caja = rng.uniform(contorno.min(axis=(0, 1)), contorno.max(axis=(0, 1)), size=(4000, 2))
    candidatos = np.concatenate([pegados, en_la_caja])
    inalcanzables = candidatos[~se_alcanza(candidatos)]
    assert len(inalcanzables) > 1000  # LA MUESTRA SÍ TRAE PUNTOS A LOS QUE EL ROBOT NO LLEGA
    assert dentro_por(contorno, inalcanzables).max() <= PROMETE_DE_MAS_MM


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% V3: EL HUECO JUNTO A LA BASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_v3_se_ve_el_hueco_junto_a_la_base(robot, con_volumen):
    # EL EJE DEL HOMBRO (DONDE ACABA LA COLUMNA DE LA BASE): EL BRAZO NO SE PUEDE DOBLAR HASTA TOCARLO
    hombro = matrices(robot, servo_a_dh(robot, [j.servo.centro for j in robot.articulaciones]))[1][:3, 3]
    assert not alcanzable(robot, hombro)
    for servo_base in bases_al_azar(robot, np.random.default_rng(3)):
        contorno = tajada(superficie_dibujada(con_volumen), robot, servo_base)
        punto = en_el_corte(robot, servo_base, [hombro])
        assert fuera_por(contorno, punto)[0] > MARGEN_MM  # FUERA DE LA SUPERFICIE, Y NO POR EL MARGEN
        # Y ES UN HUECO, NO LA ORILLA: HAY SUPERFICIE DIBUJADA DELANTE Y DETRÁS, ARRIBA Y ABAJO DEL HOMBRO
        assert (contorno.min(axis=(0, 1)) < punto[0] - MARGEN_MM).all()
        assert (contorno.max(axis=(0, 1)) > punto[0] + MARGEN_MM).all()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% V4: SIGUE A LOS LÍMITES DEL ROBOT Y NO SE ENCIMA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_v4_si_cambian_los_limites_el_dibujo_cambia_solo(robot, datos_yaml, abrir_ventana):
    # LÍMITES NUEVOS EN EL YAML, NINGUNO MÚLTIPLO DEL PASO DEL CORTE: LA BASE GIRA MENOS, EL HOMBRO BAJA MENOS
    # Y EL CODO YA NO SE CIERRA DEL TODO
    base, hombro, codo = (articulacion["servo"] for articulacion in datos_yaml["articulaciones"])
    antes = base["max"]
    base["max"], hombro["max"], codo["min"] = 95, 133, 47
    recortado = robot_desde_dict(datos_yaml)
    w = abrir_ventana(recortado)
    w.VER_VOLUMEN.click()
    rng = np.random.default_rng(4)
    # LO QUE ALCANZA CON LOS LÍMITES NUEVOS QUEDA DENTRO DE LO DIBUJADO...
    for servo_base in bases_al_azar(recortado, rng, 2):
        alcanzado = en_el_corte(recortado, servo_base, muestra_independiente(recortado, servo_base, rng))
        assert fuera_por(tajada(superficie_dibujada(w), recortado, servo_base), alcanzado).max() <= MARGEN_MM
    # ...Y LO QUE SOLO ALCANZABA CON LOS DE ANTES YA NO ESTÁ DIBUJADO: NI HACIA DONDE LA BASE TODAVÍA GIRA, NI
    # HACIA DONDE YA NO (A MEDIO CAMINO ENTRE SU LÍMITE NUEVO Y EL DE ANTES)
    for servo_base in (*bases_al_azar(recortado, rng, 2), (base["max"] + antes) / 2):
        de_antes = muestra_independiente(robot, servo_base, rng)
        ya_no = en_el_corte(recortado, servo_base, [p for p in de_antes if not alcanzable(recortado, p)])
        assert len(ya_no) > 500
        assert dentro_por(tajada(superficie_dibujada(w), recortado, servo_base), ya_no).max() <= PROMETE_DE_MAS_MM


def test_v4_presionarlo_dos_veces_no_lo_encima(con_volumen):
    plotter = con_volumen.escena.plotter
    dibujado = len(plotter.renderer.actors), superficie_dibujada(con_volumen).n_points
    con_volumen.VER_VOLUMEN.click()
    assert (len(plotter.renderer.actors), superficie_dibujada(con_volumen).n_points) == dibujado
    assert len(con_volumen.escena.actores_volumen) == 2  # LA SUPERFICIE Y SU CONTORNO, UNA SOLA VEZ


def test_la_barra_de_progreso_termina_llena(con_volumen):
    assert con_volumen.PROCESO.value() == con_volumen.PROCESO.maximum() > 0


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OTROS ROBOTS: NO SE ROMPEN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_un_brazo_que_se_sale_del_plano_se_sigue_viendo_como_nube(datos_yaml, abrir_ventana):
    # EL HOMBRO, A UN LADO DEL EJE DE LA BASE: CON LA BASE QUIETA EL EFECTOR YA NO SE MUEVE EN UN PLANO QUE PASE POR
    # ESE EJE, Y GIRAR EL CORTE NO DARÍA SU VOLUMEN. (LÍMITES CORTOS PARA QUE LA NUBE SEA CHICA.)
    datos_yaml.pop("cinematica_inversa")  # EL SOLVER DEL COSECHADOR NO ADMITE ESE DESPLAZAMIENTO
    datos_yaml["articulaciones"][1]["dh"]["d"] = 120
    for articulacion, (minimo, maximo) in zip(datos_yaml["articulaciones"], [(0, 30), (20, 60), (0, 40)]):
        articulacion["servo"].update({"min": minimo, "max": maximo})
    de_lado = robot_desde_dict(datos_yaml)
    assert contorno_del_corte(de_lado) is None
    w = abrir_ventana(de_lado)
    w.VER_VOLUMEN.click()
    (nube,) = w.escena.actores_volumen
    assert np.allclose(nube.mapper.dataset.points, barrer(de_lado, paso=5), atol=0.01)
    assert w.PROCESO.value() == w.PROCESO.maximum() > 0


@pytest.mark.parametrize("articulaciones", [1, 2])
def test_un_brazo_cuyo_corte_no_encierra_area_no_tiene_superficie(robot, articulaciones):
    # CON LA BASE QUIETA, UNA SOLA ARTICULACIÓN ALCANZA UN PUNTO Y DOS ALCANZAN UN ARCO: NO HAY QUÉ CONTORNEAR
    corto = dataclasses.replace(robot, articulaciones=robot.articulaciones[:articulaciones], cinematica_inversa=None)
    assert contorno_del_corte(corto) is None


def test_c4_el_brazo_de_4_articulaciones_tambien_tiene_superficie(robot_4gdl):
    # A UN PASO MÁS GRUESO QUE EL DE LA APP (EL CORTE CRECE CON CADA ARTICULACIÓN); LA MUESTRA, A OTRO PASO
    lazos = contorno_del_corte(robot_4gdl, paso=10)
    servo_base = robot_4gdl.articulaciones[0].servo.minimo
    alcanzado = muestra_independiente(robot_4gdl, servo_base, np.random.default_rng(8), paso=13)
    assert fuera_por(lados(lazos), en_el_corte(robot_4gdl, servo_base, alcanzado)).max() <= MARGEN_MM
