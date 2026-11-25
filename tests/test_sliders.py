"""Sliders de la pestaña «Punta X Y Z» que enseñan hasta dónde llega la punta (criterios S1 a S5, aprobados el 24-nov-2026).

«Alcanzable» lo decide `alcanzable` del robot (cinematica_inversa.py); aquí se mide contra ÉL, punto por punto, y no
contra números escritos a mano: lo pintado en cada slider se compara con una barrida de 1 mm sobre todo el recorrido.

TOLERANCIA DEL MUESTREO: el tramo se busca cada 5 mm y cada frontera se afina hasta 0.25 mm (TOLERANCIA_TRAMOS_MM).
Dentro de lo pintado todo es alcanzable; a más de 0.25 mm de una frontera, fuera de lo pintado nada lo es. Lo único que
el muestreo no ve es un tramo alcanzable más angosto que 5 mm; la barrida de 1 mm de aquí lo vigila en cada pose.
"""
import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5 import QtWidgets

import auditoria_interfaz as au
from cosechador.movimiento.programa import Punto
from cosechador.movimiento.trayectorias import posicion
from cosechador.robot import robot_desde_dict
from cosechador.robot.cinematica_inversa import PASO_TRAMOS_MM, TOLERANCIA_TRAMOS_MM, alcanzable, angulos_servo
from cosechador.ui import tema
from cosechador.ui.ventana import COLOR_NIVEL, PESTANA_XYZ
from test_editor_puntos import abrir as abrir_editor  # noqa: F401 (NO ES UN FIXTURE: ABRE EL EDITOR CON UN PROGRAMA)
from test_editor_puntos import guardar_programa, home_s, llevar, poses  # noqa: F401 (home_s Y poses SON FIXTURES)
from test_grabacion_ventana import correr, eventos, ventana  # noqa: F401 (ventana ES EL FIXTURE)
from test_punta import abrir, con_largo, datos_yaml  # noqa: F401 (SON FIXTURES)

EJES = "XYZ"
RANGO_DEL_EJE = ((-2170, 2170), (-2170, 2170), (0, 2170))  # LO QUE RECORRE CADA SLIDER CON LA HERRAMIENTA DE 120 mm


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INSTRUMENTO: LO PINTADO CONTRA LA BARRIDA DE 1 mm %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def con_el_eje(punto, eje, valor):
    q = list(punto)
    q[eje] = valor
    return tuple(q)


def tramos_a_mano(robot, punto, eje, lo, hi):
    """Los tramos alcanzables de un eje sacados a fuerza bruta, de 1 mm en 1 mm (otro camino que el del producto)."""
    tramos, inicio, previo = [], None, None
    for v in range(lo, hi + 1):
        if alcanzable(robot, con_el_eje(punto, eje, v)):
            inicio = v if inicio is None else inicio
            previo = v
        elif inicio is not None:
            tramos.append((inicio, previo))
            inicio = None
    if inicio is not None:
        tramos.append((inicio, previo))
    return tramos


def cubierto(tramos, v):
    return any(a <= v <= b for a, b in tramos)


def cerca_de_una_frontera(tramos, v):
    return any(abs(v - borde) <= TOLERANCIA_TRAMOS_MM for tramo in tramos for borde in tramo)


def defectos_de_lo_pintado(w):
    """Todo lo que lo pintado en los tres sliders dice distinto de lo que el robot alcanza desde donde está. [] = bien."""
    robot, punto, hallazgos = w.robot, w.posicion, []
    for eje, slider in enumerate(w.sliders_xyz):
        pintado = slider.tramos
        if pintado is None:
            hallazgos.append(f"{EJES[eje]}: no pinta nada")
            continue
        # LA BARRIDA DE 1 mm: LO QUE SE ALCANZA EN UN TRAMO DE AL MENOS 5 mm (EL PASO DEL MUESTREO) DEBE ESTAR PINTADO. UN
        # TRAMO MÁS ANGOSTO (P. EJ. EL PUNTO SUELTO CON EL BRAZO ESTIRADO AL MÁXIMO) NO SE VE: ESTÁ DECLARADO
        a_mano = [t for t in tramos_a_mano(robot, punto, eje, slider.minimum(), slider.maximum()) if t[1] - t[0] >= PASO_TRAMOS_MM]
        for v in range(slider.minimum(), slider.maximum() + 1):
            llega = alcanzable(robot, con_el_eje(punto, eje, v))
            falta = llega and cubierto(a_mano, v) and not cubierto(pintado, v)   # SE ALCANZA Y NO ESTÁ PINTADO
            sobra = not llega and cubierto(pintado, v)                            # ESTÁ PINTADO Y NO SE ALCANZA
            if (falta or sobra) and not cerca_de_una_frontera(pintado, v):
                hallazgos.append(f"{EJES[eje]} = {v}: el robot {'sí' if llega else 'no'} llega y el slider dice lo contrario")
                break
        for a, b in pintado:  # JUSTO FUERA DE CADA FRONTERA (A MÁS DE LA TOLERANCIA) NO SE LLEGA
            for fuera, dentro in ((a - 1.01 * TOLERANCIA_TRAMOS_MM, a), (b + 1.01 * TOLERANCIA_TRAMOS_MM, b)):
                if slider.minimum() <= fuera <= slider.maximum() and alcanzable(robot, con_el_eje(punto, eje, fuera)):
                    hallazgos.append(f"{EJES[eje]}: a {fuera:.2f} el robot sí llega y está fuera del tramo ({a:.2f}, {b:.2f})")
                if not alcanzable(robot, con_el_eje(punto, eje, dentro)):
                    hallazgos.append(f"{EJES[eje]}: la frontera {dentro:.2f} no se alcanza y está dentro del tramo")
    return hallazgos


def colocar(w, servos):
    """Deja el robot en esos servos como lo hace la app al arrancar en HOME: poner los servos y sincronizar."""
    w.poner_servos(servos)
    w._sincronizar_sliders()


def tramos_de(w):
    return [None if s.tramos is None else [tuple(t) for t in s.tramos] for s in w.sliders_xyz]


def pose_con_dos_tramos(robot, eje=0):
    """Un punto alcanzable cuyo eje tiene DOS tramos (el hueco junto a la base). Se busca con `alcanzable`, 2 mm por paso,
    en una rejilla fija: siempre devuelve el mismo punto."""
    lo, hi = RANGO_DEL_EJE[eje]
    for z in range(0, 1800, 100):
        for y in range(-600, 601, 100):
            for x in range(-600, 601, 100):
                punto = (x, y, z)
                if not alcanzable(robot, punto):
                    continue
                dentro, previo = 0, False
                for v in range(lo, hi + 1, 2):
                    ok = alcanzable(robot, con_el_eje(punto, eje, v))
                    dentro += ok and not previo
                    previo = ok
                if dentro == 2:
                    return punto
    raise AssertionError("no hay pose con dos tramos en la rejilla: el robot cambió")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% S1: CADA SLIDER PINTA EL TRAMO ALCANZABLE; EL RESTO, EN ROJO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
POSES_S1 = ("home", "arriba", "corte", "ida", "vuelta", "la_de_paul", "dos_tramos")


@pytest.fixture
def pose(robot, poses):
    """Servos de cada pose de la prueba. La de Paul es la de la captura del 24-nov; la de dos tramos se busca."""
    return {**poses, "la_de_paul": angulos_servo(robot, (-697, 718, 1862)),
            "dos_tramos": angulos_servo(robot, pose_con_dos_tramos(robot))}


@pytest.mark.parametrize("nombre", POSES_S1)
def test_s1_lo_pintado_es_lo_que_la_punta_alcanza_desde_ahi(ventana, pose, nombre):
    w = ventana
    colocar(w, pose[nombre])
    assert defectos_de_lo_pintado(w) == []


def test_s1_la_pose_de_paul_da_los_tramos_medidos_en_la_nota(ventana, pose):
    """Control del instrumento: las cifras de la nota del 24-nov (X ±710, Y ±735, Z de 0 a 1870) salen del mismo código."""
    colocar(ventana, pose["la_de_paul"])
    for slider, (a, b) in zip(ventana.sliders_xyz, ((-710, 710), (-735, 735), (0, 1870))):
        assert len(slider.tramos) == 1
        assert slider.tramos[0] == pytest.approx((a, b), abs=5)  # LA NOTA MIDIÓ CADA 5 mm: ESTO AFINA LA FRONTERA


def test_s1_con_dos_tramos_en_un_eje_se_pintan_los_dos(robot, ventana, pose):
    w = ventana
    punto = pose_con_dos_tramos(robot)
    colocar(w, pose["dos_tramos"])
    dos = w.SLIDER_X.tramos
    assert len(dos) == 2 and dos[0][1] < dos[1][0]                 # DOS TRAMOS Y UN HUECO ENTRE ELLOS
    a_mano = tramos_a_mano(robot, punto, 0, -2170, 2170)
    assert len(a_mano) == 2
    assert dos == [pytest.approx(t, abs=1 + TOLERANCIA_TRAMOS_MM) for t in a_mano]  # a_mano VA DE 1 mm EN 1 mm
    hueco = (dos[0][1] + dos[1][0]) / 2
    assert not alcanzable(robot, con_el_eje(w.posicion, 0, hueco))  # EL HUECO ES ROJO: AHÍ NO SE LLEGA


def test_s1_el_rojo_arranca_donde_se_acaba_el_tramo_en_la_perilla(ventana, pose):
    """Lo que se VE: con la perilla en el límite de un tramo, el rojo empieza pegado a ella y lo alcanzable no se pinta."""
    w = ventana
    w.resize(1366, 768)
    w.show()
    w.PESTANAS_MOVIMIENTO.setCurrentIndex(PESTANA_XYZ)
    colocar(w, pose["la_de_paul"])
    QtWidgets.QApplication.processEvents()
    rojo = np.array(tuple(int(COLOR_NIVEL["error"][i:i + 2], 16) for i in (1, 3, 5)))  # EL ROJO DE LOS ERRORES DEL REGISTRO
    manija = np.array(tuple(int(tema.TOKENS["manija"][i:i + 2], 16) for i in (1, 3, 5)))
    slider = w.SLIDER_X
    for valor, lado in ((int(math.floor(slider.tramos[0][1])), +1), (int(math.ceil(slider.tramos[0][0])), -1)):
        w.sliders_xyz[0].blockSignals(True)
        slider.setValue(valor)
        slider.blockSignals(False)
        QtWidgets.QApplication.processEvents()
        pixeles, _ = au._pixeles(w, slider)
        fila = pixeles[pixeles.shape[0] // 2].astype(int)
        es_manija = np.abs(fila - manija).sum(axis=1) < 8   # LA MANIJA Y EL ROJO SE PARECEN (24 DE DIFERENCIA): UMBRAL ESTRICTO
        assert es_manija.any()
        centro = int(round(np.flatnonzero(es_manija).mean()))
        pegado_a_la_perilla, del_lado_del_tramo = centro + lado * 12, centro - lado * 12
        assert np.abs(fila[pegado_a_la_perilla] - rojo).sum() < 8, "tras el límite debe estar el rojo"
        assert np.abs(fila[del_lado_del_tramo] - rojo).sum() > 150, "en el tramo no debe haber rojo"
    w.hide()


def test_s1_sin_solver_los_sliders_no_pintan_ni_frenan(robot_4gdl, abrir):
    """Un robot sin cinemática inversa no puede decir qué alcanza: los sliders quedan como antes (nada en rojo)."""
    w = abrir(robot_4gdl)
    assert all(s.tramos is None for s in w.sliders_xyz)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% S2: AL MOVER EL ROBOT, LOS TRAMOS SE ACTUALIZAN SOLOS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _por_slider_de_servos(w, poses):
    w.SLIDER_1.setValue(w.SLIDER_1.value() + 40)
    correr(w)


def _por_calcular(w, poses):
    for campo, valor in zip((w.TE_X, w.TE_Y, w.TE_Z), (356, 711, 1014)):
        campo.setText(str(valor))
    w.CALCULAR.click()
    correr(w)


def _por_cosechar(w, poses):
    w.COSECHAR.click()
    correr(w)


def _por_reproducir(w, poses):
    guardar_programa(w, "ir.yaml", Punto(poses["arriba"]), Punto(poses["corte"], "lineal", 40))
    w.SELECTOR_PROGRAMA.setCurrentIndex(w.SELECTOR_PROGRAMA.findData("ir.yaml"))
    w.BTN_REPRODUCIR.click()
    correr(w)


def _por_paso_a_paso(w, poses):
    abrir_editor(w, "pasos.yaml", Punto(poses["arriba"]), Punto(poses["corte"], "lineal", 40))
    w.BTN_PASO_ADELANTE.click()
    correr(w)


def _por_home(w, poses):
    w._ir_a_home()


MEDIOS_S2 = {"sliders de servos": _por_slider_de_servos, "CALCULAR": _por_calcular, "COSECHAR": _por_cosechar,
             "reproducir": _por_reproducir, "paso a paso": _por_paso_a_paso, "HOME": _por_home}


@pytest.mark.parametrize("medio", MEDIOS_S2)
def test_s2_al_mover_el_robot_los_tramos_se_actualizan_solos(ventana, poses, medio):
    w = ventana
    if medio in ("CALCULAR", "COSECHAR", "HOME"):  # ESTOS TERMINAN EN HOME: SE SALE DE OTRA POSE PARA QUE HAYA CAMBIO
        llevar(w, poses["ida"])
    antes = tramos_de(w)
    assert defectos_de_lo_pintado(w) == []
    MEDIOS_S2[medio](w, poses)
    assert w.posicion != pytest.approx(posicion(w.robot, poses["ida"]), abs=1) or medio == "sliders de servos"
    assert tramos_de(w) != antes, "el robot se movió y los tramos se quedaron como estaban"
    assert defectos_de_lo_pintado(w) == []


def test_s2_arrastrar_un_eje_cambia_los_tramos_de_los_otros_dos(ventana):
    """Arrastrar X cambia lo que alcanzan Y y Z: al llegar la punta, los tramos de Y y Z son los de la pose nueva."""
    w = ventana
    antes = tramos_de(w)
    w.SLIDER_X.setValue(500)
    correr(w)
    despues = tramos_de(w)
    assert np.allclose(despues[0], antes[0], atol=2 * TOLERANCIA_TRAMOS_MM)  # LA RECTA DE X ES LA MISMA: SU TRAMO NO CAMBIA
    assert despues[1] != antes[1] and defectos_de_lo_pintado(w) == []


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% S3: EL SLIDER SE DETIENE EN LA FRONTERA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def arrastrar_hasta(w, slider, destino, paso=7, alcanza_cada=10):
    """Un arrastre: el valor del slider cambia de poco en poco y, cada `alcanza_cada` cambios, el robot alcanza lo que
    le pidieron (en la app lo hace el temporizador entre un evento del ratón y otro). Devuelve lo que fue valiendo el slider."""
    vistos = []
    valor = slider.value()
    slider.sliderPressed.emit()  # LA PERSONA AGARRA LA PERILLA
    while valor != destino:
        valor += max(-paso, min(paso, destino - valor))
        slider.setValue(valor)
        vistos.append(slider.value())
        if len(vistos) % alcanza_cada == 0:
            correr(w)
    slider.sliderReleased.emit()  # Y LA SUELTA
    correr(w)
    return vistos


@pytest.mark.parametrize("eje", (0, 1, 2))
@pytest.mark.parametrize("hacia", ("el máximo", "el mínimo"))
def test_s3_el_slider_se_detiene_en_la_frontera_y_la_punta_llega_a_ella(ventana, pose, eje, hacia):
    w = ventana
    colocar(w, pose["la_de_paul"])
    slider = w.sliders_xyz[eje]
    (a, b), = slider.tramos
    destino = slider.maximum() if hacia == "el máximo" else slider.minimum()
    vistos = arrastrar_hasta(w, slider, destino)
    frontera = math.floor(b) if hacia == "el máximo" else math.ceil(a)
    assert all(cubierto([(a, b)], v) for v in vistos), "el slider entró al rojo"
    assert slider.value() == frontera
    assert w.posicion[eje] == pytest.approx(frontera, abs=1)
    assert defectos_de_lo_pintado(w) == []


def test_s3_con_dos_tramos_el_slider_no_salta_el_hueco(robot, ventana, pose):
    w = ventana
    colocar(w, pose["dos_tramos"])
    (a1, b1), (a2, b2) = w.SLIDER_X.tramos
    desde = w.posicion[0]
    assert a1 <= desde <= b1 or a2 <= desde <= b2
    primero = a1 <= desde <= b1
    propio, otro = ((a1, b1), (a2, b2)) if primero else ((a2, b2), (a1, b1))
    destino = int(math.ceil(otro[0])) if primero else int(math.floor(otro[1]))  # DENTRO DEL OTRO TRAMO, ALCANZABLE
    assert alcanzable(robot, con_el_eje(w.posicion, 0, destino))
    vistos = arrastrar_hasta(w, w.SLIDER_X, destino)
    assert all(cubierto([propio], v) for v in vistos)
    assert w.SLIDER_X.value() == (math.floor(propio[1]) if primero else math.ceil(propio[0]))
    assert cubierto([propio], w.posicion[0] + 0.0) or w.posicion[0] == pytest.approx(w.SLIDER_X.value(), abs=1)


def test_s3_un_clic_al_extremo_tambien_se_detiene_en_la_frontera(ventana, pose):
    w = ventana
    colocar(w, pose["la_de_paul"])
    w.SLIDER_Y.setValue(w.SLIDER_Y.maximum())  # DE UN SOLO SALTO, COMO UN CLIC EN LA PISTA
    correr(w)
    assert w.SLIDER_Y.value() == math.floor(w.SLIDER_Y.tramos[0][1])
    assert alcanzable(w.robot, w.posicion)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% S4: UN SOLO AVISO AL TOPAR, NO UNA LLUVIA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def avisos_del_jog(w):
    return [e for e in eventos(w, "aviso") if e.origen == "Jog"]


def topes(w):
    """Los avisos de «la punta no llega más allá…» (no los del planeador, que van con otro mensaje)."""
    return [e for e in avisos_del_jog(w) if "tope_mm" in e.a_dict()["datos"]]


@pytest.mark.parametrize("eje", (0, 1, 2))
def test_s4_arrastrar_de_un_extremo_al_otro_deja_un_solo_aviso_por_cada_tope(ventana, pose, eje):
    w = ventana
    colocar(w, pose["la_de_paul"])
    slider = w.sliders_xyz[eje]
    arrastrar_hasta(w, slider, slider.maximum(), paso=5)   # UN CAMBIO CADA 5 mm: MUCHOS MÁS QUE LOS QUE TOPAN
    assert len(topes(w)) == 1
    arrastrar_hasta(w, slider, slider.minimum(), paso=5)   # Y DE REGRESO HASTA EL OTRO EXTREMO
    assert len(topes(w)) == 2 - (eje == 2)                 # Z NO TIENE ROJO ABAJO: SU TRAMO ARRANCA EN EL MÍNIMO DEL SLIDER


def test_s4_el_aviso_dice_el_eje_y_hasta_donde_se_puede(ventana, pose):
    w = ventana
    colocar(w, pose["la_de_paul"])
    arrastrar_hasta(w, w.SLIDER_X, w.SLIDER_X.maximum())
    aviso = avisos_del_jog(w)[-1]
    assert "X" in aviso.mensaje and str(math.floor(w.SLIDER_X.tramos[0][1])) in aviso.mensaje
    assert aviso.datos["eje"] == "X"


def test_s4_el_aviso_del_planeador_tampoco_es_una_lluvia(ventana, robot):
    """Un punto alcanzable al que el camino recto no se puede en modo «negar»: se avisa por qué, una vez por arrastre."""
    w = ventana
    w.SELECTOR_POSTURA.setCurrentIndex(w.SELECTOR_POSTURA.findData("negar"))
    colocar(w, angulos_servo(robot, (-600, 0, 1100)))
    n0 = len(avisos_del_jog(w))
    w.SLIDER_X.sliderPressed.emit()
    for x in range(-590, 601, 1):    # SIN CORRER EL MOVIMIENTO: LA PUNTA NO SE MUEVE Y CADA VALOR NUEVO ES UN PLAN NUEVO
        w.SLIDER_X.setValue(x)
    w.SLIDER_X.sliderReleased.emit()
    nuevos = avisos_del_jog(w)[n0:]
    assert len(nuevos) == 1 and "postura" in nuevos[0].mensaje
    w.SLIDER_X.sliderPressed.emit()      # OTRO ARRASTRE, OTRO AVISO
    w.SLIDER_X.setValue(550)
    w.SLIDER_X.setValue(551)
    w.SLIDER_X.sliderReleased.emit()
    assert len(avisos_del_jog(w)) == n0 + 2


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% S5: NADA DEL ROBOT VA ESCRITO EN EL CÓDIGO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@pytest.mark.parametrize("largo", (0, 120, 300))
def test_s5_otro_largo_de_herramienta_cambia_los_tramos_solos(robot, con_largo, abrir, largo):
    w = abrir(con_largo(largo))
    colocar(w, angulos_servo(w.robot, (-300, 400, 900)))
    assert defectos_de_lo_pintado(w) == []
    assert [(s.minimum(), s.maximum()) for s in w.sliders_xyz] == [
        (-(2050 + largo), 2050 + largo), (-(2050 + largo), 2050 + largo), (0, 2050 + largo)]


def test_s5_con_otro_largo_el_mismo_punto_pinta_otros_tramos(con_largo, abrir):
    punto = (-300, 400, 900)
    chico, grande = abrir(con_largo(0)), abrir(con_largo(300))
    for w in (chico, grande):
        colocar(w, angulos_servo(w.robot, punto))
    assert tramos_de(chico) != tramos_de(grande)


def test_s5_otros_rangos_de_servo_cambian_los_tramos_solos(robot, datos_yaml, abrir):
    estrecho = {**datos_yaml}
    estrecho["articulaciones"] = [dict(j) for j in datos_yaml["articulaciones"]]
    hombro = dict(estrecho["articulaciones"][1])
    hombro["servo"] = {**hombro["servo"], "min": 60, "max": 120}
    estrecho["articulaciones"][1] = hombro
    chico, normal = abrir(robot_desde_dict(estrecho, carpeta_modelos=robot.carpeta_modelos)), abrir(robot)
    colocar(chico, angulos_servo(chico.robot, (200, 500, 1100)))
    colocar(normal, angulos_servo(normal.robot, (200, 500, 1100)))
    assert defectos_de_lo_pintado(chico) == [] and defectos_de_lo_pintado(normal) == []
    assert tramos_de(chico) != tramos_de(normal)


def test_s5_un_robot_de_otra_medida_pinta_lo_suyo(robot, datos_yaml, abrir):
    """Otro archivo de robot: eslabones más cortos. Sin ningún número del cosechador en el código, los tramos son los suyos."""
    corto = {**datos_yaml}
    corto["articulaciones"] = [dict(j) for j in datos_yaml["articulaciones"]]
    for j in corto["articulaciones"]:
        j["dh"] = {**j["dh"], "a": j["dh"]["a"] / 2, "d": j["dh"]["d"] / 2}
    corto["home_mm"] = [0, 0, 400]
    w = abrir(robot_desde_dict(corto, carpeta_modelos=robot.carpeta_modelos))
    assert w.SLIDER_X.maximum() < 1500
    colocar(w, angulos_servo(w.robot, (100, 200, 500)))
    assert defectos_de_lo_pintado(w) == []
    assert max(b for _, b in w.SLIDER_X.tramos) < 1100
