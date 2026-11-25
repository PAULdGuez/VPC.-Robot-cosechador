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
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtTest import QTest

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
ROJO = np.array(tuple(int(COLOR_NIVEL["error"][i:i + 2], 16) for i in (1, 3, 5)))    # EL ROJO DE LOS ERRORES DEL REGISTRO
MANIJA = np.array(tuple(int(tema.TOKENS["manija"][i:i + 2], 16) for i in (1, 3, 5)))


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


def a_la_vista(w):
    """La pestaña de la punta, pintada de verdad (sin pantalla: nada queda a la vista de nadie)."""
    w.resize(1366, 768)
    w.show()
    w.PESTANAS_MOVIMIENTO.setCurrentIndex(PESTANA_XYZ)
    QtWidgets.QApplication.processEvents()


def junto_a_la_perilla(w, slider, valor):
    """Lo que se VE con la perilla en `valor` (sin mover el robot): (hay rojo a su izquierda, hay rojo a su derecha),
    mirando 12 pixeles a cada lado de su centro. A la perilla la coloca Qt, no la cuenta del producto: es otra regla."""
    slider.blockSignals(True)
    slider.setValue(valor)
    slider.blockSignals(False)
    QtWidgets.QApplication.processEvents()
    pixeles, _ = au._pixeles(w, slider)
    fila = pixeles[pixeles.shape[0] // 2].astype(int)
    es_manija = np.abs(fila - MANIJA).sum(axis=1) < 8   # LA MANIJA Y EL ROJO SE PARECEN (24 DE DIFERENCIA): UMBRAL ESTRICTO
    assert es_manija.any()
    centro = int(round(np.flatnonzero(es_manija).mean()))
    lejos_del_rojo = [np.abs(fila[centro + lado * 12] - ROJO).sum() for lado in (-1, +1)]
    assert all(d < 8 or d > 150 for d in lejos_del_rojo), "ni rojo ni pista: no se sabe qué se está mirando"
    return tuple(bool(d < 8) for d in lejos_del_rojo)


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
    a_la_vista(w)  # Y LO QUE SE VE: EL HUECO SALE EN ROJO POR SUS DOS ORILLAS, Y LOS TRAMOS NO
    assert junto_a_la_perilla(w, w.SLIDER_X, math.floor(dos[0][1])) == (False, True)
    assert junto_a_la_perilla(w, w.SLIDER_X, math.ceil(dos[1][0])) == (True, False)
    w.hide()


def test_s1_el_rojo_arranca_donde_se_acaba_el_tramo_en_la_perilla(ventana, pose):
    """Lo que se VE: con la perilla en el límite de un tramo, el rojo empieza pegado a ella y lo alcanzable no se pinta."""
    w = ventana
    a_la_vista(w)
    colocar(w, pose["la_de_paul"])
    (a, b), = w.SLIDER_X.tramos
    assert junto_a_la_perilla(w, w.SLIDER_X, math.floor(b)) == (False, True)  # AL FINAL DEL TRAMO: ROJO SOLO A LA DERECHA
    assert junto_a_la_perilla(w, w.SLIDER_X, math.ceil(a)) == (True, False)   # AL INICIO: ROJO SOLO A LA IZQUIERDA
    w.hide()


def test_s1_si_en_un_eje_no_se_alcanza_nada_el_slider_sale_todo_en_rojo(ventana):
    """«Nada se alcanza» no es «no se sabe»: lo primero es todo rojo; lo segundo (sin solver) no pinta nada."""
    w = ventana
    a_la_vista(w)
    w.SLIDER_X.poner_tramos([])
    assert junto_a_la_perilla(w, w.SLIDER_X, 0) == (True, True)
    w.SLIDER_X.poner_tramos(None)
    assert junto_a_la_perilla(w, w.SLIDER_X, 0) == (False, False)
    w.hide()


def test_s1_sin_solver_los_sliders_no_pintan_ni_frenan(robot_4gdl, abrir):
    """Un robot sin cinemática inversa no puede decir qué alcanza: los sliders quedan como antes (nada en rojo)."""
    w = abrir(robot_4gdl)
    assert all(s.tramos is None for s in w.sliders_xyz)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% S2: AL MOVER EL ROBOT, LOS TRAMOS SE ACTUALIZAN SOLOS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _por_slider_de_servos(w, poses):
    w.SLIDER_1.setValue(w.SLIDER_1.value() + 40)
    correr(w)


DESTINO_DE_CALCULAR = (356, 711, 1014)


def pedir_calcular(w):
    for campo, valor in zip((w.TE_X, w.TE_Y, w.TE_Z), DESTINO_DE_CALCULAR):
        campo.setText(str(valor))
    w.CALCULAR.click()


def _por_calcular(w, poses):
    pedir_calcular(w)
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


def _por_el_arranque(w, poses):
    w._ir_a_home()  # NO HAY BOTÓN DE HOME: ESTO ES LO QUE CORRE AL ABRIR LA APP


MEDIOS_S2 = {"sliders de servos": _por_slider_de_servos, "CALCULAR": _por_calcular, "COSECHAR": _por_cosechar,
             "reproducir": _por_reproducir, "paso a paso": _por_paso_a_paso, "el arranque en HOME": _por_el_arranque}


@pytest.mark.parametrize("medio", MEDIOS_S2)
def test_s2_al_mover_el_robot_los_tramos_se_actualizan_solos(ventana, poses, medio):
    w = ventana
    if medio in ("CALCULAR", "COSECHAR", "el arranque en HOME"):  # ESTOS TERMINAN EN HOME: SE SALE DE OTRA POSE PARA QUE HAYA CAMBIO
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


def correr_pasos(w, cuantos):
    """Como `correr`, pero solo unos pasos: el movimiento queda a media marcha."""
    w.temporizador_movimiento.stop()
    for _ in range(cuantos):
        w._siguiente_paso()
        w.reloj.t += w.temporizador_movimiento.interval() / 1000


def test_s2_al_detener_a_media_marcha_los_tramos_son_los_de_donde_quedo(ventana):
    w = ventana
    antes, salida = tramos_de(w), w.posicion
    pedir_calcular(w)
    correr_pasos(w, 8)
    assert w.cola, "el movimiento ya había terminado: la prueba no detuvo nada"
    w.BTN_DETENER.click()
    assert not w.cola
    assert math.dist(w.posicion, salida) > 50 and math.dist(w.posicion, DESTINO_DE_CALCULAR) > 50  # QUEDÓ A MEDIO CAMINO
    assert tramos_de(w) != antes and defectos_de_lo_pintado(w) == []


def test_s2_la_vista_previa_no_cambia_los_tramos_porque_el_robot_no_se_movio(ventana, poses):
    w = ventana
    abrir_editor(w, "previa.yaml", Punto(poses["arriba"]), Punto(poses["corte"], "lineal", 40))
    antes = tramos_de(w)
    w.BTN_VISTA_PREVIA.click()
    correr_pasos(w, 5)
    assert w.cola and w.en_vista_previa, "la vista previa no arrancó o ya había terminado: la prueba no mide nada"
    assert tramos_de(w) == antes   # A MEDIA VISTA PREVIA
    correr(w)
    assert tramos_de(w) == antes and defectos_de_lo_pintado(w) == []


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% S3: EL SLIDER SE DETIENE EN LA FRONTERA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def arrastrar_hasta(w, slider, destino, paso=7, alcanza_cada=10):
    """Un arrastre: el valor del slider cambia de poco en poco y, cada `alcanza_cada` cambios, el robot alcanza lo que
    le pidieron (en la app lo hace el temporizador entre un evento del ratón y otro). Devuelve lo que fue valiendo el slider.
    La perilla queda AGARRADA de verdad (isSliderDown): mientras dura, la app no le recoloca ese slider a la persona."""
    vistos = []
    valor = slider.value()
    slider.setSliderDown(True)   # LA PERSONA AGARRA LA PERILLA (EMITE sliderPressed)
    while valor != destino:
        valor += max(-paso, min(paso, destino - valor))
        slider.setValue(valor)
        vistos.append(slider.value())
        if len(vistos) % alcanza_cada == 0:
            correr(w)
    slider.setSliderDown(False)  # Y LA SUELTA (EMITE sliderReleased)
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


def test_s3_un_salto_al_extremo_tambien_se_detiene_en_la_frontera(ventana, pose):
    w = ventana
    colocar(w, pose["la_de_paul"])
    w.SLIDER_Y.setValue(w.SLIDER_Y.maximum())  # DE UN SOLO SALTO, SIN PASAR POR LOS VALORES DE EN MEDIO
    correr(w)
    assert w.SLIDER_Y.value() == math.floor(w.SLIDER_Y.tramos[0][1])
    assert alcanzable(w.robot, w.posicion)


TECLAS = {"Fin": QtCore.Qt.Key_End, "Inicio": QtCore.Qt.Key_Home, "RePág": QtCore.Qt.Key_PageUp,
          "AvPág": QtCore.Qt.Key_PageDown, "derecha": QtCore.Qt.Key_Right, "izquierda": QtCore.Qt.Key_Left}


def teclear(w, slider, tecla, veces=1):
    """La persona mueve el slider con el teclado: sin agarrar la perilla (no hay «soltar»)."""
    for _ in range(veces):
        QTest.keyClick(slider, TECLAS[tecla])
        correr(w)


@pytest.mark.parametrize("eje", (0, 1, 2))
def test_s3_con_el_teclado_el_slider_tampoco_entra_al_rojo(ventana, pose, eje):
    w = ventana
    colocar(w, pose["la_de_paul"])
    slider = w.sliders_xyz[eje]
    (a, b), = slider.tramos
    # CADA TECLA DEJA EL SLIDER Y LA PUNTA EN LA FRONTERA. NO BASTA CON QUE EL SLIDER ACABE DENTRO DEL TRAMO: SIN FRENO, EL
    # PLANEADOR SE NIEGA Y EL SLIDER REBOTA A DONDE ESTABA, QUE TAMBIÉN ES «DENTRO» (ASÍ ENGAÑÓ LA PRIMERA VERSIÓN DE ESTA PRUEBA)
    for tecla, frontera in (("Fin", math.floor(b)), ("derecha", math.floor(b)), ("Inicio", math.ceil(a)), ("izquierda", math.ceil(a))):
        teclear(w, slider, tecla)
        assert slider.value() == frontera, f"con la tecla {tecla} el slider no quedó en la frontera"
        assert w.posicion[eje] == pytest.approx(frontera, abs=1)
    for tecla in ("RePág", "AvPág"):
        teclear(w, slider, tecla)
        assert cubierto([(a, b)], slider.value())
    assert [e.mensaje for e in avisos_del_jog(w) if "No se mueve" in e.mensaje] == []  # FRENÓ EL SLIDER, NO EL PLANEADOR
    assert defectos_de_lo_pintado(w) == []


# LOS RINCONES QUE HALLÓ LA AUDITORÍA DEL 24-NOV: UN TRAMO SIN NI UN MILÍMETRO ENTERO, Y LA PUNTA POR DEBAJO DE LA BASE
def negativas_del_planeador(w):
    """Si el slider frena bien, el planeador nunca tiene que negarse (y el slider nunca rebota)."""
    return [e.mensaje for e in eventos(w, "aviso") if e.origen == "Jog" and "No se mueve" in e.mensaje]


def test_s3_en_una_esquina_de_los_limites_el_slider_no_entra_al_rojo(ventana, robot):
    """Con el hombro y el codo en su límite, lo que la punta alcanza sobre Y es un solo punto: el tramo no tiene ni un
    milímetro entero donde detenerse. El slider se queda donde está la punta, y lo dice una vez."""
    w = ventana
    hombro, codo = (robot.articulaciones[j].servo.minimo for j in (1, 2))
    colocar(w, (90, hombro, codo))
    slider, punta = w.SLIDER_Y, tuple(w.posicion)
    propio, = [t for t in slider.tramos if t[0] - 0.5 <= punta[1] <= t[1] + 0.5]
    assert math.ceil(propio[0]) > math.floor(propio[1]), "control: el tramo sí tiene un milímetro entero"
    inicial = slider.value()
    vistos = arrastrar_hasta(w, slider, inicial + 60)
    assert set(vistos) == {inicial}, "el slider entró al rojo"
    assert len(topes(w)) == 1 and negativas_del_planeador(w) == []
    assert tuple(w.posicion) == pytest.approx(punta, abs=0.5)


@pytest.fixture
def bajo_la_base(ventana, robot):
    """La punta por debajo del nivel de la base (z < 0). El slider de Z no baja de 0: ahí marca 0 aunque la punta no esté en 0."""
    colocar(ventana, (90, robot.articulaciones[1].servo.maximo, 40))
    assert ventana.posicion[2] < -50, "control: la punta no quedó bajo la base"
    return ventana


def test_s3_con_la_punta_bajo_la_base_el_slider_de_z_tambien_frena(bajo_la_base):
    """De un salto (tecla Fin, o un clic en la pista): arrastrando poco a poco la punta sube sola a la parte del tramo
    que el slider sí conoce y el defecto no se ve (así pasó en verde la primera versión de esta prueba)."""
    w = bajo_la_base
    slider = w.SLIDER_Z
    techo = max(b for _, b in slider.tramos)
    assert techo < slider.maximum() - 5, "control: sobre Z no hay rojo arriba"
    teclear(w, slider, "Fin")
    assert slider.value() == math.floor(techo), "el slider entró al rojo o rebotó"
    assert w.posicion[2] == pytest.approx(math.floor(techo), abs=1)
    assert len(topes(w)) == 1 and negativas_del_planeador(w) == []


@pytest.mark.parametrize("eje", (0, 1))
def test_s3_con_la_punta_bajo_la_base_mover_x_o_y_no_la_sube(bajo_la_base, eje):
    """El slider de Z marca 0 porque no puede marcar menos: mover otro eje no es una orden de subir la punta a 0."""
    w = bajo_la_base
    antes = tuple(w.posicion)
    destino = w.sliders_xyz[eje].value() + 30
    arrastrar_hasta(w, w.sliders_xyz[eje], destino)
    esperado = con_el_eje(antes, eje, destino)
    assert tuple(w.posicion) == pytest.approx(esperado, abs=1), "la punta no fue a donde se le pidió"
    assert negativas_del_planeador(w) == []


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
    w.SLIDER_X.setSliderDown(True)
    for x in range(-590, 601, 1):    # SIN CORRER EL MOVIMIENTO: LA PUNTA NO SE MUEVE Y CADA VALOR NUEVO ES UN PLAN NUEVO
        w.SLIDER_X.setValue(x)
    w.SLIDER_X.setSliderDown(False)
    nuevos = avisos_del_jog(w)[n0:]
    assert len(nuevos) == 1 and "postura" in nuevos[0].mensaje
    w.SLIDER_X.setSliderDown(True)       # OTRO ARRASTRE, OTRO AVISO
    w.SLIDER_X.setValue(550)
    w.SLIDER_X.setValue(551)
    w.SLIDER_X.setSliderDown(False)
    assert len(avisos_del_jog(w)) == n0 + 2


def test_s4_con_el_teclado_topar_avisa_una_vez_y_vuelve_a_avisar_tras_moverse(ventana, pose):
    """Sin agarrar la perilla no hay «soltar»: el aviso se calla al topar y solo un movimiento de verdad lo rearma."""
    w = ventana
    colocar(w, pose["la_de_paul"])
    slider = w.SLIDER_X
    teclear(w, slider, "Fin")
    assert len(topes(w)) == 1
    teclear(w, slider, "derecha", veces=5)     # CINCO PULSACIONES CONTRA LA FRONTERA
    teclear(w, slider, "Fin")
    assert len(topes(w)) == 1, "cada pulsación contra la frontera dejó otro aviso"
    en_la_frontera = slider.value()
    teclear(w, slider, "izquierda", veces=3)   # SE MUEVE DE VERDAD, HACIA DENTRO DEL TRAMO
    assert slider.value() < en_la_frontera and w.posicion[0] == pytest.approx(slider.value(), abs=1)
    teclear(w, slider, "Fin")
    assert len(topes(w)) == 2, "tras moverse, volver a topar ya no avisó"


def test_m04_con_un_movimiento_en_curso_los_sliders_quedan_apagados(ventana):
    """Banco de ataque, «dos a la vez»: mientras COSECHAR mueve el robot, nadie le pide otro destino por los sliders."""
    w = ventana
    todos = (*w.sliders_xyz, *w.sliders_servo)
    assert all(s.isEnabled() for s in todos)
    w.COSECHAR.click()
    assert w.cola and not any(s.isEnabled() for s in todos)
    correr(w)
    assert all(s.isEnabled() for s in todos)


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


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% J8: LLEVAR UN SLIDER AL BORDE DEL ROJO LLEVA LA PUNTA HASTA AHÍ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def poses_al_azar(w, cuantas, semilla=2411):
    """Servos al azar con la punta dentro de los tres sliders. Semilla fija: siempre salen las mismas poses.

    Con poses elegidas a mano el defecto no se veía (las de S3 pasaban): depende de los decimales de la punta, y solo
    aparece en 7 de cada 100 poses."""
    rng = np.random.default_rng(semilla)
    limites = [(j.servo.minimo, j.servo.maximo) for j in w.robot.articulaciones]
    poses = []
    while len(poses) < cuantas:
        servos = tuple(float(rng.uniform(lo, hi)) for lo, hi in limites)
        if all(s.minimum() <= c <= s.maximum() for s, c in zip(w.sliders_xyz, posicion(w.robot, servos))):
            poses.append(servos)
    return poses


def test_j08_llevar_un_slider_al_borde_del_rojo_lleva_la_punta_hasta_ahi(ventana):
    """Criterio J8, en 30 poses al azar. El aviso «No se mueve» lo calla el candado del tope, así que se mira el efecto:
    el movimiento salió y termina en el borde. El borde se toma ANTES de mover: si el planeador se niega, el slider
    rebota a donde estaba, y leerlo después esconde justo el defecto (así pasó en verde la primera versión de esta prueba)."""
    w = ventana
    pedidos, no_llega = 0, []
    for servos in poses_al_azar(w, 30):
        for eje, slider in enumerate(w.sliders_xyz):
            for extremo in (slider.minimum(), slider.maximum()):
                colocar(w, servos)
                salida = tuple(w.posicion)
                borde = slider.acotar(extremo, salida[eje])   # DONDE EMPIEZA EL ROJO (O SE ACABA EL SLIDER)
                if abs(borde - salida[eje]) < 0.5:
                    continue                                  # LA PUNTA YA ESTABA EN ESE BORDE: NO HAY NADA QUE RECORRER
                pedidos += 1
                slider.setValue(extremo)                      # DE UN SALTO: TECLA Fin O Inicio, O UN CLIC EN LA PISTA
                camino = [paso.servos for paso in w.cola if paso.servos is not None]
                if slider.value() != borde or not camino or tuple(posicion(w.robot, camino[-1])) != pytest.approx(
                        con_el_eje(salida, eje, borde), abs=0.5):
                    no_llega.append((tuple(round(c, 2) for c in salida), "XYZ"[eje], borde))
    w.detener()
    assert pedidos > 150, "control: casi ninguna pose tenía a dónde ir; la prueba no mide nada"
    assert no_llega == [], f"{len(no_llega)} de {pedidos} veces el slider se detuvo en el borde y la punta no fue"


# AL REDONDEAR A MILÍMETROS ENTEROS, Y Y Z SE ALEJAN DEL ROBOT: EL BORDE DE X QUEDA UN POCO MÁS ADENTRO
PUNTA_CON_DECIMALES = (100.3, 700.55, 800.55)


def test_j08_con_la_tecla_fin_la_punta_llega_al_borde_y_el_slider_no_rebota(ventana, robot):
    """Un caso concreto de los que fallaban, de principio a fin y por el teclado: la punta con decimales."""
    w = ventana
    colocar(w, angulos_servo(robot, PUNTA_CON_DECIMALES))
    borde = math.floor(w.SLIDER_X.tramos[-1][1])
    assert borde < w.SLIDER_X.maximum() - 5, "control: sobre X no hay rojo a la derecha"
    teclear(w, w.SLIDER_X, "Fin")
    assert w.SLIDER_X.value() == borde, "el slider rebotó a donde estaba"
    assert tuple(w.posicion) == pytest.approx(con_el_eje(PUNTA_CON_DECIMALES, 0, borde), abs=0.01)
    assert len(topes(w)) == 1 and negativas_del_planeador(w) == []


def test_j08_mover_un_slider_no_cambia_las_otras_dos_coordenadas(ventana):
    """La causa del defecto: los otros dos sliders guardan milímetros enteros y la punta tiene decimales. Si se toman
    como destino, la punta se sale de la recta sobre la que se pintó el rojo."""
    w = ventana
    movimientos = 0
    for servos in poses_al_azar(w, 8):
        for eje, slider in enumerate(w.sliders_xyz):
            colocar(w, servos)
            salida = tuple(w.posicion)
            destino = slider.acotar(slider.value() + 37, salida[eje])   # UN PASO CORTO HACIA DENTRO DE LO ALCANZABLE
            if abs(destino - salida[eje]) < 0.5:
                continue
            movimientos += 1
            slider.setValue(destino)
            correr(w)
            assert tuple(w.posicion) == pytest.approx(con_el_eje(salida, eje, destino), abs=0.01), \
                f"al mover {'XYZ'[eje]} la punta se salió de su recta"
    assert movimientos > 15, "control: casi ninguna pose tenía a dónde ir; la prueba no mide nada"


def test_j08_un_slider_ya_movido_sigue_siendo_destino_con_el_robot_en_camino(ventana, robot):
    """Lo que NO cambia: con la punta en camino sobre X, mover Y la lleva a la X y la Y pedidas (el slider de X ya no
    marca dónde está la punta: es una orden)."""
    w = ventana
    colocar(w, angulos_servo(robot, (575, 996, 1147)))
    w.SLIDER_X.setValue(875)
    correr_pasos(w, 6)
    assert w.cola and 580 < w.posicion[0] < 870, "control: la punta no quedó a medio camino"
    w.SLIDER_Y.setValue(1296)
    correr(w)
    assert tuple(w.posicion) == pytest.approx((875, 1296, 1147), abs=0.5)
