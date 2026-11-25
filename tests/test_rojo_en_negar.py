"""N — En «Al cambiar de postura: negarse», el rojo de cada slider de la punta cubre también lo que pediría cambiar de
postura, y el mismo aviso no se repite.

De dónde sale: el 25-nov-2026 Paul quiso llevar la punta a la izquierda (Y negativa) con el modo «negarse». El slider
de Y se veía libre hasta −1190 mm, pero cruzar Y = 0 pide voltear la base, y el planeador se negó 42 veces: 31 seguidas
en 21 segundos, con el slider rebotando cada vez. Las poses de estas pruebas son las de su registro de ese día.

Sin pantalla: QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_rojo_en_negar.py
"""
import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from cosechador.movimiento.trayectorias import hasta_donde_llega, posicion, tramo_articular, tramo_continuo
from cosechador.robot.cinematica_inversa import TOLERANCIA_TRAMOS_MM, alcanzable, angulos_servo, tramos_alcanzables
from cosechador.ui.ventana import muestras_del_jog
from test_grabacion_ventana import correr, eventos, ventana  # noqa: F401 (ventana ES EL FIXTURE)
from test_sliders import RANGO_DEL_EJE, arrastrar_hasta, colocar, con_el_eje, negativas_del_planeador, topes

X, Y, Z = 0, 1, 2
# DONDE ESTABA LA PUNTA DE PAUL EN SUS NEGATIVAS (logs/registro-2026-11-25.jsonl, 00:52 Y 01:06) Y OTRAS TRES POSES
PUNTAS = {"la de Paul a las 00:52": (534.0, 310.667, 1618.439), "la de Paul a la 01:06": (855.0, 300.0, 1654.0),
          "al frente y abajo": (900.0, 200.0, 600.0), "de lado": (200.0, 900.0, 900.0), "hacia atrás": (-600.0, 400.0, 1200.0)}
PUNTA_DE_PAUL = PUNTAS["la de Paul a las 00:52"]


def poner_postura(w, clave):
    w.SELECTOR_POSTURA.setCurrentIndex(w.SELECTOR_POSTURA.findData(clave))
    assert w.postura == clave


def en_negar(w, robot, punta=PUNTA_DE_PAUL):
    """El modo «negarse» y el robot con la punta ahí. Devuelve los servos."""
    poner_postura(w, "negar")
    servos = angulos_servo(robot, punta)
    colocar(w, servos)
    return servos


def tramo_de_la_perilla(w, eje):
    aqui = w.posicion[eje]
    propio, = [t for t in w.sliders_xyz[eje].tramos if t[0] - TOLERANCIA_TRAMOS_MM <= aqui <= t[1] + TOLERANCIA_TRAMOS_MM]
    return propio


def el_planeador_acepta(robot, servos, eje, valor, postura="negar"):
    """Lo mismo que le pregunta la app al planeador cuando ese slider se lleva a ese valor con la punta quieta."""
    desde = posicion(robot, servos)
    destino = con_el_eje(desde, eje, valor)
    plan, _ = tramo_continuo(robot, servos, destino, postura, pasos=muestras_del_jog(math.dist(desde, destino)))
    return plan is not None


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% N1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_n01_en_negar_el_slider_de_y_de_paul_ya_no_pinta_libre_el_lado_que_pide_voltear_la_base(ventana, robot):
    w = ventana
    colocar(w, angulos_servo(robot, PUNTA_DE_PAUL))
    assert w.postura == "despacio", "control: el modo de arranque es «girar despacio»"
    a, b = tramo_de_la_perilla(w, Y)
    assert a == pytest.approx(-1190.78, abs=0.5) and b == pytest.approx(1190.78, abs=0.5), "control: girando despacio, Y cruza el cero"
    en_negar(w, robot)
    a, b = tramo_de_la_perilla(w, Y)
    assert a == 0 and b == pytest.approx(1190.78, abs=0.5)   # DE Y = 0 PARA ABAJO, ROJO: CRUZAR PIDE VOLTEAR LA BASE
    # X Y Z NO PIDEN CAMBIAR DE POSTURA DESDE AHÍ: SU ROJO ES EL DE SIEMPRE
    for eje in (X, Z):
        assert tramo_de_la_perilla(w, eje) == pytest.approx(
            next(t for t in tramos_alcanzables(robot, w.posicion, eje, *RANGO_DEL_EJE[eje]) if t[0] <= w.posicion[eje] <= t[1]))


@pytest.mark.parametrize("nombre", PUNTAS)
@pytest.mark.parametrize("eje", (X, Y, Z))
def test_n01_en_negar_lo_que_queda_sin_rojo_es_lo_que_el_planeador_acepta(ventana, robot, nombre, eje):
    w = ventana
    servos = en_negar(w, robot, PUNTAS[nombre])
    a, b = tramo_de_la_perilla(w, eje)
    dentro = sorted({math.ceil(a), math.floor(b), *(int(a + (b - a) * k / 7) for k in range(1, 7))})
    assert all(a <= v <= b for v in dentro) and len(dentro) >= 3, "control: el tramo tiene de dónde probar"
    negados = [v for v in dentro if not el_planeador_acepta(robot, servos, eje, v)]
    assert negados == [], f"sin rojo, pero el planeador se niega a llevar {'XYZ'[eje]} a {negados}"
    # Y JUSTO AFUERA DE CADA BORDE: O YA NO SE ALCANZA, O EL PLANEADOR SE NIEGA, O SE ACABÓ EL SLIDER
    lo, hi = RANGO_DEL_EJE[eje]
    for afuera in (math.ceil(a) - 2, math.floor(b) + 2):
        if lo <= afuera <= hi and alcanzable(robot, con_el_eje(w.posicion, eje, afuera)):
            assert not el_planeador_acepta(robot, servos, eje, afuera), \
                f"{'XYZ'[eje]} = {afuera} quedó en rojo y el planeador sí lo acepta"


def test_n01_hasta_donde_llega_una_recta_lo_dice_el_mismo_recorrido_que_la_traza(robot):
    servos = angulos_servo(robot, PUNTA_DE_PAUL)
    al_otro_lado = con_el_eje(PUNTA_DE_PAUL, Y, -400.0)
    punto, servos_ahi = hasta_donde_llega(robot, servos, al_otro_lado, "negar", pasos=muestras_del_jog(710))
    assert punto[Y] == pytest.approx(0, abs=0.5) and (punto[X], punto[Z]) == pytest.approx((PUNTA_DE_PAUL[X], PUNTA_DE_PAUL[Z]))
    assert posicion(robot, servos_ahi) == pytest.approx(punto, abs=1e-6), "los servos que devuelve son los de ese punto"
    # SI EL TRAMO LLEGA, ES EL DESTINO MISMO: GIRANDO DESPACIO, Y EN «NEGAR» SI NO HAY QUE CAMBIAR DE POSTURA
    assert hasta_donde_llega(robot, servos, al_otro_lado, "despacio")[0] == pytest.approx(al_otro_lado)
    del_mismo_lado = con_el_eje(PUNTA_DE_PAUL, Y, 900.0)
    assert hasta_donde_llega(robot, servos, del_mismo_lado, "negar")[0] == pytest.approx(del_mismo_lado)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% N2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@pytest.mark.parametrize("gesto", ("arrastrando", "de un salto"))
def test_n02_en_negar_el_slider_llega_al_borde_del_rojo_y_la_punta_tambien_sin_rebotar(ventana, robot, gesto):
    w = ventana
    en_negar(w, robot)
    slider = w.SLIDER_Y
    if gesto == "arrastrando":
        vistos = arrastrar_hasta(w, slider, -400)
    else:
        slider.setValue(slider.minimum())
        vistos = [slider.value()]
        correr(w)
    assert min(vistos) == 0, "el slider entró al rojo"
    assert slider.value() == 0 and w.posicion[Y] == pytest.approx(0, abs=0.5)
    assert (w.posicion[X], w.posicion[Z]) == pytest.approx((PUNTA_DE_PAUL[X], PUNTA_DE_PAUL[Z]), abs=0.5)
    assert negativas_del_planeador(w) == [], "el planeador se negó: el slider rebotó"
    aviso, = topes(w)
    assert "Y = 0 mm" in aviso.mensaje and "sin cambiar de postura" in aviso.mensaje and aviso.datos["por_postura"]
    # LO QUE EL AVISO PROMETE ES CIERTO: CON «GIRAR DESPACIO» EL MISMO GESTO SÍ PASA
    assert "con «girar despacio» sí pasa" in aviso.mensaje
    poner_postura(w, "despacio")
    arrastrar_hasta(w, slider, -400)
    assert slider.value() == -400 and w.posicion[Y] == pytest.approx(-400, abs=0.5)


def test_n02_un_tope_por_alcance_sigue_diciendo_que_no_se_alcanza(ventana, robot):
    w = ventana
    en_negar(w, robot)
    arrastrar_hasta(w, w.SLIDER_Y, w.SLIDER_Y.maximum())   # HACIA ARRIBA NO HAY POSTURA QUE CAMBIAR: SE ACABA EL ALCANCE
    aviso, = topes(w)
    assert "no se alcanza" in aviso.mensaje and not aviso.datos["por_postura"]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% N3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_n03_cambiar_de_modo_repinta_el_rojo_al_momento_sin_mover_nada(ventana, robot):
    w = ventana
    colocar(w, angulos_servo(robot, PUNTA_DE_PAUL))
    enviados = len(w.enviados)
    assert tramo_de_la_perilla(w, Y)[0] < -1000
    poner_postura(w, "negar")
    assert tramo_de_la_perilla(w, Y)[0] == 0
    poner_postura(w, "despacio")
    assert tramo_de_la_perilla(w, Y)[0] < -1000
    assert len(w.enviados) == enviados, "cambiar de modo movió el robot"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% N4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def avisos_de_jog(w):
    return [e for e in eventos(w, "aviso") if e.origen == "Jog"]


def test_n04_insistir_contra_el_rojo_avisa_una_sola_vez_aunque_se_suelte_y_se_agarre(ventana, robot):
    w = ventana
    en_negar(w, robot)
    for _ in range(5):                       # CINCO GESTOS: AGARRAR, LLEVAR AL ROJO, SOLTAR
        arrastrar_hasta(w, w.SLIDER_Y, -300)
    assert len(avisos_de_jog(w)) == 1 and len(topes(w)) == 1
    # ALGO SE MOVIÓ DENTRO DE LO LIBRE: EL AVISO VUELVE A VALER
    arrastrar_hasta(w, w.SLIDER_Y, 200)
    arrastrar_hasta(w, w.SLIDER_Y, -300)
    assert len(topes(w)) == 2


def test_n04_la_racha_de_paul_el_planeador_se_niega_muchas_veces_y_avisa_una(ventana, robot):
    """Lo que le pasó a Paul, tal cual: el slider de Y pintado libre hasta −1190, el modo en «negarse» y cada gesto negado
    por el planeador. Para que el planeador tenga que negarse hoy, el rojo se deja como estaba ese día (sin N1)."""
    w = ventana
    colocar(w, angulos_servo(robot, PUNTA_DE_PAUL))
    w._pintar_alcance = lambda *a, **k: None          # EL ROJO SE QUEDA COMO SE PINTÓ GIRANDO DESPACIO
    w.SELECTOR_POSTURA.blockSignals(True)
    poner_postura(w, "negar")
    w.SELECTOR_POSTURA.blockSignals(False)
    for _ in range(13):
        arrastrar_hasta(w, w.SLIDER_Y, -300)
    assert len(negativas_del_planeador(w)) == 1, "la misma negativa salió más de una vez"
    # LA PUNTA AVANZÓ POR LO LIBRE Y SE QUEDÓ DE ESTE LADO; EL SLIDER REGRESÓ A DONDE ESTÁ LA PUNTA
    assert 0 <= w.posicion[Y] < 10 and w.SLIDER_Y.value() == round(w.posicion[Y])
    assert (w.posicion[X], w.posicion[Z]) == pytest.approx((PUNTA_DE_PAUL[X], PUNTA_DE_PAUL[Z]), abs=0.5)


def test_n04_otro_movimiento_del_robot_hace_que_el_aviso_vuelva_a_valer(ventana, robot):
    w = ventana
    en_negar(w, robot)
    arrastrar_hasta(w, w.SLIDER_Y, -300)
    arrastrar_hasta(w, w.SLIDER_Y, -300)
    assert len(topes(w)) == 1
    # EL ROBOT SE MUEVE POR OTRO LADO (UN PROGRAMA QUE SUBE EL HOMBRO 10°). LA PUNTA SIGUE SOBRE Y = 0: ES EL MISMO BORDE,
    # Y AUN ASÍ EL AVISO VUELVE A VALER, PORQUE EL ROBOT YA NO ESTÁ DONDE ESTABA
    hombro = list(w.servos)
    hombro[1] += 10
    w.ejecutar(tramo_articular(robot, w.servos, hombro), origen="Programa")
    correr(w)
    assert w.posicion[Y] == pytest.approx(0, abs=0.5) and w.SLIDER_Y.value() == 0, "control: la punta sigue en el borde"
    arrastrar_hasta(w, w.SLIDER_Y, -300)
    assert [t.datos["tope_mm"] for t in topes(w)] == [0, 0]


def test_n04_la_racha_con_la_punta_viajando_avisa_una_aunque_cambie_el_texto(ventana, robot):
    """La racha de la 01:06: la punta iba viajando en X y cada gesto sobre Y lo negaba el planeador con un texto distinto,
    porque cambiaba la X del cruce (572, 603, 618…). Es la misma negativa: sale una vez. El rojo se deja como ese día."""
    w = ventana
    colocar(w, angulos_servo(robot, PUNTA_DE_PAUL))
    w.selector_en_camino.setCurrentIndex(w.selector_en_camino.findData("cancelar"))
    w._pintar_alcance = lambda *a, **k: None          # EL ROJO SE QUEDA COMO SE PINTÓ GIRANDO DESPACIO
    w.SELECTOR_POSTURA.blockSignals(True)
    poner_postura(w, "negar")
    w.SELECTOR_POSTURA.blockSignals(False)
    w.SLIDER_X.setValue(w.SLIDER_X.value() + 300)     # LA PUNTA SALE DE VIAJE EN X
    por_donde_iba = []
    for _ in range(6):
        for _ in range(4):
            w._siguiente_paso()
            w.reloj.t += w.temporizador_movimiento.interval() / 1000
        por_donde_iba.append(round(w.posicion[X]))
        w.SLIDER_Y.setSliderDown(True)                # UN GESTO SOBRE Y: AGARRA, LLEVA A LA IZQUIERDA Y SUELTA
        w.SLIDER_Y.setValue(-300)
        w.SLIDER_Y.setSliderDown(False)
    assert w.cola and len(set(por_donde_iba)) == 6, "control: la punta seguía viajando y cada gesto la agarró en otra X"
    assert len(negativas_del_planeador(w)) == 1
    correr(w)
    assert w.posicion[X] == pytest.approx(PUNTA_DE_PAUL[X] + 300, abs=0.5), "lo que iba en camino llegó: negarse no lo cortó"


def test_n01_el_tramo_libre_del_otro_lado_de_un_hueco_se_queda_como_estaba(ventana, robot):
    """Abajo y cerca de la base, el slider de X tiene un hueco en medio (no se alcanza) y dos tramos libres. En «negarse»
    solo se recorta el tramo donde está la perilla; el otro, al que el slider de todos modos no llega, no se toca."""
    w = ventana
    en_negar(w, robot, (500.0, 150.0, 300.0))
    alcance = tramos_alcanzables(robot, w.posicion, X, *RANGO_DEL_EJE[X])
    assert len(alcance) == 2, "control: en esta pose el alcance de X tiene dos tramos"
    assert [tuple(t) for t in w.SLIDER_X.tramos] == [pytest.approx(t, abs=0.01) for t in alcance]


def test_n01_con_la_punta_en_camino_el_rojo_es_el_que_habra_al_llegar(ventana, robot):
    """«En cola»: con la punta en camino los sliders ya marcan el punto al que va, y su rojo es el de ese punto con la
    postura que el robot tendrá ahí. Abajo y cerca de la base el slider de X tiene un hueco; 150 mm más arriba ya no."""
    w = ventana
    w.selector_en_camino.setCurrentIndex(w.selector_en_camino.findData("en_cola"))
    en_negar(w, robot, (500.0, 150.0, 300.0))
    assert len(w.SLIDER_X.tramos) == 2, "control: aquí abajo X tiene un hueco"
    w.SLIDER_Z.setValue(w.SLIDER_Z.value() + 150)     # LA PUNTA SALE HACIA ARRIBA Y TODAVÍA NO LLEGA
    assert w.cola, "control: la punta va en camino"
    w._pintar_alcance()
    en_camino = [tuple(t) for t in w.SLIDER_X.tramos]
    correr(w)
    w._pintar_alcance()
    assert len(w.SLIDER_X.tramos) == 1 and en_camino == [pytest.approx(t, abs=0.5) for t in w.SLIDER_X.tramos]
