"""Pasar a «negarse» con un cambio de postura ya aceptado y esperando en cola.

D6 (opción A, aprobada por Paul el 25-nov-2026). Lo encontró la exploración larga: en «En cola», un arrastre pedía cambiar de
postura, «girar despacio» lo aceptaba, y si antes de que ese tramo empezara se pasaba a «negarse», el slider dejaba entrar a
donde el planeador se negaba y volvía solo. Ahora, al pasar a «negarse»:

  D6.1  se cancela lo pendiente desde el cambio de postura que no ha empezado; lo anterior se termina y el robot no brinca
  D6.2  sale UN aviso que dice qué se canceló y dónde queda la punta, y los sliders marcan ese punto
  D6.3  un giro que ya empezó se termina; no se deja a medias
  D6.4  si nada pendiente pide cambiar de postura, no se cancela nada ni se avisa
  D6.5  arrastrar otra vez el mismo slider ya no rebota (la lista de gestos del hallazgo deja de reproducirlo)
"""
import math

import numpy as np
import pytest

import explorador as ex
from cosechador.movimiento.trayectorias import Paso, posicion
from cosechador.robot import cargar_robot
from explorador import Gesto
from helpers_ventana import sin_dialogos_de_verdad

ROBOT = cargar_robot()
PUNTA = Gesto("elegir", "PESTANAS_MOVIMIENTO", 0.5)
# LOS TRES ARRASTRES DEL HALLAZGO: EL TERCERO PIDE UN CAMBIO DE POSTURA, «GIRAR DESPACIO» LO ACEPTA Y QUEDA ESPERANDO EN COLA
HASTA_EL_GIRO_EN_COLA = (PUNTA, Gesto("arrastrar", "SLIDER_Y", 0.1901583561638126, 402479, 1.5),
                         Gesto("arrastrar", "SLIDER_X", 0.25891675029296335, 536110, 0.008),
                         Gesto("arrastrar", "SLIDER_Y", 0.9677999949201714, 375441, 0.05))
A_NEGARSE = Gesto("elegir", "SELECTOR_POSTURA", 0.9)
EL_MISMO_SLIDER = Gesto("arrastrar", "SLIDER_Y", 0.6837036588129731, 624556, 0.008)
LLEGAR = Gesto("llegar", "", 0.0, 0, 0.3)
UN_MOMENTO = Gesto("esperar", "", 0.0, 0, 0.02)


@pytest.fixture
def sesion():
    s = ex.Sesion(ROBOT)
    with sin_dialogos_de_verdad():
        yield s
    s.cerrar()


def _con_el_giro_en_cola(s):
    """Deja la punta en camino con un cambio de postura aceptado y sin empezar. Devuelve en qué lugar de la cola está."""
    for g in HASTA_EL_GIRO_EN_COLA:
        s.hacer(g)
    w = s.w
    assert (w.postura, w.en_camino) == ("despacio", "en_cola")
    giros = [i for i, p in enumerate(w.cola) if p.cambio_de_postura]
    assert giros, "la preparación ya no deja un cambio de postura esperando en cola: estas pruebas no probarían nada"
    assert any(p.servos is not None for p in list(w.cola)[:giros[0]]), "no queda nada pendiente antes del giro"
    return giros[0]


def _avisos_de_postura(w, desde=0):
    return [e for e in w.x.eventos[desde:] if e.origen == "Postura"]


def _lo_que_falta_de(pasos, w):
    """De unos pasos que iban al frente de la cola, los que el robot todavía no ha hecho."""
    hechos = next((i + 1 for i, p in enumerate(pasos) if p.servos is not None and np.allclose(p.servos, w.servos)), 0)
    return pasos[hechos:]


def _servos_pendientes(w):
    return [p.servos for p in w.cola if p.servos is not None]


def test_d6_1_al_pasar_a_negarse_se_cancela_desde_el_cambio_de_postura_y_lo_anterior_se_termina(sesion):
    s, w = sesion, sesion.w
    corte = _con_el_giro_en_cola(s)
    se_queda = list(w.cola)[:corte]
    s.hacer(A_NEGARSE)
    assert not any(p.cambio_de_postura for p in w.cola), "el cambio de postura sigue en la cola"
    assert list(w.cola) == se_queda[len(se_queda) - len(w.cola):], "se tocó lo que iba ANTES del cambio de postura"
    pendientes, enviadas = _servos_pendientes(w), len(w.x.enviados)
    assert pendientes, "no quedó nada por terminar: la prueba no vería que lo anterior se termina"
    s.hacer(LLEGAR)                                  # SI ALGO BRINCA, SE NIEGA O REVIENTA, LA SESIÓN LO LEVANTA COMO HALLAZGO
    mandadas = [orden[1] for orden in w.x.enviados[enviadas:]]
    assert len(mandadas) == len(pendientes) and np.allclose(mandadas, pendientes), "a la placa salió algo de lo cancelado"
    assert not w.cola and not w.ocupado


def test_d6_1_tambien_un_programa_en_marcha_se_corta_en_el_cambio_de_postura(sesion):
    s, w = sesion, sesion.w
    paso = lambda grados: Paso(servos=(w.servos[0] + grados, *w.servos[1:]))
    antes, despues = [paso(1), paso(2)], [paso(3), paso(4), Paso(aviso="Programa terminado.", nivel="ok")]
    w.ejecutar(antes + [Paso(aviso="Cambio de postura cerca de (0, 0, 0).", nivel="aviso", cambio_de_postura=True)] + despues,
               origen="Programa")
    s.hacer(A_NEGARSE)
    s.hacer(LLEGAR)
    assert np.allclose([orden[1] for orden in w.x.enviados], [p.servos for p in antes]), "salió algo de lo cancelado"
    assert not any("Programa terminado" in e.mensaje for e in w.x.eventos), "dijo que terminó un programa que se cortó"
    assert len(_avisos_de_postura(w)) == 1 and not w.ocupado


def test_d6_1_si_el_cambio_de_postura_era_lo_siguiente_el_robot_se_queda_donde_esta(sesion):
    s, w = sesion, sesion.w
    aqui = tuple(w.posicion)
    w.ejecutar([Paso(aviso="Cambio de postura cerca de (0, 0, 0).", nivel="aviso", cambio_de_postura=True),
                Paso(servos=(w.servos[0] + 1, *w.servos[1:])), Paso(servos=(w.servos[0] + 2, *w.servos[1:]))], origen="Programa")
    s.hacer(A_NEGARSE)
    assert not w.cola
    s.hacer(LLEGAR)
    assert not w.x.enviados and math.dist(w.posicion, aqui) < 0.01 and not w.ocupado and not w.BTN_DETENER.isEnabled()
    avisos = _avisos_de_postura(w)
    assert len(avisos) == 1 and str(tuple(int(round(c)) for c in aqui)) in avisos[0].mensaje


def test_d6_2_sale_un_solo_aviso_que_dice_donde_queda_la_punta_y_los_sliders_lo_marcan(sesion):
    s, w = sesion, sesion.w
    corte = _con_el_giro_en_cola(s)
    fin = posicion(ROBOT, [p.servos for p in list(w.cola)[:corte] if p.servos is not None][-1])
    eventos = len(w.x.eventos)
    s.hacer(A_NEGARSE)
    avisos = _avisos_de_postura(w, eventos)
    assert len(avisos) == 1, [e.mensaje for e in w.x.eventos[eventos:]]
    aviso = avisos[0]
    assert aviso.nivel == "aviso" and "«negarse»" in aviso.mensaje and "se canceló el cambio de postura" in aviso.mensaje
    assert str(tuple(int(round(c)) for c in fin)) in aviso.mensaje, aviso.mensaje
    assert math.dist(aviso.datos["queda_mm"], fin) < 0.01 and math.dist(w._destino_jog, fin) < 0.01
    assert all(abs(slider.value() - c) <= 1 for slider, c in zip(w.sliders_xyz, fin)), "los sliders no marcan dónde queda"
    s.hacer(LLEGAR)
    assert math.dist(w.posicion, fin) < 0.01, "la punta no quedó donde dijo el aviso"
    assert len(_avisos_de_postura(w)) == 1, "el aviso se repitió"


def test_d6_3_un_giro_que_ya_empezo_se_termina(sesion):
    s, w = sesion, sesion.w
    _con_el_giro_en_cola(s)
    destino = w._destino_jog
    for _ in range(5000):
        if not any(p.cambio_de_postura for p in w.cola):
            break
        s.hacer(UN_MOMENTO)
    assert _servos_pendientes(w), "el giro terminó antes de poder cambiar de modo: la prueba no probaría nada"
    cola = list(w.cola)
    s.hacer(A_NEGARSE)
    assert list(w.cola) == cola[len(cola) - len(w.cola):] and not _avisos_de_postura(w), "se canceló un giro que ya había empezado"
    s.hacer(LLEGAR)
    assert math.dist(w.posicion, destino) < 1.0, "el movimiento que ya giraba no llegó a donde iba"


def test_d6_4_sin_cambio_de_postura_pendiente_no_se_cancela_nada_ni_se_avisa(sesion):
    s, w = sesion, sesion.w
    s.hacer(PUNTA)
    s.hacer(Gesto("arrastrar", "SLIDER_Y", 0.1901583561638126, 402479, 0.05))
    cola, destino = list(w.cola), w._destino_jog
    assert _servos_pendientes(w) and not any(p.cambio_de_postura for p in cola)
    s.hacer(A_NEGARSE)
    assert list(w.cola) == cola[len(cola) - len(w.cola):] and w._destino_jog == destino and not _avisos_de_postura(w)
    s.hacer(LLEGAR)
    s.hacer(Gesto("elegir", "SELECTOR_POSTURA", 0.1))
    s.hacer(A_NEGARSE)                               # CON EL ROBOT QUIETO TAMPOCO HAY NADA QUE CANCELAR
    assert not _avisos_de_postura(w)


def test_d6_4_un_giro_en_el_lugar_no_se_cancela_porque_negarse_tambien_lo_permite(sesion):
    s, w = sesion, sesion.w
    plan = [Paso(servos=(w.servos[0] + 1, *w.servos[1:])), Paso(aviso="Giro en el lugar cerca de (0, 0, 0) (la punta no se mueve)."),
            Paso(servos=(w.servos[0] + 2, *w.servos[1:])), Paso(aviso="Programa terminado.", nivel="ok")]
    w.ejecutar(plan, origen="Programa")
    s.hacer(A_NEGARSE)
    s.hacer(LLEGAR)
    assert np.allclose([orden[1] for orden in w.x.enviados], [p.servos for p in plan if p.servos is not None])
    assert any("Programa terminado" in e.mensaje for e in w.x.eventos) and not _avisos_de_postura(w)


def test_d6_4_en_vista_previa_no_se_cancela_nada_porque_el_robot_no_se_mueve(sesion):
    s, w = sesion, sesion.w
    plan = [Paso(servos=(w.servos[0] + 1, *w.servos[1:])), Paso(aviso="Cambio de postura.", nivel="aviso", cambio_de_postura=True),
            Paso(servos=(w.servos[0] + 2, *w.servos[1:]))]
    w.ejecutar(plan, solo_simulacion=True, origen="Vista previa")
    cola = list(w.cola)
    s.hacer(A_NEGARSE)
    assert list(w.cola) == cola[len(cola) - len(w.cola):] and any(p.cambio_de_postura for p in w.cola)
    assert not _avisos_de_postura(w)
    s.hacer(LLEGAR)
    assert not w.x.enviados


def test_d6_5_arrastrar_otra_vez_el_mismo_slider_ya_no_rebota(sesion):
    s, w = sesion, sesion.w
    _con_el_giro_en_cola(s)
    delante = list(w.cola)[:len(w.cola) - w._tramo_jog[1]]  # LO DEL OTRO SLIDER, QUE VA ANTES DEL TRAMO QUE PEDÍA EL GIRO
    s.hacer(A_NEGARSE)
    s.hacer(EL_MISMO_SLIDER)                         # ANTES: HALLAZGO «rebote» (EL SLIDER DEJABA ENTRAR Y VOLVÍA SOLO)
    falta = _lo_que_falta_de(delante, w)
    assert falta and list(w.cola)[:len(falta)] == falta, "«En cola» perdió lo que iba delante del tramo de ese slider"
    s.hacer(LLEGAR)
    assert not w.cola and not w.ocupado
