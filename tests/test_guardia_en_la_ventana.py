"""La guardia de salida DENTRO de la ventana (fase C): GS1, GS4, GS6, GS10 (la parte de pantalla) y GS12.

Quién lo lee y cuándo: quien revise o toque `poner_servos`, la franja de «Movimiento detenido» o el instrumento de
pruebas de la ventana (helpers_ventana.py). La guardia en sí (qué es un paso válido, el candado, `reiniciar`) se prueba en
test_guardia.py; aquí solo se prueba que la ventana REAL la usa, y bien.

Todo se ejerce como lo haría una persona: arrastrando un slider, pulsando un botón. Para provocar una negativa de forma
legítima se SIEMBRA un defecto con monkeypatch (un plan que brinca 40° en un paso), como hace test_explorador.py: el
planeador sano nunca lo produce. El reloj es el falso del instrumento (`w.x.reloj`), compartido con la guardia.

Cada criterio nace con su prueba nombrada por su número:
  GS1  cualquier orden hacia la placa pasa por la guardia; no hay otro camino
  GS4  la guardia se niega: el movimiento se detiene, el robot se queda, se avisa UNA vez y queda en el registro
  GS5/GS10  tras negarse queda detenida hasta que la persona reinicia; la pantalla lo dice y cómo reiniciar
  GS6  todo lo que hoy funciona pasa igual: la guardia no estorba
  GS12 no hay forma de apagarla ni de rodearla: un solo lugar le habla a la placa
"""
import ast
import glob
import inspect
import os
import sys
import time

import pytest
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtTest import QTest

import explorador as ex
from cosechador import registro as reg
from cosechador.hardware.guardia import GuardiaDeSalida, OrdenNegada
from cosechador.movimiento.trayectorias import Paso
from cosechador.ui import ventana as ventana_real
from cosechador.ui.ventana import VentanaPrincipal
from helpers_guardia import CARPETA_REPO, EnlaceFalso, robot_de
from helpers_ventana import (APP, avanzar, borrar_desechable, carpeta_desechable, cerrar, correr, sin_dialogos_de_verdad,
                             ventana_aislada)

ROBOT = robot_de("3gdl")
DESTINO_DE_CALCULAR = (356, 711, 1014)       # UN PUNTO QUE EL ROBOT ALCANZA (EL MISMO DE test_sliders.py)
MODOS_EN_CAMINO = ("reemplazar", "en_cola", "cancelar")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INSTRUMENTO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def pasar(n=6):
    for _ in range(n):
        APP.processEvents()


@pytest.fixture
def w():
    """La ventana REAL aislada, a la vista (sin pantalla). Al cerrar, nada reventó dentro de un slot de Qt."""
    carpeta = carpeta_desechable()
    with sin_dialogos_de_verdad():
        ventana = ventana_aislada(ROBOT, carpeta)
        ex.sembrar_programas(ventana)
        ventana.show()
        ventana.resize(1280, 720)
        pasar()
        yield ventana
        reventones = list(ventana.x.excepciones)
    cerrar(ventana)
    borrar_desechable(carpeta)
    assert not reventones, f"una excepción se escapó a un slot de Qt: {reventones[0][1]!r}"


def del_registro(w, origen="Guardia", nivel=None):
    return [e for e in w.x.eventos if e.origen == origen and (nivel is None or e.nivel == nivel)]


def arrastrar(w, slider, destino, eventos=12, entre=0.05):
    """Una persona agarra la perilla, la lleva a `destino` con `eventos` movimientos del ratón y la suelta."""
    salida = slider.value()
    slider.setSliderDown(True)
    for i in range(1, eventos + 1):
        slider.setSliderPosition(int(round(salida + (destino - salida) * i / eventos)))
        avanzar(w, entre)
    slider.setSliderDown(False)
    pasar()


def clic(boton):
    QTest.mouseClick(boton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, boton.rect().center())
    pasar()


def sembrar_salto(monkeypatch):
    """Defecto sembrado: el tramo de un slider de servo llega de UN solo paso, brincando todo lo pedido (el planeador sano
    lo parte en pasos de 15°)."""
    monkeypatch.setattr(ventana_real, "tramo_articular",
                        lambda robot, desde, hasta, ms=None: [Paso(servos=tuple(float(v) for v in hasta), ms=ms)])


def sembrar_salto_a_media_marcha(monkeypatch):
    """Defecto sembrado: dos pasos buenos (5° cada uno), un brinco de 40° y dos pasos más. Sirve para ver que lo pendiente
    después del brinco no sale."""
    def tramo(robot, desde, hasta, ms=None):
        a = [float(v) for v in desde]
        pasos = []
        for grados in (5, 10, 50, 55, 60):
            # CADA PASO LLEVA SU AVISO DE «LLEGUÉ»: SI SE NEGÓ, ESE AVISO NO DEBE DARSE (MENTIRÍA)
            pasos.append(Paso(servos=(a[0] + grados, *a[1:]), ms=ms, aviso=f"llegué a +{grados}", nivel="ok"))
        return pasos
    monkeypatch.setattr(ventana_real, "tramo_articular", tramo)


def a_la_placa(w):
    """Lo que de verdad salió hacia la placa, en orden (el espía del instrumento)."""
    return [servos for _hora, servos, _origen in w.x.enviados]


def se_nego_una_vez(w, monkeypatch):
    """Deja a la guardia detenida por UN brinco legítimo de sembrar: arrastre de 40° del servo 1 en un solo movimiento."""
    sembrar_salto(monkeypatch)
    arrastrar(w, w.SLIDER_1, w.SLIDER_1.value() + 40, eventos=1)
    assert w.enlace.detenida
    return tuple(w.servos)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LO QUE HOY FUNCIONA: UNA PERSONA LO HACE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _escenario_slider_de_servo(w):
    arrastrar(w, w.SLIDER_1, w.SLIDER_1.value() + 40)
    correr(w)


def _jog_de_punta(modo):
    def escenario(w):
        w.selector_en_camino.setCurrentIndex(w.selector_en_camino.findData(modo))
        w.PESTANAS_MOVIMIENTO.setCurrentIndex(ventana_real.PESTANA_XYZ)
        arrastrar(w, w.SLIDER_X, w.SLIDER_X.value() + 60, eventos=6, entre=0.02)
        avanzar(w, 0.1)                                    # LA PUNTA VA EN CAMINO...
        arrastrar(w, w.SLIDER_Y, w.SLIDER_Y.value() - 40, eventos=6, entre=0.02)   # ...Y SE MUEVE OTRO SLIDER
        correr(w)
    return escenario


def _escenario_calcular(w):
    for campo, valor in zip((w.TE_X, w.TE_Y, w.TE_Z), DESTINO_DE_CALCULAR):
        campo.setText(str(valor))
    clic(w.CALCULAR)
    correr(w)


def _escenario_cosechar(w):
    clic(w.COSECHAR)
    correr(w)


def _elegir_programa(w, archivo):
    indice = w.SELECTOR_PROGRAMA.findData(archivo)
    assert indice >= 0, f"control: el selector no trae {archivo}"
    w.SELECTOR_PROGRAMA.setCurrentIndex(indice)
    pasar()


def _escenario_programa_de_puntos(w):
    _elegir_programa(w, "demo-cosecha.yaml")
    clic(w.BTN_REPRODUCIR)
    correr(w)


def _escenario_programa_grabado(w):
    _elegir_programa(w, "ejemplo-grabacion.yaml")
    clic(w.BTN_REPRODUCIR)
    correr(w)


def _escenario_grabar_y_reproducir(w):
    clic(w.BTN_GRABAR)
    arrastrar(w, w.SLIDER_2, w.SLIDER_2.value() + 30)
    correr(w)
    w.x.nombres.append("mi-grabacion")
    clic(w.BTN_GRABAR)                                      # DETENER LA GRABACIÓN: SE PIDE EL NOMBRE Y SE GUARDA
    assert os.path.isfile(os.path.join(w.carpeta_programas, "mi-grabacion.yaml"))
    _elegir_programa(w, "mi-grabacion.yaml")
    clic(w.BTN_REPRODUCIR)
    correr(w)


def _escenario_detener(w):
    for campo, valor in zip((w.TE_X, w.TE_Y, w.TE_Z), DESTINO_DE_CALCULAR):
        campo.setText(str(valor))
    clic(w.CALCULAR)
    avanzar(w, 0.4)
    assert w.cola, "control: el movimiento ya había terminado y Detener no iba a detener nada"
    clic(w.BTN_DETENER)


def _escenario_pasos_del_editor(w):
    _elegir_programa(w, "demo-cosecha.yaml")
    clic(w.BTN_EDITAR_PUNTOS)
    assert w.editando
    for boton in (w.BTN_PASO_ADELANTE, w.BTN_PASO_ADELANTE, w.BTN_PASO_ADELANTE, w.BTN_PASO_ATRAS, w.BTN_PASO_ATRAS):
        clic(boton)
        correr(w)


ESCENARIOS = {
    "slider de servo": _escenario_slider_de_servo,
    **{f"jog de la punta, en camino «{m}»": _jog_de_punta(m) for m in MODOS_EN_CAMINO},
    "CALCULAR": _escenario_calcular,
    "COSECHAR": _escenario_cosechar,
    "programa de puntos": _escenario_programa_de_puntos,
    "programa grabado": _escenario_programa_grabado,
    "grabar y reproducir": _escenario_grabar_y_reproducir,
    "Detener a media marcha": _escenario_detener,
    "pasos adelante y atrás del editor": _escenario_pasos_del_editor,
}


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GS1: TODA ORDEN PASA POR LA GUARDIA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _dentro_de_la_guardia():
    marco = sys._getframe(2)
    while marco is not None:
        if marco.f_code is GuardiaDeSalida.enviar.__code__:
            return True
        marco = marco.f_back
    return False


@pytest.mark.parametrize("escenario", list(ESCENARIOS))
def test_gs01_toda_orden_que_llega_a_la_placa_paso_por_la_guardia(w, escenario):
    """Se vigila el enlace CRUDO (el que habla con la placa): cada orden que le llega debe llegar desde dentro de
    `GuardiaDeSalida.enviar`. Si algún camino de la ventana lo llamara directo, la orden saldría sin revisión."""
    crudo = w.enlace._enlace
    saltaron_la_guardia, llegadas = [], []
    original = crudo.enviar

    def vigilado(*servos):
        llegadas.append(servos)
        if not _dentro_de_la_guardia():
            saltaron_la_guardia.append(servos)
        return original(*servos)

    crudo.enviar = vigilado
    ESCENARIOS[escenario](w)
    assert len(llegadas) > 3, "control: el escenario no mandó nada a la placa, no prueba nada"
    assert saltaron_la_guardia == []
    assert not w.enlace.detenida


def test_gs01_la_ventana_siempre_habla_con_la_placa_por_una_guardia(w):
    assert type(w.enlace) is GuardiaDeSalida
    assert w.enlace.robot is w.robot, "la guardia debe revisar con las reglas del robot de la ventana"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GS4: LA GUARDIA SE NIEGA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_gs04_si_la_guardia_se_niega_el_robot_se_queda_se_avisa_una_vez_y_queda_en_el_registro(w, monkeypatch):
    antes_servos, antes_pos = tuple(w.servos), tuple(w.posicion)
    pedido = [float(s.value()) for s in w.sliders_servo]   # LOS SLIDERS DAN GRADOS ENTEROS: ESO ES LO QUE SE PIDE
    pedido[0] += 40
    salio_antes = len(w.x.enviados)
    muestras_antes = w.grabadora.muestras
    dibujos = []
    original = w._dibujar
    monkeypatch.setattr(w, "_dibujar", lambda servos: (dibujos.append(tuple(servos)), original(servos))[1])
    sembrar_salto(monkeypatch)

    arrastrar(w, w.SLIDER_1, w.SLIDER_1.value() + 40, eventos=1)

    assert len(w.x.enviados) == salio_antes, "la orden negada salió hacia la placa"
    assert tuple(w.servos) == antes_servos and tuple(w.posicion) == antes_pos, "el robot se movió"
    assert dibujos == [], "se dibujó un movimiento que no ocurrió"
    assert not w.cola and not w.temporizador_movimiento.isActive() and not w.en_movimiento
    assert not w.BTN_DETENER.isEnabled() and not w.ocupado
    assert w.grabadora.muestras == muestras_antes, "la grabadora anotó una orden que no salió"
    # LA PERSONA SOLTÓ EL SLIDER: SU PERILLA Y LAS ETIQUETAS VUELVEN A DONDE ESTÁ EL ROBOT
    assert w.SLIDER_1.value() == round(antes_servos[0]) and w.VS_1.text() == f"{antes_servos[0]:.0f}°"
    assert [e.text() for e in w.valores_xyz] == [f"{round(c) + 0:.0f}" for c in antes_pos]
    # UN AVISO, EN EL REGISTRO, CON LO QUE LA PERSONA NECESITA
    eventos = del_registro(w)
    assert len(eventos) == 1, [e.mensaje for e in eventos]
    evento = eventos[0]
    negada = w.x.negadas[-1][1]
    assert (evento.nivel, evento.origen) == ("error", "Guardia")
    assert negada.motivo in evento.mensaje
    assert evento.datos["clave"] == negada.clave == "salto_excedido" and evento.datos["motivo"] == negada.motivo
    assert tuple(evento.datos["servos"]) == tuple(pedido)
    assert negada.motivo in w.statusBar().currentMessage()
    assert len(w.x.negadas) == 1


def test_gs04_lo_que_faltaba_del_movimiento_se_cancela_y_nada_posterior_sale(w, monkeypatch):
    """Cinco pasos: dos buenos, un brinco, dos más. Salen los dos buenos; el brinco y lo que seguía, no."""
    antes = tuple(w.servos)
    salio_antes = len(w.x.enviados)
    sembrar_salto_a_media_marcha(monkeypatch)
    w.SLIDER_1.setValue(w.SLIDER_1.value() + 1)               # (la tecla de flecha: un gesto sin arrastre)
    for _ in range(10):                                       # PASO A PASO, HASTA EL INSTANTE EN QUE LA GUARDIA SE NIEGA
        avanzar(w, w.temporizador_movimiento.interval() / 1000)
        if w.enlace.detenida:
            break
    assert w.enlace.detenida
    # ...Y AHÍ MISMO (SIN DEJAR PASAR MÁS TIEMPO) LO PENDIENTE YA ESTÁ CANCELADO Y LOS CONTROLES LIBRES
    assert not w.cola and not w.temporizador_movimiento.isActive() and not w.ocupado and not w.BTN_DETENER.isEnabled()
    avisos = [e.mensaje for e in w.x.eventos]
    assert "llegué a +10" in avisos and not any(m in avisos for m in ("llegué a +50", "llegué a +55", "llegué a +60")), avisos
    correr(w)
    salieron = a_la_placa(w)[salio_antes:]
    assert [round(s[0] - antes[0]) for s in salieron] == [5, 10], salieron
    assert tuple(w.servos) == salieron[-1], "el robot debe quedarse donde llegó el último paso bueno"
    assert not w.cola and not w.temporizador_movimiento.isActive()
    assert len(del_registro(w, nivel="error")) == 1
    avanzar(w, 2.0)
    assert len(a_la_placa(w)) == salio_antes + 2, "después de negarse salió algo más"


def test_gs04_con_una_grabacion_en_curso_la_orden_negada_no_se_graba(w, monkeypatch):
    clic(w.BTN_GRABAR)
    arrastrar(w, w.SLIDER_2, w.SLIDER_2.value() + 20)
    correr(w)
    grabadas = len(w.grabadora.muestras)
    assert grabadas > 1
    sembrar_salto(monkeypatch)
    arrastrar(w, w.SLIDER_1, w.SLIDER_1.value() + 40, eventos=1)
    assert w.enlace.detenida
    assert len(w.grabadora.muestras) == grabadas, "la grabación anotó la orden negada"


def test_gs04_un_jog_de_la_punta_negado_tambien_se_detiene_y_avisa_una_vez(w, monkeypatch):
    """El otro camino de los sliders: la punta. El brinco se siembra en `tramo_continuo`."""
    monkeypatch.setattr(ventana_real, "tramo_continuo",
                        lambda robot, desde, destino, postura, pasos=None: (
                            [Paso(servos=(desde[0] + 40, *desde[1:]), ms=None)], None))
    w.PESTANAS_MOVIMIENTO.setCurrentIndex(ventana_real.PESTANA_XYZ)
    antes = tuple(w.servos)
    arrastrar(w, w.SLIDER_X, w.SLIDER_X.value() + 30, eventos=1)
    assert w.enlace.detenida and tuple(w.servos) == antes and not w.cola
    assert len(del_registro(w, nivel="error")) == 1


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GS5 / GS10: DETENIDA HASTA QUE LA PERSONA REINICIA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_gs10_la_franja_esta_escondida_mientras_todo_va_bien(w):
    assert w.FRANJA_GUARDIA.isHidden() and not w.FRANJA_GUARDIA.isVisibleTo(w)
    assert w.centralwidget.layout().indexOf(w.FRANJA_GUARDIA) == 1, "la franja va bajo el título (índice 1)"
    assert w.BTN_REINICIAR_GUARDIA.text() == "Reiniciar"


def test_gs10_detenida_la_pantalla_dice_que_paso_y_como_reiniciar(w, monkeypatch):
    se_nego_una_vez(w, monkeypatch)
    franja = w.FRANJA_GUARDIA
    assert franja.isVisibleTo(w)
    textos = " ".join(e.text() for e in franja.findChildren(QtWidgets.QLabel))
    assert "MOVIMIENTO DETENIDO" in textos
    assert w.enlace.violacion.motivo in textos
    assert "No se mandó esa orden: el robot sigue donde estaba. Revise y pulse Reiniciar para volver a moverlo." in textos
    boton = w.BTN_REINICIAR_GUARDIA
    assert boton.isVisibleTo(w) and boton.isEnabled() and boton.text() == "Reiniciar"
    assert all(ord(c) < 0x2000 for c in boton.text() + textos), "la franja no lleva emojis ni símbolos"


def test_gs10_detenida_cada_gesto_da_el_mismo_aviso_una_sola_vez_y_un_arrastre_es_un_gesto(w, monkeypatch):
    se_nego_una_vez(w, monkeypatch)
    sin_sembrar = monkeypatch.undo   # (EL DEFECTO YA NO HACE FALTA: DETENIDA, TODO SE NIEGA, SEA BUENO O MALO)
    sin_sembrar()
    salio = len(w.x.enviados)
    errores_antes = len(del_registro(w, nivel="error"))

    def avisos():
        return del_registro(w, nivel="aviso")

    # UN ARRASTRE COMPLETO DE 30 MOVIMIENTOS DEL RATÓN: UN SOLO AVISO
    arrastrar(w, w.SLIDER_2, w.SLIDER_2.value() + 30, eventos=30, entre=0.01)
    assert len(avisos()) == 1, [e.mensaje for e in avisos()]
    texto = avisos()[0].mensaje
    assert w.enlace.violacion.motivo in texto and "Reiniciar" in texto
    # OTRO ARRASTRE ES OTRO GESTO: OTRO AVISO, IGUAL
    arrastrar(w, w.SLIDER_3, w.SLIDER_3.value() + 20, eventos=20, entre=0.01)
    assert len(avisos()) == 2 and avisos()[1].mensaje == texto
    # UN BOTÓN DE MOVER (CALCULAR) ES OTRO GESTO
    for campo, valor in zip((w.TE_X, w.TE_Y, w.TE_Z), DESTINO_DE_CALCULAR):
        campo.setText(str(valor))
    avanzar(w, 1.0)
    clic(w.CALCULAR)
    correr(w)
    assert len(avisos()) == 3 and avisos()[2].mensaje == texto
    # NADA SALIÓ A LA PLACA, EL ROBOT SIGUE DONDE ESTABA Y NO SE REPITIÓ EL ERROR (EL ERROR ES DE LA PRIMERA NEGATIVA)
    assert len(w.x.enviados) == salio
    assert len(del_registro(w, nivel="error")) == errores_antes
    assert w.FRANJA_GUARDIA.isVisibleTo(w)
    # LOS SLIDERS Y LAS ETIQUETAS DICEN DONDE ESTÁ EL ROBOT
    assert [s.value() for s in w.sliders_servo] == [round(v) for v in w.servos]


def test_gs10_no_hay_racha_de_avisos_con_una_rueda_que_gira_rapido(w, monkeypatch):
    """La rueda o la flecha sobre un slider no es un arrastre: son muchos eventos seguidos en pocos milisegundos."""
    se_nego_una_vez(w, monkeypatch)
    monkeypatch.undo()
    for _ in range(20):
        w.SLIDER_2.setValue(w.SLIDER_2.value() + 1)
        avanzar(w, 0.005)
    assert len(del_registro(w, nivel="aviso")) <= 1, "una racha de avisos por una rueda que gira"


def test_gs10_reiniciar_no_mueve_el_robot_ni_reenvia_la_orden_negada(w, monkeypatch):
    donde = se_nego_una_vez(w, monkeypatch)
    monkeypatch.undo()
    salio = len(w.x.enviados)
    salidas_de_la_placa = len(w.enlace._enlace.dispositivo.historial)
    clic(w.BTN_REINICIAR_GUARDIA)
    avanzar(w, 3.0)                                            # AUNQUE PASE EL TIEMPO, NADA SALE
    assert not w.enlace.detenida
    assert w.FRANJA_GUARDIA.isHidden() and not w.FRANJA_GUARDIA.isVisibleTo(w)
    assert len(w.x.enviados) == salio and len(w.enlace._enlace.dispositivo.historial) == salidas_de_la_placa
    assert tuple(w.servos) == donde and not w.cola and not w.en_movimiento
    eventos = del_registro(w)
    assert eventos[-1].nivel != "error" and "reinici" in eventos[-1].mensaje.lower()
    assert len(del_registro(w, nivel="error")) == 1, "Reiniciar no es otro error"
    # YA REINICIADA, LO LEGÍTIMO VUELVE A SALIR
    arrastrar(w, w.SLIDER_1, w.SLIDER_1.value() - 30)
    correr(w)
    assert len(w.x.enviados) > salio and not w.enlace.detenida


def test_gs10_si_el_defecto_sigue_reiniciar_no_lo_esconde(w, monkeypatch):
    """Reiniciar no arregla nada: si lo que pide el slider sigue siendo malo, se vuelve a negar y se vuelve a detener."""
    se_nego_una_vez(w, monkeypatch)
    clic(w.BTN_REINICIAR_GUARDIA)
    assert w.FRANJA_GUARDIA.isHidden()
    arrastrar(w, w.SLIDER_1, w.SLIDER_1.value() + 40, eventos=1)
    assert w.enlace.detenida and w.FRANJA_GUARDIA.isVisibleTo(w)
    assert len(del_registro(w, nivel="error")) == 2


def test_gs10_detenida_la_vista_previa_sigue_funcionando_porque_no_manda_nada(w, monkeypatch):
    se_nego_una_vez(w, monkeypatch)
    monkeypatch.undo()
    _elegir_programa(w, "demo-cosecha.yaml")
    clic(w.BTN_EDITAR_PUNTOS)
    salio = len(w.x.enviados)
    clic(w.BTN_VISTA_PREVIA)
    correr(w)
    assert len(w.x.enviados) == salio and not w.cola
    assert w.FRANJA_GUARDIA.isVisibleTo(w)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GS6: LO QUE HOY FUNCIONA, SIGUE FUNCIONANDO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@pytest.mark.parametrize("escenario", list(ESCENARIOS))
def test_gs06_la_guardia_no_estorba_a_lo_que_hoy_funciona(w, escenario):
    salio = len(w.x.enviados)
    ESCENARIOS[escenario](w)
    assert w.x.negadas == [], [(v.clave, v.motivo) for _h, v, _s in w.x.negadas]
    assert not w.enlace.detenida and w.FRANJA_GUARDIA.isHidden()
    assert del_registro(w) == []
    assert len(w.x.enviados) > salio + 3, "control: el escenario no movió el robot"
    assert not w.cola and not w.en_movimiento


def test_gs06_calcular_llega_al_punto_y_regresa_a_home_por_la_guardia(w):
    home = tuple(w.servos)
    _escenario_calcular(w)
    assert all(abs(a - b) < 1.0 for a, b in zip(w.servos, home)), (w.servos, home)
    # LO QUE LA PLACA RECIBIÓ ES LO QUE LA GUARDIA DEJÓ SALIR: ni una orden de más ni de menos (la pose de arranque va aparte)
    assert len(a_la_placa(w)) > 30
    assert len(w.enlace._enlace.dispositivo.historial) == len(a_la_placa(w)) + 1


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GS6 / #7: CON EL TEMPORIZADOR Y EL RELOJ REALES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@pytest.fixture
def ventana_real_en_tiempo_real(tmp_path):
    """La ventana REAL con su temporizador y el reloj de verdad (time.monotonic): nada falso."""
    carpeta = tmp_path / "programas"
    carpeta.mkdir()
    ajustes = QtCore.QSettings(str(tmp_path / "ajustes.ini"), QtCore.QSettings.IniFormat)
    with sin_dialogos_de_verdad():
        v = VentanaPrincipal(ROBOT, ajustes=ajustes, carpeta_programas=str(carpeta), registro=reg.Registro(carpeta=None))
        v.show()
        yield v
        reventones = list(ventana_excepciones())
    v.close()
    pasar()
    assert not reventones


def ventana_excepciones():
    from helpers_ventana import EXCEPCIONES
    return EXCEPCIONES


def _esperar(v, tope_s=8.0):
    fin = time.monotonic() + tope_s
    while v.temporizador_movimiento.isActive() and time.monotonic() < fin:
        APP.processEvents()
        time.sleep(0.001)
    assert not v.temporizador_movimiento.isActive(), "el movimiento no terminó en el tiempo de la prueba"


def _tocar(v, slider, destino, eventos, entre_s):
    """Un arrastre en tiempo real: un movimiento del ratón cada `entre_s` segundos de reloj de verdad."""
    salida = slider.value()
    slider.setSliderDown(True)
    for i in range(1, eventos + 1):
        slider.setSliderPosition(int(round(salida + (destino - salida) * i / eventos)))
        fin = time.monotonic() + entre_s
        while time.monotonic() < fin:
            APP.processEvents()
            time.sleep(0.0005)
    slider.setSliderDown(False)
    APP.processEvents()


@pytest.mark.parametrize("modo", MODOS_EN_CAMINO)
def test_gs06_el_jog_de_la_punta_en_tiempo_real_nunca_topa_con_la_guardia(ventana_real_en_tiempo_real, modo):
    """La pregunta del riesgo: `ejecutar` dispara el primer paso 1 ms después de encolarlo sin mirar cuánto hace del último
    envío. Con el reloj y el QTimer de verdad, en los tres modos, con un segundo slider movido con la punta en camino."""
    v = ventana_real_en_tiempo_real
    horas = []
    original = v.enlace._enlace.enviar
    v.enlace._enlace.enviar = lambda *s: (horas.append(time.monotonic()), original(*s))[1]
    v.selector_en_camino.setCurrentIndex(v.selector_en_camino.findData(modo))
    v.PESTANAS_MOVIMIENTO.setCurrentIndex(ventana_real.PESTANA_XYZ)
    time.sleep(0.2)
    for eje, delta in ((v.SLIDER_X, 60), (v.SLIDER_Y, -40), (v.SLIDER_Z, -30), (v.SLIDER_X, -60)):
        _tocar(v, eje, eje.value() + delta, eventos=8, entre_s=0.016)
        _esperar_un_rato(v, 0.15)                              # LA PUNTA SIGUE EN CAMINO CUANDO LLEGA EL SIGUIENTE
    _esperar(v)
    assert len(horas) > 20, "control: casi no se movió nada"
    assert not v.enlace.detenida, v.enlace.violacion
    assert [e for e in v.registro.eventos if e.origen == "Guardia"] == []


def _esperar_un_rato(v, segundos):
    fin = time.monotonic() + segundos
    while time.monotonic() < fin:
        APP.processEvents()
        time.sleep(0.001)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GS12: NO HAY FORMA DE APAGARLA NI DE RODEARLA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _llamadas_a_enviar():
    """[(archivo relativo, receptor de la llamada)] de cada `X.enviar(...)` en el código de la app."""
    encontradas = []
    for ruta in sorted(glob.glob(os.path.join(CARPETA_REPO, "cosechador", "**", "*.py"), recursive=True)):
        arbol = ast.parse(open(ruta, encoding="utf-8").read())
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Attribute) and nodo.func.attr == "enviar":
                encontradas.append((os.path.relpath(ruta, CARPETA_REPO), ast.unparse(nodo.func.value)))
    return encontradas


def test_gs12_en_la_app_solo_la_guardia_le_habla_al_enlace_crudo():
    """Dos lugares llaman `.enviar(`: la guardia (a su enlace crudo, `self._enlace`) y la ventana (a `self.enlace`, que es la
    guardia). Un tercer lugar, o la ventana llamando a otra cosa, es un camino que rodea la guardia."""
    permitidas = {("cosechador/hardware/guardia.py", "self._enlace"), ("cosechador/ui/ventana.py", "self.enlace")}
    assert set(_llamadas_a_enviar()) == permitidas, _llamadas_a_enviar()


def test_gs12_solo_la_ventana_fuera_del_paquete_hardware_abre_o_construye_un_enlace():
    """Quien importe `abrir_enlace` o las clases del enlace fuera de hardware/ puede hablarle a la placa sin guardia."""
    nombres = ("abrir_enlace", "Enlace", "ESP32Virtual", "ESP32Serial", "dispositivo")
    fuera = {}
    for ruta in sorted(glob.glob(os.path.join(CARPETA_REPO, "cosechador", "**", "*.py"), recursive=True)):
        rel = os.path.relpath(ruta, CARPETA_REPO)
        if rel.startswith("cosechador/hardware/"):
            continue
        arbol = ast.parse(open(ruta, encoding="utf-8").read())
        usados = {n.id for n in ast.walk(arbol) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(arbol)
                                                                              if isinstance(n, ast.Attribute)}
        importados = {a.name for n in ast.walk(arbol) if isinstance(n, ast.ImportFrom) and n.module and
                      n.module.startswith("cosechador.hardware") for a in n.names}
        tocan = (usados | importados) & set(nombres) - {"dispositivo"}
        if tocan:
            fuera[rel] = sorted(tocan)
    assert fuera == {"cosechador/ui/ventana.py": ["abrir_enlace"]}, fuera


def test_gs12_la_ventana_no_acepta_ningun_parametro_para_quitarla():
    """Si alguien agrega un parámetro a la ventana, esta prueba lo obliga a decidir a conciencia (y a revisar que no sea un
    interruptor de la guardia)."""
    parametros = list(inspect.signature(VentanaPrincipal.__init__).parameters)
    assert parametros == ["self", "robot", "enlace", "fuente_camara", "ajustes", "carpeta_programas", "registro", "reloj"]
    with pytest.raises(TypeError):
        VentanaPrincipal(ROBOT, guardia=False)
    with pytest.raises(TypeError):
        VentanaPrincipal(ROBOT, sin_guardia=True)


def test_gs12_la_ventana_solo_lee_del_entorno_la_camara():
    fuente = open(os.path.join(CARPETA_REPO, "cosechador", "ui", "ventana.py"), encoding="utf-8").read()
    arbol = ast.parse(fuente)
    claves = {ast.unparse(n.args[0]) for n in ast.walk(arbol) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and ast.unparse(n.func.value) == "os.environ" and n.func.attr == "get" and n.args}
    assert claves == {"'ROBOT_CAMARA'"}, claves
    assert "getenv" not in fuente and "os.environ[" not in fuente
    assert "environ" not in open(os.path.join(CARPETA_REPO, "cosechador", "hardware", "guardia.py"), encoding="utf-8").read()


def test_gs12_ninguna_variable_de_entorno_ni_ajuste_la_quita(tmp_path, monkeypatch):
    for nombre in ("ROBOT_SIN_GUARDIA", "GUARDIA", "GUARDIA_OFF", "COSECHADOR_GUARDIA", "ROBOT_GUARDIA", "SIN_GUARDIA"):
        monkeypatch.setenv(nombre, "0")
        monkeypatch.setenv(nombre + "_DESACTIVADA", "1")
    carpeta = tmp_path / "programas"
    carpeta.mkdir()
    ajustes = QtCore.QSettings(str(tmp_path / "ajustes.ini"), QtCore.QSettings.IniFormat)
    for clave in ("guardia", "guardia/activa", "guardia/apagada", "guardia/desactivada", "sin_guardia"):
        ajustes.setValue(clave, False)
        ajustes.setValue(clave + "/off", True)
    v = VentanaPrincipal(ROBOT, ajustes=ajustes, carpeta_programas=str(carpeta), registro=reg.Registro(carpeta=None))
    try:
        assert type(v.enlace) is GuardiaDeSalida
        with pytest.raises(OrdenNegada):
            v.enlace.enviar(*(j.servo.maximo + 50 for j in ROBOT.articulaciones))
    finally:
        v.close()
        pasar()


class _DispositivoFalso:
    def leer_respuestas(self):
        return []


class _EnlaceYaHecho(EnlaceFalso):
    nombre = "enlace ya hecho"

    def __init__(self):
        super().__init__()
        self.dispositivo = _DispositivoFalso()


def test_gs12_aunque_le_pasen_un_enlace_ya_hecho_lo_envuelve(tmp_path):
    carpeta = tmp_path / "programas"
    carpeta.mkdir()
    ajustes = QtCore.QSettings(str(tmp_path / "ajustes.ini"), QtCore.QSettings.IniFormat)
    ya_hecho = _EnlaceYaHecho()
    v = VentanaPrincipal(ROBOT, enlace=ya_hecho, ajustes=ajustes, carpeta_programas=str(carpeta),
                         registro=reg.Registro(carpeta=None))
    try:
        assert type(v.enlace) is GuardiaDeSalida and v.enlace._enlace is ya_hecho
        salio = len(ya_hecho.salidas)
        assert salio == 1, "la pose de arranque sale por el enlace que le dieron, pasando por la guardia"
        with pytest.raises(OrdenNegada):
            v.enlace.enviar(*(j.servo.maximo + 50 for j in ROBOT.articulaciones))
        assert len(ya_hecho.salidas) == salio, "la orden negada llegó al enlace que le pasaron"
    finally:
        v.close()
        pasar()


def test_gs12_una_orden_negada_nunca_llega_al_enlace_crudo(w, monkeypatch):
    crudo = w.enlace._enlace
    llegadas = len(crudo.dispositivo.historial)
    se_nego_una_vez(w, monkeypatch)
    monkeypatch.undo()
    arrastrar(w, w.SLIDER_2, w.SLIDER_2.value() + 30)
    avanzar(w, 2.0)
    assert len(crudo.dispositivo.historial) == llegadas
