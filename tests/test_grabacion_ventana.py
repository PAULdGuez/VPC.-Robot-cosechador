"""F1, bloque 2: grabación de movimientos, en la ventana (B2.1, B2.3–B2.6). La lógica: test_grabacion.py.

La ventana se instancia como en la app (QT_QPA_PLATFORM=offscreen: no se abre nada a la vista ni toma el
teclado) y los pasos del temporizador se ejecutan a mano, adelantando un reloj falso lo que dura cada paso.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5 import QtCore, QtWidgets

import auditoria_interfaz as au
from cosechador import registro as reg
from cosechador.movimiento import programa as prog
from cosechador.movimiento.grabadora import Grabadora, programa_de_grabacion
from cosechador.robot.cinematica_inversa import salto
from cosechador.ui.ventana import DESCARTAR, TEXTO_DETENER_GRABACION, TEXTO_GRABAR

PASO_S = 0.05  # LO QUE DURA UN PASO A VELOCIDAD 100 % (ms_por_paso DEL ROBOT)


class RelojFalso:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def ventana(robot, tmp_path):
    from cosechador.ui.ventana import VentanaPrincipal
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    carpeta = tmp_path / "programas"
    carpeta.mkdir()
    ajustes = QtCore.QSettings(str(tmp_path / "ajustes.ini"), QtCore.QSettings.IniFormat)
    reloj = RelojFalso()
    # EL RELOJ FALSO ENTRA AL ABRIR: LA GUARDIA DE LA VENTANA LO USA PARA LA POSE DE ARRANQUE, Y DESPUÉS PARA CADA PASO
    w = VentanaPrincipal(robot, ajustes=ajustes, carpeta_programas=str(carpeta), registro=reg.Registro(carpeta=None), reloj=reloj)
    reloj.t += 1.0   # LA POSE DE ARRANQUE SALIÓ HACE RATO: NADIE TOCA UN CONTROL EN EL MILISEGUNDO EN QUE ABRE LA VENTANA
    w.grabadora = Grabadora(reloj=w.reloj)
    # TODO LO QUE DE VERDAD SALE HACIA LA PLACA, CON SU HORA
    w.enviados = []
    enviar = w.enlace.enviar
    w.enlace.enviar = lambda *servos: (w.enviados.append((w.reloj.t, tuple(servos))), enviar(*servos))[1]
    # LOS DIÁLOGOS SE CONTESTAN DESDE LA PRUEBA
    w.nombres, w.preguntas = [], []
    w._pedir_nombre = lambda sugerido, tipo="grabacion": w.nombres.pop(0) if w.nombres else None
    w._confirmar_sobrescribir = lambda archivo, tipo="grabacion": (w.preguntas.append(archivo), w.respuesta_sobrescribir)[1]
    w.respuesta_sobrescribir = False
    yield w
    w.close()
    app.processEvents()


def correr(w):
    """Ejecuta lo que haya en la cola, paso por paso, adelantando el reloj lo que dura cada paso."""
    w.temporizador_movimiento.stop()
    while w.cola:
        w._siguiente_paso()
        w.reloj.t += w.temporizador_movimiento.interval() / 1000
    w._siguiente_paso()  # COLA VACÍA: EL MOVIMIENTO TERMINA (Y SE REACTIVAN LOS CONTROLES)


def mover_con_las_dos_pestanas(w):
    """Mueve con la pestaña Punta X Y Z, se queda quieto 2 s y mueve con la pestaña Servos."""
    w.SLIDER_Z.setValue(w.SLIDER_Z.value() - 120)
    correr(w)
    por_xyz = len(w.enviados)
    w.reloj.t += 2.0
    w.SLIDER_1.setValue(w.SLIDER_1.value() + 40)
    correr(w)
    return por_xyz, len(w.enviados) - por_xyz


def otro_programa(w):
    """Un programa cualquiera en la carpeta, para que Reproducir tenga qué reproducir."""
    inicio = tuple(float(s) for s in w.servos)  # DESPUÉS DE MOVER, LOS SERVOS LLEGAN COMO NÚMEROS DE NUMPY
    prog.guardar(programa_de_grabacion((prog.Muestra(0.0, inicio), prog.Muestra(1.0, (100.0, *inicio[1:]))),
                                       "otra", w.robot), os.path.join(w.carpeta_programas, "otra.yaml"))
    w.recargar_programas()


def eventos(w, nivel=None):
    return [e for e in w.registro.eventos if nivel is None or e.nivel == nivel]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% B2.1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_b2_1_se_graba_cada_pose_enviada_a_los_servos_con_su_tiempo(ventana):
    w = ventana
    w.alternar_grabacion()
    por_xyz, por_servos = mover_con_las_dos_pestanas(w)
    assert por_xyz > 0 and por_servos > 0          # LAS DOS PESTAÑAS MOVIERON EL ROBOT
    w.nombres = ["prueba b2"]
    w.alternar_grabacion()

    g = prog.cargar(os.path.join(w.carpeta_programas, "prueba-b2.yaml"), w.robot)
    assert g.tipo == "grabacion" and g.nombre == "prueba b2"
    # SIN LAS MUESTRAS DE «SE QUEDÓ QUIETO» (MISMA POSE QUE LA ANTERIOR), QUEDA EXACTAMENTE LO ENVIADO, EN ORDEN
    movimientos = [m for anterior, m in zip(g.muestras, g.muestras[1:]) if salto(anterior.servos, m.servos) > 1e-9]
    assert len(movimientos) == len(w.enviados)
    t0 = w.enviados[0][0]
    for (t_envio, servos), m in zip(w.enviados, movimientos):
        assert m.servos == pytest.approx(servos, abs=1e-5)
        assert m.t == pytest.approx(t_envio - t0 + PASO_S * 100 / w.velocidad, abs=1e-3)  # LLEGA AL TERMINAR SU PASO
    assert g.muestras[0].servos == pytest.approx(w.enviados[0][1], abs=w.robot.movimiento.salto_max_por_paso)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% B2.3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_b2_3_al_detener_guarda_aparece_en_el_selector_y_queda_elegida(ventana):
    w = ventana
    w.alternar_grabacion()
    mover_con_las_dos_pestanas(w)
    w.nombres = ["Mi grabación"]
    w.alternar_grabacion()
    assert os.path.exists(os.path.join(w.carpeta_programas, "mi-grabacion.yaml"))
    assert w.SELECTOR_PROGRAMA.currentData() == "mi-grabacion.yaml"
    assert w.ajustes.value("programa/ultimo") == "mi-grabacion.yaml"   # SE RECUERDA AL REABRIR
    assert w.BTN_GRABAR.text() == "⏺ Grabar" and w.BTN_REPRODUCIR.isEnabled()
    assert w.preguntas == []                                          # NO EXISTÍA: NO PREGUNTA


def test_b2_3_si_el_nombre_ya_existe_pregunta_antes_de_sobrescribir(ventana):
    w = ventana
    ruta = os.path.join(w.carpeta_programas, "repetida.yaml")
    for respuesta, nombres, queda in ((False, ["Repetida", "Otra"], "otra.yaml"), (True, ["repetida"], "repetida.yaml")):
        with open(ruta, "w", encoding="utf-8") as f:
            f.write("original\n")
        w.alternar_grabacion()
        mover_con_las_dos_pestanas(w)
        w.preguntas, w.respuesta_sobrescribir, w.nombres = [], respuesta, list(nombres)
        w.alternar_grabacion()
        assert w.preguntas == ["repetida.yaml"]                     # PREGUNTÓ, CON EL NOMBRE DEL ARCHIVO
        if respuesta:
            assert prog.cargar(ruta, w.robot).nombre == "repetida"  # DIJO QUE SÍ: SE REEMPLAZÓ
        else:
            assert open(ruta, encoding="utf-8").read() == "original\n"  # DIJO QUE NO: INTACTO, Y PIDIÓ OTRO NOMBRE
        assert w.SELECTOR_PROGRAMA.currentData() == queda


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% G1, G2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# (25-nov-2026) ANTES, «Cancelar» EN EL NOMBRE TIRABA LA GRABACIÓN: A PAUL LE COSTÓ 170 MUESTRAS. UN «Cancelar» NO DESTRUYE.

def test_g01_cancelar_el_nombre_sigue_grabando_y_no_pierde_nada(ventana):
    w = ventana
    w.alternar_grabacion()
    mover_con_las_dos_pestanas(w)
    antes = w.grabadora.resumen()[0]
    assert antes > 2

    def cancela_tras_medio_minuto(sugerido, tipo="grabacion"):
        w.reloj.t += 30.0                                             # LO QUE TARDÓ CON EL DIÁLOGO ABIERTO
        return None
    w._pedir_nombre = cancela_tras_medio_minuto
    w.alternar_grabacion()                                            # ⏹ DETENER → «Cancelar»
    assert w.grabadora.grabando and w.grabadora.resumen()[0] == antes
    assert w.BTN_GRABAR.text() == TEXTO_DETENER_GRABACION and not w.aviso_grabando.isHidden()
    assert prog.listar(w.carpeta_programas) == []
    assert not any("descartada" in e.mensaje for e in eventos(w))

    w.SLIDER_1.setValue(w.SLIDER_1.value() - 20)                      # SIGUE MOVIENDO: SE SIGUE GRABANDO
    correr(w)
    w._pedir_nombre = lambda sugerido, tipo="grabacion": "dos partes"
    w.alternar_grabacion()
    assert prog.listar(w.carpeta_programas) == ["dos-partes.yaml"]
    muestras = prog.cargar(os.path.join(w.carpeta_programas, "dos-partes.yaml"), w.robot).muestras
    assert len(muestras) > antes
    huecos = [b.t - a.t for a, b in zip(muestras, muestras[1:])]
    assert max(huecos) < 5.0, "el tiempo con el diálogo abierto no es parte de la grabación"


def test_g02_descartar_pregunta_antes_y_dice_cuanto_se_pierde(ventana):
    w = ventana
    w.alternar_grabacion()
    mover_con_las_dos_pestanas(w)
    cuantas, segundos = w.grabadora.resumen()
    preguntas, respuestas = [], [False, True]
    w._pedir_nombre = lambda sugerido, tipo="grabacion": DESCARTAR if len(preguntas) < 2 else None
    w._confirmar = lambda titulo, pregunta, si, no: (preguntas.append(pregunta), respuestas[len(preguntas) - 1])[1]
    w.alternar_grabacion()                                            # «Descartar…» → No → «Descartar…» → Sí
    assert len(preguntas) == 2
    assert str(cuantas) in preguntas[0] and f"{segundos:.1f} s" in preguntas[0]
    assert not w.grabadora.grabando and w.BTN_GRABAR.text() == TEXTO_GRABAR
    assert prog.listar(w.carpeta_programas) == []
    assert any("descartada" in e.mensaje and str(cuantas) in e.mensaje for e in eventos(w, "aviso"))


def test_g02_decir_que_no_a_descartar_no_tira_nada(ventana):
    w = ventana
    w.alternar_grabacion()
    mover_con_las_dos_pestanas(w)
    cuantas = w.grabadora.resumen()[0]
    pedidos = []
    w._pedir_nombre = lambda sugerido, tipo="grabacion": (pedidos.append(1), DESCARTAR if len(pedidos) == 1 else None)[1]
    w._confirmar = lambda titulo, pregunta, si, no: False
    w.alternar_grabacion()                                            # «Descartar…» → No → vuelve al nombre → «Cancelar»
    assert len(pedidos) == 2
    assert w.grabadora.grabando and w.grabadora.resumen()[0] == cuantas


def test_g02_el_dialogo_del_nombre_guarda_sigue_o_descarta(ventana):
    w = ventana
    dialogo, campo, botones = w._dialogo_del_nombre("prueba", "grabacion")
    assert [b.text() for b in botones] == ["Guardar", "Seguir grabando", "Descartar…"]
    assert campo.text() == "prueba"
    botones[2].click()
    assert w._respuesta_del_nombre(dialogo, campo) is DESCARTAR
    dialogo, campo, botones = w._dialogo_del_nombre("prueba", "grabacion")
    botones[1].click()
    assert w._respuesta_del_nombre(dialogo, campo) is None
    dialogo, campo, botones = w._dialogo_del_nombre("prueba", "grabacion")
    campo.setText("otra")
    botones[0].click()
    assert w._respuesta_del_nombre(dialogo, campo) == "otra"
    _, _, de_puntos = w._dialogo_del_nombre("p", "puntos")
    assert [b.text() for b in de_puntos] == ["Guardar", "Cancelar"]  # UN PROGRAMA POR PUNTOS NO TIENE QUÉ DESCARTAR


def test_g02_en_el_dialogo_intro_guarda_y_descartar_queda_aparte(ventana):
    """Lo que destruye no es el botón por omisión ni queda junto a «Guardar»; y el nombre sugerido se lee entero."""
    w, sugerido = ventana, "grabacion-2026-11-25-0131"
    dialogo, campo, (guardar, seguir, descartar) = w._dialogo_del_nombre(sugerido, "grabacion")
    dialogo.show()
    QtWidgets.QApplication.processEvents()
    assert guardar.isDefault() and not any(b.autoDefault() for b in (guardar, seguir, descartar))
    assert descartar.geometry().right() < seguir.geometry().left() < guardar.geometry().left()
    assert seguir.geometry().left() - descartar.geometry().right() > guardar.geometry().left() - seguir.geometry().right()
    assert dialogo.rect().contains(guardar.geometry()) and dialogo.rect().contains(descartar.geometry())
    dialogo.close()
    for tipo, nombre in (("grabacion", sugerido), ("puntos", "programa-2026-11-25-0131")):
        dialogo, campo, _ = w._dialogo_del_nombre(nombre, tipo)
        dialogo.show()
        QtWidgets.QApplication.processEvents()
        assert campo.fontMetrics().horizontalAdvance(nombre) < campo.contentsRect().width() - 8, tipo
        dialogo.close()


def test_g02_de_punta_a_punta_con_los_dialogos_de_verdad(ventana, monkeypatch):
    """Sin simular _pedir_nombre ni _confirmar: se contestan los diálogos reales, botón por botón."""
    w, respuestas = ventana, []
    del w._pedir_nombre                                               # EL DE VERDAD (EL FIXTURE LO HABÍA SUSTITUIDO)

    def contestar(dialogo):
        boton = next(b for b in dialogo.findChildren(QtWidgets.QPushButton) if b.text() == respuestas[0])
        respuestas.pop(0)
        boton.click()
        return dialogo.result()
    monkeypatch.setattr(QtWidgets.QDialog, "exec_", contestar)
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec_", contestar, raising=False)
    w.alternar_grabacion()
    mover_con_las_dos_pestanas(w)
    cuantas = w.grabadora.resumen()[0]

    respuestas[:] = ["Descartar…", "Conservar", "Seguir grabando"]
    w.alternar_grabacion()
    assert respuestas == [] and w.grabadora.grabando and w.grabadora.resumen()[0] == cuantas

    respuestas[:] = ["Descartar…", "Descartar"]
    w.alternar_grabacion()
    assert respuestas == [] and not w.grabadora.grabando and prog.listar(w.carpeta_programas) == []

    w.alternar_grabacion()                                            # OTRA GRABACIÓN, Y ESTA SÍ SE GUARDA
    mover_con_las_dos_pestanas(w)
    respuestas[:] = ["Guardar"]
    w.alternar_grabacion()
    assert respuestas == [] and len(prog.listar(w.carpeta_programas)) == 1


def test_g02_la_pregunta_trae_sus_dos_verbos_y_por_omision_no_destruye(ventana):
    w = ventana
    caja, si, no = w._caja_de_confirmar("Descartar grabación", "¿Descartar la grabación?", "Descartar", "Conservar")
    assert (si.text(), no.text()) == ("Descartar", "Conservar") and set(caja.buttons()) == {si, no}
    assert caja.defaultButton() is no and caja.escapeButton() is no
    si.click()
    assert caja.clickedButton() is si
    caja, si, no = w._caja_de_confirmar("Descartar grabación", "¿Descartar la grabación?", "Descartar", "Conservar")
    no.click()
    assert caja.clickedButton() is no and caja.clickedButton() is not si


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% B2.4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_b2_4_grabar_sin_mover_no_guarda_y_avisa(ventana):
    w = ventana
    w.alternar_grabacion()
    w.reloj.t += 5
    w.nombres = ["no debe usarse"]
    w.alternar_grabacion()
    assert w.nombres == ["no debe usarse"]                            # NI SIQUIERA PIDIÓ NOMBRE
    assert prog.listar(w.carpeta_programas) == []
    assert any("no hubo movimiento" in e.mensaje for e in eventos(w, "aviso"))
    assert w.BTN_GRABAR.text() == "⏺ Grabar"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% B2.5 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_b2_5_mientras_graba_no_se_puede_reproducir_calcular_ni_cosechar(ventana):
    w = ventana
    otro_programa(w)
    bloqueados = (w.BTN_REPRODUCIR, w.CALCULAR, w.COSECHAR, w.SELECTOR_PROGRAMA)
    assert all(c.isEnabled() for c in bloqueados)

    w.alternar_grabacion()
    for momento in ("al empezar", "después de mover con los sliders"):
        assert not any(c.isEnabled() for c in bloqueados), momento
        assert all(s.isEnabled() for s in (*w.sliders_servo, *w.sliders_xyz)), momento
        assert w.BTN_GRABAR.isEnabled(), momento
        mover_con_las_dos_pestanas(w)

    # AUNQUE ALGO LLAME DIRECTO A LAS ACCIONES (SIN EL BOTÓN), NO SE MUEVE NADA
    enviados = len(w.enviados)
    w.TE_X.setText("356"), w.TE_Y.setText("711"), w.TE_Z.setText("1014")
    for accion in (w.reproducir, w.ir_y_volver, w.cosechar):
        accion()
        assert not w.cola, accion.__name__
    assert len(w.enviados) == enviados

    w.nombres = ["b2-5"]
    w.alternar_grabacion()
    assert all(c.isEnabled() for c in bloqueados)                     # AL DETENER, TODO REGRESA


def test_b2_5_no_se_graba_mientras_se_reproduce(ventana):
    w = ventana
    otro_programa(w)
    w.reproducir()
    assert not w.BTN_GRABAR.isEnabled()
    w.alternar_grabacion()
    assert not w.grabadora.grabando
    correr(w)
    assert w.BTN_GRABAR.isEnabled()


def test_b2_8_mientras_graba_el_aviso_se_ve_desde_las_tres_pestanas(ventana):
    """El robot se mueve desde Servos y Punta X Y Z: ahí es donde hay que ver que se está grabando."""
    w = ventana
    w.resize(1280, 760)
    w.show()                                              # OFFSCREEN: NO SE ABRE NADA A LA VISTA
    QtWidgets.QApplication.processEvents()
    aviso, pestanas = w.aviso_grabando, w.PESTANAS_MOVIMIENTO
    assert not aviso.isVisible()

    w.alternar_grabacion()
    for indice in range(pestanas.count()):
        pestanas.setCurrentIndex(indice)
        QtWidgets.QApplication.processEvents()
        donde = pestanas.tabText(indice)
        assert aviso.isVisible(), donde
        # EN COORDENADAS DE LAS PESTAÑAS (EL AVISO COMPARTE LA ESQUINA CON EL SELECTOR «EN CAMINO»)
        zona = QtCore.QRect(aviso.mapTo(pestanas, QtCore.QPoint(0, 0)), aviso.size())
        assert zona.width() >= aviso.sizeHint().width() and zona.height() >= aviso.sizeHint().height(), donde
        assert pestanas.rect().contains(zona) and not zona.intersects(pestanas.tabBar().geometry()), donde
        assert au.auditar(w) == [], donde                 # NI SE ENCIMA, NI SE CORTA, NI PIERDE CONTRASTE
    mover_con_las_dos_pestanas(w)
    assert aviso.isVisible()                              # SIGUE AHÍ DESPUÉS DE MOVER

    w.alternar_grabacion()                                # «Seguir grabando» (G1): EL AVISO NO SE VA
    assert aviso.isVisible()
    w.nombres = ["b2-8"]
    w.alternar_grabacion()                                # AL GUARDAR, SÍ
    assert not aviso.isVisible()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% B2.6 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_b2_6_reproducir_lo_grabado_pasa_por_las_mismas_poses(ventana):
    w = ventana
    w.alternar_grabacion()
    mover_con_las_dos_pestanas(w)
    w.nombres = ["ida y vuelta"]
    w.alternar_grabacion()
    grabadas = [servos for _, servos in w.enviados]

    w.SLIDER_2.setValue(w.SLIDER_2.value() + 30)                     # EL ROBOT QUEDA EN OTRO LADO
    correr(w)
    w.enviados.clear()
    inicio = w.servos
    w.reproducir()
    correr(w)
    reproducidas = [servos for _, servos in w.enviados]

    # CADA POSE GRABADA APARECE, EN ORDEN, ENTRE LAS QUE SE LE MANDARON A LA PLACA AL REPRODUCIR (±0.5°)
    i = 0
    for pose in grabadas:
        while i < len(reproducidas) and salto(reproducidas[i], pose) > 0.5:
            i += 1
        assert i < len(reproducidas), f"no pasó por {pose}"
    assert salto(reproducidas[-1], grabadas[-1]) <= 0.5
    assert salto(inicio, reproducidas[0]) <= w.robot.movimiento.salto_max_por_paso + 1e-6  # ARRANQUE SUAVE


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% REPRODUCIR SIN PROGRAMAS (DEFECTO DEL 24-NOV) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_reproducir_sigue_apagado_sin_programas_despues_de_mover(ventana):
    """Sin programas no hay qué reproducir: el botón no se enciende al terminar un movimiento ni al detenerlo."""
    w = ventana
    assert not w.BTN_REPRODUCIR.isEnabled()               # LA CARPETA DE LA PRUEBA NACE VACÍA
    w.SLIDER_1.setValue(w.SLIDER_1.value() + 20)
    correr(w)
    assert not w.BTN_REPRODUCIR.isEnabled()
    w.SLIDER_1.setValue(w.SLIDER_1.value() + 20)          # TAMPOCO AL DETENER A MEDIO MOVIMIENTO
    w.detener()
    assert not w.BTN_REPRODUCIR.isEnabled()

    otro_programa(w)                                      # CON UN PROGRAMA SÍ SE ENCIENDE, Y SIGUE ASÍ DESPUÉS DE MOVER
    assert w.BTN_REPRODUCIR.isEnabled()
    w.SLIDER_1.setValue(w.SLIDER_1.value() + 20)
    correr(w)
    assert w.BTN_REPRODUCIR.isEnabled()
