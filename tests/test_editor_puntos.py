"""F1, bloque 3: el editor de puntos de un programa (B3.1–B3.11). Cada prueba lleva el número de su criterio.

La ventana es la de la app, con la misma receta de test_grabacion_ventana.py (se reutiliza su fixture): no se
abre nada a la vista, la carpeta de programas y los ajustes son temporales, todo lo que sale hacia la placa se
cuenta en `w.enviados` y los diálogos se contestan desde la prueba.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import yaml
from PyQt5 import QtCore, QtGui, QtWidgets

import auditoria_interfaz as au
from cosechador import registro as reg
from cosechador.movimiento import programa as prog
from cosechador.movimiento.programa import Punto
from cosechador.movimiento.trayectorias import posicion, servos_del_plan, tramo_a_punto, tramo_articular
from cosechador.robot.cinematica_inversa import angulos_servo
from cosechador.ui import tema
from test_grabacion_ventana import correr, eventos, otro_programa, ventana  # noqa: F401 (ventana ES EL FIXTURE)


# LA RECTA DE A A B CRUZA EL EJE DE LA BASE: NO SE PUEDE RECORRER (EL MISMO CASO DE test_trayectorias.py)
A, B = (-1656, 0, 630), (1656, 0, 630)
# DE IDA A VUELTA LA RECTA SÍ SE PUEDE, Y EN ARTICULAR LA PUNTA SE ALEJA MUCHO DE ELLA: SIRVE PARA DISTINGUIR EL TIPO
IDA, VUELTA = (800, 300, 900), (-800, 300, 900)


@pytest.fixture(scope="module")
def home_s(robot):
    return angulos_servo(robot, robot.home)


@pytest.fixture(scope="module")
def poses(robot, home_s):
    """Servos de poses que el robot alcanza (las de test_programas.py) y las cuatro de arriba."""
    xyz = {"arriba": (356, 711, 1150), "corte": (356, 711, 1014), "a": A, "b": B, "ida": IDA, "vuelta": VUELTA}
    return {"home": home_s, **{nombre: angulos_servo(robot, punto) for nombre, punto in xyz.items()}}


def guardar_programa(w, archivo, *puntos):
    """Deja en la carpeta de la ventana un programa por puntos, elegido en el selector."""
    prog.guardar(prog.Programa(archivo[:-5], "puntos", w.robot.nombre, w.robot.gdl, puntos=tuple(puntos)),
                 os.path.join(w.carpeta_programas, archivo))
    w.ajustes.setValue("programa/ultimo", archivo)
    w.recargar_programas()


def abrir(w, archivo, *puntos):
    guardar_programa(w, archivo, *puntos)
    w.BTN_EDITAR_PUNTOS.click()
    assert w.editando


def en_disco(w, archivo):
    return prog.cargar(os.path.join(w.carpeta_programas, archivo), w.robot)


def elegir(w, numero):
    """Deja elegido el punto `numero` (el primero es el 1), o ninguno con None."""
    w.LISTA_PUNTOS.setCurrentRow(-1 if numero is None else numero - 1)
    if numero is None:
        w.LISTA_PUNTOS.clearSelection()


def renglones(w):
    return [w.LISTA_PUNTOS.item(i).text() for i in range(w.LISTA_PUNTOS.count())]


def imposibles(w):
    """Los números de los puntos que la lista marca como imposibles."""
    lista = w.LISTA_PUNTOS
    return [i + 1 for i in range(lista.count()) if lista.item(i).data(QtCore.Qt.UserRole) == "imposible"]


def servos_en_disco(w, archivo):
    return [p.servos for p in en_disco(w, archivo).puntos]


def cerca(*servos):
    """Para comparar con lo guardado: el archivo redondea los servos a 6 decimales."""
    return [pytest.approx(s, abs=1e-5) for s in servos]


def llevar(w, servos):
    """Lleva el robot a esos servos con un movimiento de la app (así quedan como números de numpy)."""
    w.ejecutar(tramo_articular(w.robot, w.servos, servos), bloquear=False, origen="Prueba")
    correr(w)


def contestar(w, respuesta):
    """Las preguntas de sí o no se contestan desde la prueba. Devuelve la lista de lo que se va preguntando."""
    preguntado = []
    w._confirmar = lambda titulo, pregunta: (preguntado.append((titulo, pregunta)), respuesta)[1]
    return preguntado


@pytest.fixture
def reabrir(robot):
    """«Cerrar la app y volver a abrirla»: otra ventana, con la misma carpeta de programas y los mismos ajustes."""
    from cosechador.ui.ventana import VentanaPrincipal
    abiertas = []

    def abrir_otra(w):
        w.ajustes.sync()
        ajustes = QtCore.QSettings(w.ajustes.fileName(), QtCore.QSettings.IniFormat)
        abiertas.append(VentanaPrincipal(robot, ajustes=ajustes, carpeta_programas=w.carpeta_programas,
                                         registro=reg.Registro(carpeta=None)))
        return abiertas[-1]

    yield abrir_otra
    for otra in abiertas:
        otra.close()


def lejos_de_la_recta(robot, servos, desde, hasta):
    """Cuántos mm se aleja la punta, en cada pose, de la recta desde → hasta."""
    a, b = np.asarray(desde, float), np.asarray(hasta, float)
    u = (b - a) / np.linalg.norm(b - a)
    return [float(np.linalg.norm((p - a) - np.dot(p - a, u) * u)) for p in (posicion(robot, s) for s in servos)]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% B3.6: EL ARCHIVO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_b3_6_una_postura_con_numeros_de_numpy_se_guarda_como_numeros_normales(robot, home_s, tmp_path):
    # ASÍ QUEDAN LOS SERVOS DESPUÉS DE CUALQUIER MOVIMIENTO: tramo_articular LOS CALCULA CON NUMPY
    servos = servos_del_plan(tramo_articular(robot, home_s, (100, 60, 60)))[-1]
    assert type(servos[0]) is np.float64                               # SI ESTO CAMBIA, LA PRUEBA YA NO PRUEBA NADA
    por_puntos = prog.Programa("numpy", "puntos", robot.nombre, robot.gdl, puntos=(prog.Punto(servos, "lineal", 55),))
    grabacion = prog.Programa("numpy", "grabacion", robot.nombre, robot.gdl,
                              muestras=(prog.Muestra(0.0, home_s), prog.Muestra(1.0, servos)))
    for programa in (por_puntos, grabacion):
        ruta = str(tmp_path / f"{programa.tipo}.yaml")
        prog.guardar(programa, ruta)
        with open(ruta, encoding="utf-8") as f:
            crudo = yaml.safe_load(f)                                   # safe_load NO ACEPTA ETIQUETAS DE PYTHON NI DE NUMPY
        guardados = crudo.get("puntos") or crudo.get("muestras")
        assert all(type(s) is float for s in guardados[-1]["servos"])
        leido = prog.cargar(ruta, robot)
        assert (leido.puntos or leido.muestras)[-1].servos == pytest.approx(servos, abs=1e-6)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% B3.8: CÓMO SE LLEGA A UN PUNTO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_b3_8_a_un_punto_se_llega_con_su_tipo_y_su_velocidad(robot):
    """La trampa de UR: «ir al punto» iba siempre en articular, aunque el punto fuera lineal."""
    inicio, destino, ms = angulos_servo(robot, IDA), angulos_servo(robot, VUELTA), robot.movimiento.ms_por_paso
    recta, motivo = tramo_a_punto(robot, inicio, prog.Punto(destino, "lineal", 50))
    assert motivo is None
    assert max(lejos_de_la_recta(robot, servos_del_plan(recta), IDA, VUELTA)) < 0.5
    assert posicion(robot, servos_del_plan(recta)[-1]) == pytest.approx(VUELTA, abs=0.5)
    assert {p.ms for p in recta} == {ms * 2}                             # AL 50 %, CADA PASO DURA EL DOBLE

    giro, motivo = tramo_a_punto(robot, inicio, prog.Punto(destino, "articular", 20))
    assert motivo is None and servos_del_plan(giro)[-1] == pytest.approx(destino)
    assert max(lejos_de_la_recta(robot, servos_del_plan(giro), IDA, VUELTA)) > 50   # LA PUNTA NO VA EN RECTA
    assert {p.ms for p in giro} == {ms * 5}                              # AL 20 %, CINCO VECES MÁS

    nada, motivo = tramo_a_punto(robot, angulos_servo(robot, A), prog.Punto(angulos_servo(robot, B), "lineal", 100))
    assert nada is None and motivo.startswith("el camino pasa por") and "no alcanza" in motivo


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% B3.1: ABRIR Y CERRAR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_b3_1_editar_abre_la_columna_puntos_donde_va_vision_y_cerrar_la_quita(ventana, poses):
    w = ventana
    assert w.FONDO_PUNTOS.isHidden() and not w.FONDO_VISION.isHidden() and not w.editando
    guardar_programa(w, "cosecha.yaml", Punto(poses["arriba"]), Punto(poses["corte"], "lineal", 20), Punto(poses["home"]))
    recordado = w.ajustes.value("vision/contraida")

    w.BTN_EDITAR_PUNTOS.click()
    assert not w.FONDO_PUNTOS.isHidden() and w.FONDO_VISION.isHidden() and not w.FRANJA_VISION.isHidden()
    assert w.lay_columnas.indexOf(w.FONDO_PUNTOS) == w.lay_columnas.indexOf(w.FONDO_VISION) + 1   # EN EL LUGAR DE VISIÓN
    assert renglones(w) == ["1   Articular · 100 %", "2   Lineal · 20 %", "3   Articular · 100 %"]
    assert w.NOTA_PUNTOS.text() == "cosecha.yaml · 3 puntos"
    assert w.fila_elegida() is None and w.SUB_PUNTO_ELEGIDO.text() == "NINGÚN PUNTO ELEGIDO"

    w.BTN_CERRAR_PUNTOS.click()
    assert w.FONDO_PUNTOS.isHidden() and not w.FONDO_VISION.isHidden() and w.FRANJA_VISION.isHidden() and not w.editando
    assert w.ajustes.value("vision/contraida") == recordado      # ABRIR Y CERRAR EL EDITOR NO CAMBIA LO QUE SE RECUERDA


def test_b3_1_cerrar_deja_la_vision_como_estaba_tambien_si_estaba_oculta(ventana, poses):
    w = ventana
    guardar_programa(w, "uno.yaml", Punto(poses["home"]))
    w.contraer_vision(True)                                       # LA PERSONA TENÍA LA VISIÓN GUARDADA EN SU FRANJA
    w.BTN_EDITAR_PUNTOS.click()
    w.BTN_CERRAR_PUNTOS.click()
    assert w.FONDO_VISION.isHidden() and not w.FRANJA_VISION.isHidden() and w.ajustes.value("vision/contraida") == "true"
    # LA FRANJA «▸ VISIÓN» CON EL EDITOR ABIERTO: VISIÓN Y PUNTOS OCUPAN EL MISMO LUGAR, ASÍ QUE EL EDITOR SE CIERRA
    w.BTN_EDITAR_PUNTOS.click()
    w.FRANJA_VISION.click()
    assert w.FONDO_PUNTOS.isHidden() and not w.FONDO_VISION.isHidden() and w.FRANJA_VISION.isHidden()


def test_b3_1_con_una_grabacion_elegida_editar_sale_apagado_y_dice_por_que(ventana, poses):
    w = ventana
    otro_programa(w)                                              # «otra.yaml»: UNA GRABACIÓN, Y QUEDA ELEGIDA
    assert w.SELECTOR_PROGRAMA.currentData() == "otra.yaml" and not w.BTN_EDITAR_PUNTOS.isEnabled()
    assert w.BTN_EDITAR_PUNTOS.toolTip() == "Una grabación son muestras por tiempo:\nno se edita por puntos."
    w.abrir_editor()                                              # NI LLAMÁNDOLO SIN EL BOTÓN
    assert w.FONDO_PUNTOS.isHidden()
    # CON UN PROGRAMA POR PUNTOS SE ENCIENDE, CON SU AYUDA DE SIEMPRE
    guardar_programa(w, "puntos.yaml", Punto(poses["home"]))
    assert w.BTN_EDITAR_PUNTOS.isEnabled() and w.BTN_EDITAR_PUNTOS.toolTip() == "Abre el editor de puntos del programa elegido"
    # Y SI CON EL EDITOR ABIERTO SE ELIGE LA GRABACIÓN, EL EDITOR SE CIERRA: NO HAY PUNTOS QUE ENSEÑAR
    w.BTN_EDITAR_PUNTOS.click()
    w.SELECTOR_PROGRAMA.setCurrentIndex(w.SELECTOR_PROGRAMA.findData("otra.yaml"))
    assert w.FONDO_PUNTOS.isHidden() and not w.BTN_EDITAR_PUNTOS.isEnabled()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% B3.2: + PUNTO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_b3_2_mas_punto_guarda_la_postura_actual_despues_del_elegido_o_al_final(ventana, poses):
    w = ventana
    abrir(w, "p.yaml", Punto(poses["arriba"]), Punto(poses["corte"], "lineal", 20))
    w.SLIDER_1.setValue(w.SLIDER_1.value() + 40)                  # LA PERSONA MUEVE EL ROBOT CON LA PESTAÑA SERVOS
    correr(w)
    primera = tuple(w.servos)
    assert type(primera[0]) is np.float64                         # LA TRAMPA: ASÍ QUEDA LA POSE DESPUÉS DE MOVER
    w.BTN_PUNTO_NUEVO.click()                                     # NINGUNO ELEGIDO: VA AL FINAL
    assert servos_en_disco(w, "p.yaml") == cerca(poses["arriba"], poses["corte"], primera)
    nuevo = en_disco(w, "p.yaml").puntos[2]
    assert (nuevo.movimiento, nuevo.velocidad) == ("articular", 100)
    assert w.fila_elegida() == 2 and renglones(w)[2] == "3   Articular · 100 %"       # Y QUEDA ELEGIDO
    with open(os.path.join(w.carpeta_programas, "p.yaml"), encoding="utf-8") as f:
        assert all(type(s) is float for s in yaml.safe_load(f)["puntos"][2]["servos"])  # NÚMEROS NORMALES EN EL ARCHIVO

    llevar(w, poses["ida"])
    elegir(w, 1)
    w.BTN_PUNTO_NUEVO.click()                                     # CON EL 1 ELEGIDO: VA DESPUÉS DEL 1
    assert servos_en_disco(w, "p.yaml") == cerca(poses["arriba"], poses["ida"], poses["corte"], primera)
    assert w.fila_elegida() == 1 and w.NOTA_PUNTOS.text() == "p.yaml · 4 puntos"
    assert w.INFO_PROGRAMA.text() == "Por puntos · 4 puntos"      # LA PESTAÑA PROGRAMA TAMBIÉN SE ENTERA


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% B3.3: RE-ENSEÑAR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_b3_3_re_ensenar_cambia_el_punto_elegido_por_la_postura_actual_despues_de_confirmar(ventana, poses):
    w = ventana
    abrir(w, "p.yaml", Punto(poses["arriba"]), Punto(poses["corte"], "lineal", 20), Punto(poses["home"]))
    llevar(w, poses["ida"])
    assert not w.BTN_REENSENAR.isEnabled()                        # SIN PUNTO ELEGIDO NO HAY QUÉ RE-ENSEÑAR
    elegir(w, 2)
    preguntado = contestar(w, False)
    w.BTN_REENSENAR.click()
    assert preguntado == [("Re-enseñar", "¿Reemplazar el punto 2 con la pose actual del robot?")]
    assert servos_en_disco(w, "p.yaml") == cerca(poses["arriba"], poses["corte"], poses["home"])   # DIJO QUE NO

    contestar(w, True)
    w.BTN_REENSENAR.click()
    assert servos_en_disco(w, "p.yaml") == cerca(poses["arriba"], poses["ida"], poses["home"])
    cambiado = en_disco(w, "p.yaml").puntos[1]
    assert (cambiado.movimiento, cambiado.velocidad) == ("lineal", 20)   # CONSERVA SU TIPO Y SU VELOCIDAD
    assert w.fila_elegida() == 1 and len(renglones(w)) == 3


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% B3.4: QUITAR, SUBIR Y BAJAR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_b3_4_quitar_actua_sobre_el_punto_elegido(ventana, poses):
    w = ventana
    abrir(w, "p.yaml", Punto(poses["arriba"]), Punto(poses["corte"], "lineal", 20), Punto(poses["home"]))
    assert not w.BTN_QUITAR_PUNTO.isEnabled()                     # NINGUNO ELEGIDO
    w.quitar_punto()
    assert len(en_disco(w, "p.yaml").puntos) == 3
    elegir(w, 2)
    w.BTN_QUITAR_PUNTO.click()
    assert servos_en_disco(w, "p.yaml") == cerca(poses["arriba"], poses["home"])
    assert w.fila_elegida() == 1 and renglones(w) == ["1   Articular · 100 %", "2   Articular · 100 %"]
    w.BTN_QUITAR_PUNTO.click()                                    # EL ÚLTIMO DE LA LISTA: QUEDA ELEGIDO EL DE ARRIBA
    assert servos_en_disco(w, "p.yaml") == cerca(poses["arriba"]) and w.fila_elegida() == 0
    # UN PROGRAMA NECESITA AL MENOS UN PUNTO (ASÍ LO PIDE EL FORMATO): EL ÚNICO QUE QUEDA NO SE QUITA
    assert not w.BTN_QUITAR_PUNTO.isEnabled() and w.BTN_QUITAR_PUNTO.toolTip() == "Un programa necesita al menos un punto"
    w.quitar_punto()
    assert len(en_disco(w, "p.yaml").puntos) == 1


def test_b3_4_subir_y_bajar_mueven_el_punto_elegido(ventana, poses):
    w = ventana
    abrir(w, "p.yaml", Punto(poses["arriba"]), Punto(poses["ida"], "lineal", 20), Punto(poses["home"], "articular", 50))
    assert not w.BTN_SUBIR_PUNTO.isEnabled() and not w.BTN_BAJAR_PUNTO.isEnabled()    # NINGUNO ELEGIDO
    elegir(w, 2)
    w.BTN_SUBIR_PUNTO.click()
    assert servos_en_disco(w, "p.yaml") == cerca(poses["ida"], poses["arriba"], poses["home"])
    assert w.fila_elegida() == 0 and not w.BTN_SUBIR_PUNTO.isEnabled()                # LA SELECCIÓN SIGUE AL PUNTO
    w.BTN_BAJAR_PUNTO.click()
    w.BTN_BAJAR_PUNTO.click()
    assert servos_en_disco(w, "p.yaml") == cerca(poses["arriba"], poses["home"], poses["ida"])
    assert w.fila_elegida() == 2 and not w.BTN_BAJAR_PUNTO.isEnabled()
    assert renglones(w) == ["1   Articular · 100 %", "2   Articular · 50 %", "3   Lineal · 20 %"]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% B3.5: TIPO Y VELOCIDAD DEL PUNTO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_b3_5_el_tipo_del_punto_elegido_se_cambia_en_su_fila(ventana, poses):
    w = ventana
    abrir(w, "p.yaml", Punto(poses["arriba"]), Punto(poses["corte"], "lineal", 20))
    tipo = w.SELECTOR_TIPO_PUNTO
    assert [tipo.itemText(i) for i in range(tipo.count())] == ["Articular", "Lineal"]
    assert not tipo.isEnabled() and tipo.currentIndex() == -1     # NINGÚN PUNTO ELEGIDO: LA FILA ESTÁ APAGADA
    elegir(w, 2)
    assert tipo.isEnabled() and tipo.currentData() == "lineal" and w.SUB_PUNTO_ELEGIDO.text() == "PUNTO 2"
    tipo.setCurrentIndex(tipo.findData("articular"))
    assert [(p.movimiento, p.velocidad) for p in en_disco(w, "p.yaml").puntos] == [("articular", 100), ("articular", 20)]
    assert renglones(w)[1] == "2   Articular · 20 %" and w.fila_elegida() == 1
    assert servos_en_disco(w, "p.yaml") == cerca(poses["arriba"], poses["corte"])     # LA POSE NO SE TOCA
    elegir(w, 1)                                                  # MIRAR OTRO PUNTO NO CAMBIA NINGUNO
    tipo.setCurrentIndex(tipo.findData("lineal"))
    assert [p.movimiento for p in en_disco(w, "p.yaml").puntos] == ["lineal", "articular"]


def test_b3_5_la_velocidad_del_punto_elegido_se_cambia_de_5_en_5(ventana, poses):
    w = ventana
    abrir(w, "p.yaml", Punto(poses["arriba"]), Punto(poses["corte"], "lineal", 20))
    assert w.ETIQUETA_VELOCIDAD_PUNTO.text().split() == ["Velocidad", "del", "punto"]   # NO ES LA VELOCIDAD GENERAL
    velocidad, general = w.SLIDER_VELOCIDAD_PUNTO, w.velocidad
    assert not velocidad.isEnabled() and not w.VS_VELOCIDAD_PUNTO.isVisibleTo(w)
    elegir(w, 2)
    assert velocidad.isEnabled() and w.VS_VELOCIDAD_PUNTO.text() == "20 %"

    def guardada():
        return en_disco(w, "p.yaml").puntos[1].velocidad

    velocidad.triggerAction(QtWidgets.QAbstractSlider.SliderSingleStepAdd)             # UN PASO DEL SLIDER
    assert guardada() == 25 and type(guardada()) is int
    assert w.VS_VELOCIDAD_PUNTO.text() == "25 %" and renglones(w)[1] == "2   Lineal · 25 %" and w.fila_elegida() == 1
    velocidad.setSliderDown(True)                                 # ARRASTRANDO: EL VALOR SE VE, PERO TODAVÍA NO SE GUARDA
    velocidad.setSliderPosition(11)
    assert w.VS_VELOCIDAD_PUNTO.text() == "55 %" and guardada() == 25
    velocidad.setSliderDown(False)                                # AL SOLTAR
    assert guardada() == 55 and renglones(w)[1] == "2   Lineal · 55 %"
    # DE UN EXTREMO AL OTRO: SIEMPRE MÚLTIPLOS DE 5, DENTRO DE LO QUE ACEPTA EL FORMATO
    posibles = []
    for posicion_slider in range(velocidad.minimum(), velocidad.maximum() + 1):
        velocidad.setValue(posicion_slider)
        posibles.append(guardada())
    assert posibles == list(range(prog.VELOCIDAD_MIN, prog.VELOCIDAD_MAX + 1, 5))
    assert en_disco(w, "p.yaml").puntos[0].velocidad == 100 and w.velocidad == general  # NI EL OTRO PUNTO NI LA GENERAL


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% B3.6: SE GUARDA SOLO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_b3_6_todo_cambio_se_guarda_solo_y_sigue_ahi_al_reabrir_la_app(ventana, poses, reabrir):
    w = ventana
    abrir(w, "p.yaml", Punto(poses["arriba"]), Punto(poses["corte"], "lineal", 20))
    assert not [b.text() for b in w.FONDO_PUNTOS.findChildren(QtWidgets.QPushButton) if "guardar" in b.text().lower()]
    llevar(w, poses["ida"])
    w.BTN_PUNTO_NUEVO.click()                                     # + PUNTO (QUEDA ELEGIDO EL 3)
    w.SELECTOR_TIPO_PUNTO.setCurrentIndex(w.SELECTOR_TIPO_PUNTO.findData("lineal"))
    w.SLIDER_VELOCIDAD_PUNTO.setValue(7)
    w.BTN_SUBIR_PUNTO.click()
    esperado = ["1   Articular · 100 %", "2   Lineal · 35 %", "3   Lineal · 20 %"]
    assert renglones(w) == esperado

    otra = reabrir(w)                                             # NADIE PULSÓ «GUARDAR»: NO EXISTE
    assert otra.SELECTOR_PROGRAMA.currentData() == "p.yaml" and otra.INFO_PROGRAMA.text() == "Por puntos · 3 puntos"
    otra.BTN_EDITAR_PUNTOS.click()
    assert renglones(otra) == esperado
    assert [p.servos for p in otra.editado.puntos] == cerca(poses["arriba"], poses["ida"], poses["corte"])


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% B3.7: EL PUNTO IMPOSIBLE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_b3_7_un_punto_imposible_sale_en_rojo_con_su_motivo_escrito_debajo(ventana, poses):
    w = ventana
    abrir(w, "p.yaml", Punto(poses["a"]), Punto(poses["b"], "lineal", 40), Punto(poses["home"]))
    assert imposibles(w) == [2]
    malo, bueno = w.LISTA_PUNTOS.item(1), w.LISTA_PUNTOS.item(0)
    assert malo.text() == "2   Lineal · 40 %   ⚠\nEl camino pasa por (-87, 0, 630), una zona que el robot no alcanza"
    assert malo.foreground().color().name() == tema.TOKENS["rojo"] != bueno.foreground().color().name()
    assert "Punto 2 (lineal): el camino pasa por (-87, 0, 630)" in malo.toolTip() and bueno.toolTip() == ""
    assert bueno.text() == "1   Articular · 100 %"
    # ES LO MISMO QUE DICE «REPRODUCIR» CUANDO SE NIEGA A ARRANCAR: LO DECIDE UN SOLO LUGAR
    w.BTN_REPRODUCIR.click()
    assert eventos(w, "aviso")[-1].mensaje == ("No arranca: Punto 2 (lineal): el camino pasa por (-87, 0, 630), "
                                                "una zona que el robot no alcanza")


def test_b3_7_el_punto_imposible_sigue_en_rojo_cuando_esta_elegido(ventana, poses):
    """La hoja del tema pone en negro todo renglón elegido: el rojo se perdía justo al elegir el punto para arreglarlo."""
    w = ventana
    abrir(w, "p.yaml", Punto(poses["a"]), Punto(poses["b"], "lineal", 40), Punto(poses["home"], "lineal", 40))
    assert imposibles(w) == [2]
    w.resize(1280, 720)
    w.show()                                                      # OFFSCREEN: NO SE ABRE NADA A LA VISTA

    def tinta_roja(numero):
        QtWidgets.QApplication.processEvents()
        lista = w.LISTA_PUNTOS
        imagen = lista.viewport().grab(lista.visualItemRect(lista.item(numero - 1))).toImage()
        colores = (imagen.pixelColor(x, y) for x in range(imagen.width()) for y in range(imagen.height()))
        return sum(1 for c in colores if c.red() - max(c.green(), c.blue()) > 80)

    sin_elegir = tinta_roja(2)
    assert sin_elegir > 100 and tinta_roja(3) == 0                # EL 3 SÍ SE PUEDE: NI UN PIXEL ROJO
    elegir(w, 2)
    assert tinta_roja(2) == pytest.approx(sin_elegir, rel=0.2)    # LAS MISMAS LETRAS ROJAS, SOBRE EL FONDO DE «ELEGIDO»
    elegir(w, 3)
    assert tinta_roja(3) == 0                                     # Y UN RENGLÓN NORMAL ELEGIDO SIGUE COMO SIEMPRE


def test_b3_7_se_revisa_otra_vez_tras_cada_cambio(ventana, poses):
    w = ventana
    abrir(w, "p.yaml", Punto(poses["a"]), Punto(poses["b"], "lineal", 40), Punto(poses["home"]))
    tipo = w.SELECTOR_TIPO_PUNTO
    assert imposibles(w) == [2]
    elegir(w, 2)
    tipo.setCurrentIndex(tipo.findData("articular"))              # CAMBIAR EL TIPO: EN ARTICULAR NO HAY RECTA QUE RECORRER
    assert imposibles(w) == []
    tipo.setCurrentIndex(tipo.findData("lineal"))
    assert imposibles(w) == [2]
    w.BTN_SUBIR_PUNTO.click()                                     # MOVER: B(lineal), A, home — EL PRIMERO NO TIENE ANTERIOR
    assert imposibles(w) == []
    w.BTN_BAJAR_PUNTO.click()
    assert imposibles(w) == [2]
    elegir(w, 1)
    w.BTN_PUNTO_NUEVO.click()                                     # AGREGAR: A, home, B(lineal), home — DESDE HOME SÍ SE LLEGA
    assert imposibles(w) == [] and len(renglones(w)) == 4
    w.BTN_QUITAR_PUNTO.click()                                    # QUITAR EL QUE SE AGREGÓ: A, B(lineal), home
    assert imposibles(w) == [2] and len(renglones(w)) == 3
    contestar(w, True)
    elegir(w, 1)
    w.BTN_REENSENAR.click()                                       # RE-ENSEÑAR EL 1 CON HOME: home, B(lineal), home
    assert imposibles(w) == []
    llevar(w, poses["a"])
    w.BTN_REENSENAR.click()                                       # Y OTRA VEZ CON A: VUELVE A SER IMPOSIBLE
    assert imposibles(w) == [2]


def test_b3_7_tambien_se_revisa_al_cambiar_el_modo_de_postura(ventana, robot):
    # LA RECTA ENTRE ESTOS DOS PUNTOS CRUZA y = 0, DONDE EL ROBOT TIENE QUE CAMBIAR DE POSTURA (EL CASO DE test_trayectorias.py)
    w = ventana
    postura = w.SELECTOR_POSTURA
    postura.setCurrentIndex(postura.findData("despacio"))
    abrir(w, "p.yaml", Punto(angulos_servo(robot, (-1400, 150, 1300))), Punto(angulos_servo(robot, (-1400, -150, 1300)), "lineal", 100))
    assert imposibles(w) == []
    elegir(w, 2)
    postura.setCurrentIndex(postura.findData("negar"))
    assert imposibles(w) == [2] and w.fila_elegida() == 1         # SE REVISA SOLO, Y EL PUNTO SIGUE ELEGIDO
    assert "cambiar de postura" in w.LISTA_PUNTOS.item(1).text() and "«negar»" in w.LISTA_PUNTOS.item(1).text()
    postura.setCurrentIndex(postura.findData("despacio"))
    assert imposibles(w) == []


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% B3.8: PASO A PASO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_b3_8_paso_adelante_y_paso_atras_llevan_el_robot_al_punto_y_se_detienen(ventana, poses):
    w = ventana
    abrir(w, "p.yaml", Punto(poses["arriba"]), Punto(poses["corte"], "lineal", 20), Punto(poses["ida"]))
    antes = len(w.enviados)
    w.BTN_PASO_ADELANTE.click()                                   # NINGUNO ELEGIDO: VA AL PRIMERO
    assert w.fila_elegida() == 0 and w.en_movimiento              # LA LISTA MARCA A DÓNDE VA DESDE QUE ARRANCA
    correr(w)
    assert w.servos == pytest.approx(poses["arriba"]) and not w.cola and not w.en_movimiento   # LLEGÓ Y SE DETUVO AHÍ
    assert len(w.enviados) > antes                                # EL ROBOT REAL SÍ SE MUEVE (NO ES LA VISTA PREVIA)
    w.BTN_PASO_ADELANTE.click()
    correr(w)
    assert w.fila_elegida() == 1 and posicion(w.robot, w.servos) == pytest.approx((356, 711, 1014), abs=0.5)
    w.BTN_PASO_ADELANTE.click()
    correr(w)
    assert w.fila_elegida() == 2 and w.servos == pytest.approx(poses["ida"])
    assert not w.BTN_PASO_ADELANTE.isEnabled()                    # ES EL ÚLTIMO: NO HAY SIGUIENTE
    w.paso(+1)
    assert not w.cola and w.fila_elegida() == 2
    w.BTN_PASO_ATRAS.click()
    correr(w)
    assert w.fila_elegida() == 1 and posicion(w.robot, w.servos) == pytest.approx((356, 711, 1014), abs=0.5)
    w.BTN_PASO_ATRAS.click()
    correr(w)
    assert w.fila_elegida() == 0 and w.servos == pytest.approx(poses["arriba"])
    assert not w.BTN_PASO_ATRAS.isEnabled() and w.BTN_PASO_ADELANTE.isEnabled()         # ES EL PRIMERO: NO HAY ANTERIOR
    assert len(en_disco(w, "p.yaml").puntos) == 3                 # DAR PASOS NO CAMBIA EL PROGRAMA


def test_b3_8_paso_atras_se_mueve_con_el_tipo_del_punto_al_que_llega(ventana, robot, poses):
    """La trampa de UR: hacia atrás se iba siempre en articular. Aquí IDA es lineal y VUELTA es articular."""
    w = ventana
    abrir(w, "p.yaml", Punto(poses["ida"], "lineal", 50), Punto(poses["vuelta"], "articular", 100))
    llevar(w, poses["vuelta"])
    elegir(w, 2)
    assert w.velocidad == 100                                     # LA GENERAL, AL 100 %: LO QUE SE MIDE ES LA DEL PUNTO
    antes = len(w.enviados)
    w.BTN_PASO_ATRAS.click()                                      # DE VUELTA A IDA: SE LLEGA A UN PUNTO LINEAL
    correr(w)
    horas, recorrido = zip(*w.enviados[antes:])
    assert max(lejos_de_la_recta(robot, recorrido, VUELTA, IDA)) < 0.5                  # EN RECTA
    assert posicion(robot, w.servos) == pytest.approx(IDA, abs=0.5) and w.fila_elegida() == 0
    paso_s = robot.movimiento.ms_por_paso / 1000
    assert {round(b - a, 3) for a, b in zip(horas, horas[1:])} == {round(2 * paso_s, 3)}   # Y A SU 50 %: EL DOBLE POR PASO

    antes = len(w.enviados)
    w.BTN_PASO_ADELANTE.click()                                   # DE IDA A VUELTA: SE LLEGA A UN PUNTO ARTICULAR
    correr(w)
    horas, recorrido = zip(*w.enviados[antes:])
    assert max(lejos_de_la_recta(robot, recorrido, IDA, VUELTA)) > 50                   # LA PUNTA NO VA EN RECTA
    assert w.servos == pytest.approx(poses["vuelta"]) and w.fila_elegida() == 1
    assert {round(b - a, 3) for a, b in zip(horas, horas[1:])} == {round(paso_s, 3)}


def test_b3_8_si_no_se_puede_llegar_el_paso_no_mueve_el_robot_y_dice_por_que(ventana, poses):
    w = ventana
    abrir(w, "p.yaml", Punto(poses["a"]), Punto(poses["b"], "lineal", 40))
    w.BTN_PASO_ADELANTE.click()
    correr(w)                                                     # AL 1 (ARTICULAR) SÍ LLEGA
    antes = len(w.enviados)
    w.BTN_PASO_ADELANTE.click()                                   # AL 2: LA RECTA DESDE EL 1 NO SE PUEDE RECORRER
    assert not w.cola and not w.en_movimiento and len(w.enviados) == antes
    assert w.fila_elegida() == 0                                  # LA LISTA SIGUE DICIENDO DÓNDE ESTÁ EL ROBOT
    assert eventos(w, "aviso")[-1].mensaje == ("No se mueve: Punto 2 (lineal): el camino pasa por (-87, 0, 630), "
                                                "una zona que el robot no alcanza")
