"""El explorador al azar (tests/explorador.py) y sus pruebas: EX1 a EX11.

Cada criterio aprobado tiene su prueba, con el número en el nombre (test_ex01_…, test_ex02_…): si se rompe, la suite se pone
roja y dice cuál.

  EX1  un comando hace cientos de gestos al azar sobre la ventana real, sin pantalla, en todos los modos
  EX2  tras cada gesto revisa: avisos idénticos en racha, rebotes, brincos, pérdidas, excepciones, órdenes negadas, no terminar
  EX3  lo que encuentra queda reproducible (semilla y lista de gestos)
  EX4  misma semilla, mismo recorrido
  EX5  demuestra que caza: con los defectos de anoche sembrados a propósito, encuentra el del rojo y el del brinco
  EX6  un detector de rachas lee el registro real y encuentra la racha de 31 avisos del 25-nov
  EX7  trabaja en carpeta desechable: nunca escribe ni borra en lo de la persona
  EX8  al terminar dice qué controles y modos tocó, cuántas veces, y cuáles nunca
  EX9  reduce cada hallazgo a la lista de gestos más corta que lo reproduce
  EX10 cada hallazgo se queda como prueba fija
  EX11 dos tamaños: corto (siempre, con la suite) y largo (a pedido: EXPLORADOR_LARGO=1)

Los defectos que el explorador encontró en el código de hoy están en HALLAZGOS como fallo esperado ESTRICTO: la prueba falla
(xfail) mientras el defecto siga vivo y se pone ROJA (XPASS estricto) el día que se arregle, para que alguien quite la marca.
"""
import contextlib
import datetime as dt
import functools
import glob
import hashlib
import os
import sys
import tempfile
import warnings
from unittest import mock

import numpy as np
import pytest
from PyQt5 import QtCore

import explorador as ex
import helpers_ventana
from cosechador import registro as reg
from cosechador.movimiento import programa as prog
from cosechador.hardware.guardia import OrdenNegada
from cosechador.movimiento.invariantes import avance_punta_mm
from cosechador.movimiento.trayectorias import Paso
from cosechador.robot import cargar_robot
from cosechador.ui import ventana as ventana_real
from cosechador.ui.slider_alcance import SliderConAlcance
from explorador import Gesto, Hallazgo
from helpers_ventana import (DESCARTAR_NOMBRE, NoTermina, borrar_desechable, carpeta_desechable, cerrar, correr,
                             ventana_aislada)
from rachas import forma_del_aviso, rachas_de_avisos, rachas_del_archivo

CARPETA_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBOT = cargar_robot()
PRUEBA = Gesto("tecla", "PRUEBA", 0.0, 0, 0.0)       # UN GESTO QUE NO EXISTE: LOS EFECTOS DE ABAJO LO ATRAPAN EN _ejecutar
SEMILLAS_A_PROBAR = range(1, 13)                     # DONDE SE BUSCA SI LA SEMILLA CALIBRADA DE UN DEFECTO SEMBRADO YA NO LO ENCUENTRA


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOS DEFECTOS DE ANOCHE, SEMBRADOS A PROPÓSITO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _sembrar_el_rojo():
    """Defectos 1 y 2: el slider no se frena en el rojo: deja entrar donde el planeador se niega y rebota solo."""
    return mock.patch.object(SliderConAlcance, "acotar", lambda self, valor, desde: valor)


def _sembrar_el_brinco():
    """Defecto 4 llevado al extremo: un tramo articular que llega de un solo paso (sin partirlo en pasos de 15°)."""
    return mock.patch.object(ventana_real, "tramo_articular",
                             lambda robot, a, b, ms=None: [Paso(servos=tuple(float(v) for v in b), ms=ms)])


def _sembrar_la_grabacion_tirada():
    """Defecto 3: «Cancelar» en el diálogo del nombre tiraba la grabación sin preguntar."""
    original = ventana_real.VentanaPrincipal.terminar_grabacion

    def terminar_grabacion(self):
        if self.grabadora.muestras is not None:
            guardado = self._guardar_con_nombre("g", "grabacion",
                                                lambda nombre: ventana_real.programa_de_grabacion(self.grabadora.muestras, nombre,
                                                                                                  self.robot))
            if guardado is None:
                self._soltar_grabacion()           # LO VIEJO: TIRAR LO GRABADO SIN PREGUNTAR
                return
        original(self)
    return mock.patch.object(ventana_real.VentanaPrincipal, "terminar_grabacion", terminar_grabacion)


# defecto: (cómo se siembra, clases que cuentan como encontrarlo, lo que debe decir el detalle según la clase, si se toleran
#           los defectos conocidos, semilla calibrada). El tramo que llega de un solo paso lo acusa el brinco de la punta o,
#           antes, la guardia observadora (un salto de más de 15°): son dos redes para el mismo defecto y cualquiera vale.
DEFECTOS_SEMBRADOS = {
    "rojo": (_sembrar_el_rojo, ("rebote", "racha_de_avisos"), {}, False, 5),
    "brinco": (_sembrar_el_brinco, ("brinco", "orden_negada"), {"orden_negada": "salto_excedido"}, False, 5),
    "perdida": (_sembrar_la_grabacion_tirada, ("perdida",), {}, True, 6),
}


@functools.lru_cache(maxsize=None)
def _donde_lo_encuentra(defecto):
    """(semilla, hallazgo con el defecto sembrado, hallazgo de esa misma semilla SIN sembrarlo).

    Una semilla vale si CON el defecto sembrado la corrida corta lo encuentra y SIN sembrarlo no acusa nada de esa clase (si
    no, lo que acusó podría ser otro defecto, real, y no demostraría que el explorador caza el sembrado). Primero se prueba
    la calibrada; si dejó de servir se buscan otras y se AVISA. No puede ser una semilla fija a secas: cada control que se
    le agrega a la ventana cambia qué gesto sale con cada semilla (al traer las vistas, la 2 y la 5 dejaron de cazar)."""
    sembrar, clases, exige, tolerar, calibrada = DEFECTOS_SEMBRADOS[defecto]
    vistas = []
    for semilla in (calibrada, *(s for s in SEMILLAS_A_PROBAR if s != calibrada)):
        with sembrar():
            h = ex.explorar("corto", semilla, tolerar=tolerar).hallazgo
        if h is None or h.clase not in clases or exige.get(h.clase, "") not in str(h):
            vistas.append(f"{semilla}: sembrado dio «{h.clase if h else 'nada'}»")
            continue
        limpio = ex.explorar("corto", semilla, tolerar=tolerar).hallazgo
        if limpio is not None and limpio.clase in clases:
            vistas.append(f"{semilla}: sin sembrar también acusa «{limpio.clase}»")
            continue
        if semilla != calibrada:
            warnings.warn(f"EX5: la semilla calibrada del defecto «{defecto}» ({calibrada}) ya no lo encuentra y ahora lo "
                          f"encuentra la {semilla}: actualiza DEFECTOS_SEMBRADOS para no pagar la búsqueda en cada corrida. "
                          f"Las anteriores: {'; '.join(vistas)}")
        return semilla, h, limpio
    raise AssertionError(f"ninguna semilla de {list(SEMILLAS_A_PROBAR)} encontró el defecto sembrado «{defecto}»: "
                         + "; ".join(vistas))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EX1 y EX11: LA CORRIDA CORTA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@pytest.fixture(scope="module")
def corrida_corta():
    """UNA sola corrida corta compartida por las pruebas que miran su informe (cuesta unos 5 s)."""
    return ex.explorar("corto", ex.SEMILLA)


def test_ex01_un_comando_hace_cientos_de_gestos_en_todos_los_modos(corrida_corta):
    cob = corrida_corta.cobertura
    assert corrida_corta.hallazgo is None, str(corrida_corta.hallazgo)
    assert cob.gestos >= 200, f"se esperaban cientos de gestos y hubo {cob.gestos}"
    assert cob.ventanas >= 2, "cada ejemplo de Hypothesis abre su propia ventana"
    posturas = {p for p, _ in cob.modos}
    caminos = {c for _, c in cob.modos}
    assert posturas == {"despacio", "negar"}, f"posturas tocadas: {posturas}"
    assert caminos == {"reemplazar", "en_cola", "cancelar"}, f"modos «en camino» tocados: {caminos}"
    assert cob.pasos_a_la_placa > 0, "ningún gesto movió el robot"


def test_ex01_sin_pantalla():
    """Nunca una ventana en la pantalla: la plataforma de Qt es offscreen y ninguna ventana queda abierta al terminar."""
    from PyQt5 import QtGui, QtWidgets
    assert QtGui.QGuiApplication.platformName() == "offscreen"
    assert not [w for w in QtWidgets.QApplication.topLevelWidgets() if w.isVisible()], "quedó una ventana visible"


def test_ex01_abrir_la_ventana_y_mover_de_inmediato_no_es_ir_demasiado_rapido():
    """La guardia observadora anota la pose inicial como enviada al abrir. Si el reloj falso no avanzara después, el primer
    paso del primer gesto «saldría» 1 ms más tarde y la guardia lo negaría por ritmo: un hallazgo falso (nadie toca un
    control en el milisegundo en que abre la ventana)."""
    h = ex.reproducir([Gesto("arrastrar", "SLIDER_1", 0.0, 88, 0.0), Gesto("llegar", "", 0.0, 0, 0.0)])
    assert h is None, str(h)


def test_ex01_el_dialogo_del_nombre_solo_contesta_lo_que_el_de_verdad_ofrece():
    """«Descartar…» solo existe al guardar una GRABACIÓN; un programa por puntos trae «Guardar» y «Cancelar». Si el gancho
    de pruebas contestara «descartar» ahí, la ventana recibiría una respuesta que su diálogo no puede dar."""
    carpeta = carpeta_desechable()
    w = ventana_aislada(ROBOT, carpeta)
    try:
        botones = {tipo: [b.text() for b in w._dialogo_del_nombre("x", tipo)[2]] for tipo in ("puntos", "grabacion")}
        assert botones == {"puntos": ["Guardar", "Cancelar"], "grabacion": ["Guardar", "Seguir grabando", "Descartar…"]}
        w.x.nombres.extend([DESCARTAR_NOMBRE, DESCARTAR_NOMBRE])
        assert w._pedir_nombre("x", "puntos") is None, "en un programa por puntos, «descartar» no existe: es cancelar"
        assert w._pedir_nombre("x", "grabacion") is ventana_real.DESCARTAR
        assert [d[2] for d in w.x.dialogos] == [None, DESCARTAR_NOMBRE], "la bitácora debe decir lo que de verdad se contestó"
    finally:
        cerrar(w)
        borrar_desechable(carpeta)


def test_ex11_la_corrida_corta_cabe_en_la_suite(corrida_corta):
    assert corrida_corta.segundos < 10, f"la corrida corta tardó {corrida_corta.segundos:.1f} s (el tope son 10 s)"


def test_ex11_los_dos_tamanos_existen():
    corto, largo = ex.TAMANOS["corto"], ex.TAMANOS["largo"]
    assert corto[0] * corto[1] <= 200, "el corto es de unos cientos de gestos como mucho"
    assert largo[0] * largo[1] >= 50 * corto[0] * corto[1], "el largo debe ser MUCHO más largo que el corto"


@pytest.mark.skipif(not os.environ.get("EXPLORADOR_LARGO"),
                    reason="la corrida larga (≈15 min) es solo a pedido: EXPLORADOR_LARGO=1 pytest tests/test_explorador.py "
                           "(o: python tests/explorador.py --largo)")
def test_ex11_la_corrida_larga_a_pedido():
    resultado = ex.explorar("largo", int(os.environ.get("EXPLORADOR_SEMILLA", ex.SEMILLA)), con_reduccion=True)
    print(resultado.cobertura.informe())
    assert resultado.hallazgo is None, str(resultado.hallazgo) + "\n" + resultado.hallazgo.como_prueba_fija()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EX2: LO QUE SE REVISA TRAS CADA GESTO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _servos_lejanos(robot):
    """Dos poses válidas cuya punta queda muy lejos (más de lo que ningún paso legal recorre)."""
    minimos = tuple(j.servo.minimo for j in robot.articulaciones)
    maximos = tuple(j.servo.maximo for j in robot.articulaciones)
    return minimos, maximos


def _efecto_racha(s):
    for i in range(3):
        s.w.avisar(f"Aviso repetido número {i}", "aviso", "Prueba")


def _efecto_rebote(s):
    s.w.avisar("No se mueve. Para pasar por (1, 0, 1) hay que cambiar de postura.", "aviso", "Jog", destino_mm=[1, 0, 1])


def _efecto_brinco(s):
    a, b = _servos_lejanos(s.w.robot)
    # AL ENLACE CRUDO, NO POR LA GUARDIA: ES EL DEFECTO «LA APP MANDÓ UN BRINCO A LA PLACA». POR LA GUARDIA YA NO SALDRÍA (ESO LO
    # PRUEBA test_guardia_en_la_ventana) Y LA REVISIÓN `brinco` DEL EXPLORADOR SE QUEDARÍA SIN DEFECTO QUE CAZAR
    s.w.enlace._enlace.enviar(*a)
    s.w.enlace._enlace.enviar(*b)


def _efecto_perdida_borra_un_programa(s):
    os.remove(os.path.join(s.w.carpeta_programas, ex.SEMBRADOS[0]))


def _efecto_perdida_cambia_un_programa_sin_preguntar(s):
    with open(os.path.join(s.w.carpeta_programas, ex.SEMBRADOS[0]), "a", encoding="utf-8") as f:
        f.write("\n# cambiado a escondidas\n")


def _efecto_excepcion(s):
    raise RuntimeError("algo reventó")


def _efecto_excepcion_en_un_slot(s):
    try:
        raise ValueError("algo reventó dentro de un slot")
    except ValueError:
        sys.excepthook(*sys.exc_info())          # LO QUE HACE PyQt5 CUANDO UN SLOT LEVANTA


def _efecto_orden_negada(s):
    # LA GUARDIA DE LA VENTANA SE NIEGA DE VERDAD Y LEVANTA OrdenNegada: LO QUE LA REVISIÓN LEE ES x.negadas, NO LA EXCEPCIÓN
    with contextlib.suppress(OrdenNegada):
        s.w.enlace.enviar(*(j.servo.maximo + 50 for j in s.w.robot.articulaciones))


def _efecto_reposo_boton_detener_trabado(s):
    s.w.BTN_DETENER.setEnabled(True)


EFECTOS = [
    ("racha_de_avisos", _efecto_racha),
    ("rebote", _efecto_rebote),
    ("brinco", _efecto_brinco),
    ("perdida", _efecto_perdida_borra_un_programa),
    ("perdida", _efecto_perdida_cambia_un_programa_sin_preguntar),
    ("excepcion", _efecto_excepcion),
    ("excepcion", _efecto_excepcion_en_un_slot),
    ("orden_negada", _efecto_orden_negada),
    ("reposo", _efecto_reposo_boton_detener_trabado),
]


@pytest.fixture
def con_efecto(monkeypatch):
    """Hace que el gesto PRUEBA ejecute un defecto sembrado, en vez de tocar un control."""
    def poner(efecto):
        original = ex.Sesion._ejecutar

        def _ejecutar(self, g):
            if g == PRUEBA:
                efecto(self)
            else:
                original(self, g)
        monkeypatch.setattr(ex.Sesion, "_ejecutar", _ejecutar)
    return poner


@pytest.mark.parametrize("clase,efecto", EFECTOS, ids=[f"{c}-{e.__name__[8:]}" for c, e in EFECTOS])
def test_ex02_cada_revision_caza_su_defecto(con_efecto, clase, efecto):
    con_efecto(efecto)
    hallazgo = ex.reproducir([Gesto("esperar", "", 0.0, 0, 0.1), PRUEBA])
    assert hallazgo is not None, f"nadie vio el defecto de clase «{clase}»"
    assert hallazgo.clase == clase, f"se esperaba «{clase}» y fue «{hallazgo.clase}»: {hallazgo.detalle}"
    assert hallazgo.gestos[-1] == PRUEBA, "el hallazgo debe señalar el gesto que lo provocó"


def test_ex02_un_movimiento_que_no_termina_es_hallazgo(monkeypatch):
    def nunca_termina(w, tope=0):
        raise NoTermina("el movimiento sigue pidiendo pasos")
    monkeypatch.setattr(ex, "correr", nunca_termina)
    hallazgo = ex.reproducir([Gesto("llegar")])
    assert hallazgo is not None and hallazgo.clase == "no_termina"


def test_ex02_sin_defecto_no_hay_acusacion_falsa():
    """El control de sanidad de cada revisión: gestos normales sobre una ventana sana no acusan a nadie."""
    gestos = [Gesto("elegir", "PESTANAS_MOVIMIENTO", 0.5), Gesto("arrastrar", "SLIDER_1", 0.3, 5, 0.05),
              Gesto("arrastrar", "SLIDER_2", 0.6, 7, 0.3), Gesto("llegar"), Gesto("esperar", "", 0.0, 0, 1.0)]
    assert ex.reproducir(gestos) is None


def test_ex02_los_clics_repetidos_de_la_persona_no_son_una_racha(con_efecto):
    """Tres avisos seguidos escritos por TRES CLICS del mismo botón son decisión de la persona, no un defecto."""
    def efecto(s):
        s.w.avisar("Mismo aviso de un clic", "aviso", "Prueba")
    con_efecto(efecto)
    # el gesto PRUEBA es de tipo «tecla»; uno de tipo click con el mismo efecto sí queda exento
    clic = Gesto("click", "PRUEBA", 0.0, 0, 0.0)
    original = ex.Sesion._ejecutar

    def _ejecutar(self, g):
        if g.control == "PRUEBA":
            efecto(self)
        else:
            original(self, g)
    with mock.patch.object(ex.Sesion, "_ejecutar", _ejecutar):
        assert ex.reproducir([clic, clic, clic]) is None


def test_ex02_insistir_contra_el_rojo_moviendo_el_robot_entre_intentos_no_es_una_racha():
    """Tres arrastres DISTINTOS contra el rojo, cada uno con su único aviso y con el robot moviéndose entre uno y otro, son tres
    respuestas a tres intentos de la persona. Lo encontró la exploración larga del 25-nov-2026 y no era un defecto de la app."""
    gestos = [Gesto("elegir", "PESTANAS_MOVIMIENTO", 0.5), Gesto("arrastrar", "SLIDER_X", 0.9926006880852335, 945821, 0.3),
              Gesto("arrastrar", "SLIDER_Z", 0.009328207233619379, 261169, 0.3),
              Gesto("arrastrar", "SLIDER_X", 0.43939451090100756, 162581, 0.3),
              Gesto("arrastrar", "SLIDER_X", 0.7688710002305318, 351944, 0.3),
              Gesto("arrastrar", "SLIDER_X", 0.34134116422095195, 22030, 0.3)]
    assert ex.reproducir(gestos) is None


def test_ex02_topar_arriba_y_luego_abajo_son_dos_avisos_distintos_no_una_racha():
    """Topar con el rojo de arriba y enseguida con el de abajo da dos avisos con OTRO tope: dicen cosas distintas aunque el robot
    no haya alcanzado a moverse entre uno y otro (aquí va al 46 % de velocidad). Tercera vuelta de la exploración larga."""
    gestos = [Gesto("arrastrar", "SLIDER_VELOCIDAD", 0.3953828366880614, 501895, 0.0),
              Gesto("arrastrar", "SLIDER_3", 0.8571246634377789, 423644, 1.5), Gesto("elegir", "PESTANAS_MOVIMIENTO", 0.5),
              Gesto("arrastrar", "SLIDER_X", 0.9926006880852335, 945821, 0.008),
              Gesto("arrastrar", "SLIDER_Z", 0.28345775271251406, 290231, 0.008),
              Gesto("arrastrar", "SLIDER_X", 0.43939451090100756, 162581, 0.0),
              Gesto("arrastrar", "SLIDER_X", 0.7688710002305318, 351944, 0.0),
              Gesto("arrastrar", "SLIDER_X", 0.34134116422095195, 22030, 0.0)]
    assert ex.reproducir(gestos) is None


def _mover_un_grado(s):
    servos = list(s.w.servos)
    servos[0] += 1
    s.w.poner_servos(servos)
    s.w._sincronizar_sliders()          # COMO CUANDO ALGO TERMINA DE MOVERSE: LOS SLIDERS DICEN DÓNDE QUEDÓ


def test_ex02_un_aviso_por_gesto_sin_que_nada_se_mueva_si_es_una_racha(con_efecto):
    """Control: tres gestos distintos que escriben el mismo aviso SIN que el robot se mueva entre uno y otro siguen siendo
    una racha (la app debió callarse hasta que algo se moviera)."""
    con_efecto(lambda s: s.w.avisar("La punta no llega más allá de X = 219 mm", "aviso", "Jog", tope_mm=219))
    hallazgo = ex.reproducir([PRUEBA, PRUEBA, PRUEBA])
    assert hallazgo is not None and hallazgo.clase == "racha_de_avisos", str(hallazgo)


def test_ex02_varios_avisos_de_un_mismo_gesto_si_son_una_racha_aunque_el_robot_se_mueva(con_efecto):
    """Control: el defecto de origen (un solo arrastre que avisa una y otra vez mientras el robot se mueve) sigue siendo racha."""
    def efecto(s):
        for i in range(3):
            _mover_un_grado(s)
            s.w.avisar(f"La punta no llega más allá de X = {200 + i} mm", "aviso", "Jog", tope_mm=200 + i)
    con_efecto(efecto)
    hallazgo = ex.reproducir([PRUEBA])
    assert hallazgo is not None and hallazgo.clase == "racha_de_avisos", str(hallazgo)


def test_ex02_avisos_mientras_la_persona_solo_espera_si_son_una_racha():
    """Control: avisos que salen mientras la persona no hace nada, aunque el robot se mueva entre uno y otro, nadie los pidió."""
    original = ex.Sesion._ejecutar

    def _ejecutar(self, g):
        if g.tipo == "esperar":
            _mover_un_grado(self)
            self.w.avisar("Aviso que nadie pidió", "aviso", "Prueba")
        else:
            original(self, g)
    espera = Gesto("esperar", "", 0.0, 0, 0.1)
    with mock.patch.object(ex.Sesion, "_ejecutar", _ejecutar):
        hallazgo = ex.reproducir([espera, espera, espera])
    assert hallazgo is not None and hallazgo.clase == "racha_de_avisos", str(hallazgo)


def test_ex02_el_tope_del_brinco_no_deja_fuera_un_paso_legal():
    """El umbral de «brinco» (BRINCO_ROJO_MM) tiene que estar POR ENCIMA de lo que un paso legal (15° por servo) puede mover
    la punta; si no, el explorador acusaría a pasos legales. Medido con poses al azar en todo el rango de cada servo."""
    rng = np.random.default_rng(20261125)
    minimos = np.array([j.servo.minimo for j in ROBOT.articulaciones], float)
    maximos = np.array([j.servo.maximo for j in ROBOT.articulaciones], float)
    salto = ROBOT.movimiento.salto_max_por_paso
    mayor = 0.0
    for _ in range(4000):
        a = rng.uniform(minimos, maximos)
        b = np.clip(a + rng.uniform(-salto, salto, size=a.shape), minimos, maximos)
        mayor = max(mayor, avance_punta_mm(ROBOT, a, b))
    assert mayor < ex.BRINCO_ROJO_MM, f"un paso legal movió la punta {mayor:.0f} mm: el umbral {ex.BRINCO_ROJO_MM} no sirve"
    assert ex.BRINCO_REPORTE_MM < ex.BRINCO_ROJO_MM
    a, b = _servos_lejanos(ROBOT)
    assert avance_punta_mm(ROBOT, a, b) > ex.BRINCO_ROJO_MM, "el brinco sembrado de la prueba tiene que pasar el umbral"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EX3 y EX4: REPRODUCIBLE, MISMA SEMILLA MISMO RECORRIDO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_ex03_un_hallazgo_trae_semilla_y_gestos_y_se_repite():
    semilla, h, _ = _donde_lo_encuentra("rojo")
    assert h.semilla == semilla and len(h.gestos) > 0
    with _sembrar_el_rojo():
        otra_vez = ex.reproducir(h.gestos, semilla=h.semilla)
    assert otra_vez is not None and otra_vez.clase == h.clase, "la lista de gestos no repite el hallazgo"
    assert str(h.semilla) in str(h) and "Gesto(" in str(h), "el reporte debe traer la semilla y los gestos"
    # EL TEXTO QUE SE PEGA COMO PRUEBA FIJA SE PUEDE LEER DE VUELTA Y DA LOS MISMOS GESTOS
    texto = h.como_prueba_fija("copia")
    nombre, clase, gestos = eval(texto.split("\n", 1)[1].strip().rstrip(","), {"Gesto": Gesto})
    assert (nombre, clase, tuple(gestos)) == ("copia", h.clase, h.gestos)


def test_ex04_misma_semilla_mismo_recorrido():
    a = ex.explorar("corto", 31, tolerar=True).cobertura.recorrido
    b = ex.explorar("corto", 31, tolerar=True).cobertura.recorrido
    c = ex.explorar("corto", 32, tolerar=True).cobertura.recorrido
    assert len(a) > 50 and a == b, "la misma semilla debe dar exactamente los mismos gestos"
    assert a != c, "otra semilla debe dar otro recorrido"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EX5: DEMUESTRA QUE CAZA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_ex05_encuentra_el_rojo_que_dejaba_rebotar_al_slider():
    _, hallazgo, _ = _donde_lo_encuentra("rojo")
    assert hallazgo.clase in ("rebote", "racha_de_avisos"), str(hallazgo)


def test_ex05_encuentra_el_brinco_grosero():
    _, hallazgo, _ = _donde_lo_encuentra("brinco")
    assert hallazgo.clase == "brinco" or (hallazgo.clase == "orden_negada" and "salto_excedido" in str(hallazgo)), str(hallazgo)


def test_ex05_encuentra_la_grabacion_tirada_al_cancelar():
    _, hallazgo, _ = _donde_lo_encuentra("perdida")
    assert hallazgo.clase == "perdida", str(hallazgo)


def test_ex05_sin_los_defectos_las_mismas_semillas_no_encuentran_nada_de_eso():
    """El control: con el código de hoy, las MISMAS semillas no acusan lo que acusaron con el defecto sembrado (si no, las
    pruebas de arriba no probarían que el explorador caza: acusaría siempre, o acusaría un defecto real distinto)."""
    for defecto, (_, clases, _, _, _) in DEFECTOS_SEMBRADOS.items():
        semilla, _, limpio = _donde_lo_encuentra(defecto)
        assert limpio is None or limpio.clase not in clases, f"semilla {semilla} sin sembrar acusa {limpio.clase}: {limpio}"


def test_ex05_la_busqueda_de_semilla_no_acepta_la_que_acusa_sin_sembrar(monkeypatch):
    """La búsqueda, con corridas de mentira: la semilla calibrada (5) acusa «rebote» también SIN sembrar (no demostraría
    nada), la 1 no encuentra el defecto y la 2 sí, limpia sin sembrar: gana la 2 y avisa que hay que recalibrar. Y si
    ninguna lo encuentra, la prueba falla diciendo qué dio cada una."""
    sembrado = []

    @contextlib.contextmanager
    def sembrar():
        sembrado.append(True)
        try:
            yield
        finally:
            sembrado.pop()

    de_mentira = {(5, True): "rebote", (5, False): "rebote", (2, True): "rebote", (2, False): "excepcion"}

    def explorar(tamano, semilla, tolerar=True):
        clase = de_mentira.get((semilla, bool(sembrado)))
        return ex.Resultado(None, None if clase is None else Hallazgo(clase, "de mentira", (), semilla), 0.0)

    monkeypatch.setattr(ex, "explorar", explorar)
    monkeypatch.setitem(DEFECTOS_SEMBRADOS, "de-mentira", (sembrar, ("rebote",), {}, False, 5))
    with pytest.warns(UserWarning, match="ya no lo encuentra"):
        semilla, hallazgo, limpio = _donde_lo_encuentra.__wrapped__("de-mentira")
    assert (semilla, hallazgo.clase, limpio.clase) == (2, "rebote", "excepcion")
    monkeypatch.setitem(DEFECTOS_SEMBRADOS, "inencontrable", (sembrar, ("perdida",), {}, False, 5))
    with pytest.raises(AssertionError, match="5: sembrado dio «rebote».*2: sembrado dio «rebote»"):
        _donde_lo_encuentra.__wrapped__("inencontrable")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EX6: EL DETECTOR DE RACHAS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _aviso(segundo, mensaje, nivel="aviso", origen="Jog"):
    return reg.Evento(dt.datetime(2026, 11, 25, 1, 6, 12) + dt.timedelta(seconds=segundo), nivel, origen, mensaje, {})


def _registro_del_25_de_nov():
    """Un extracto SINTÉTICO con la forma del registro real: 31 negativas seguidas del planeador en 21 s, rodeadas de lo normal."""
    eventos = [_aviso(-30, "Programa «demo» cargado", "info", "Programa"),
               _aviso(-20, "La punta no llega más allá de Y = 0 mm desde esta pose."),
               _aviso(-10, "La punta no llega más allá de Y = 0 mm desde esta pose."),
               _aviso(-5, "Cosecha lista", "ok", "Cosecha")]
    eventos += [_aviso(21 * i / 30, f"No se mueve. Para pasar por ({155 + i}, 0, 1654) el robot tendría que cambiar de postura "
                                    f"(el brazo barrería hasta {20 - i % 7} mm fuera de la línea) y el modo es «negar».")
                for i in range(31)]
    eventos += [_aviso(30, "Cambio de postura cerca de (855, 0, 1654)"), _aviso(31, "Listo", "ok", "Jog")]
    return eventos


def test_ex06_el_detector_encuentra_la_racha_de_31_avisos():
    rachas = rachas_de_avisos(_registro_del_25_de_nov())
    assert [r.longitud for r in rachas] == [31], rachas
    r = rachas[0]
    assert (r.nivel, r.origen) == ("aviso", "Jog") and r.primero == 4
    assert r.segundos == pytest.approx(21)


def test_ex06_dos_avisos_iguales_no_son_racha_y_tres_si():
    dos = [_aviso(i, f"Mismo aviso en ({i}, 0, 0)") for i in range(2)]
    tres = [_aviso(i, f"Mismo aviso en ({i}, 0, 0)") for i in range(3)]
    assert rachas_de_avisos(dos) == [] and len(rachas_de_avisos(tres)) == 1
    assert forma_del_aviso("Para (155, 0, 1654) y 3,5") == forma_del_aviso("Para (1, -2, 99) y 40")


def test_ex06_lo_que_rompe_una_racha_la_corta():
    eventos = [_aviso(0, "A"), _aviso(1, "A"), _aviso(2, "otro aviso"), _aviso(3, "A"), _aviso(4, "A"),
               _aviso(5, "A", "info"), _aviso(6, "A"), _aviso(7, "A", origen="Otro")]
    assert rachas_de_avisos(eventos) == []


def test_ex06_lee_un_archivo_de_registro(tmp_path):
    ruta = tmp_path / "registro-2026-11-25.jsonl"
    reg.exportar_jsonl(_registro_del_25_de_nov(), str(ruta))
    rachas = rachas_del_archivo(str(ruta))
    assert [r.longitud for r in rachas] == [31]
    assert rachas[0].desde == dt.datetime(2026, 11, 25, 1, 6, 12)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EX7: CARPETA DESECHABLE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _huella_de(carpeta):
    if not os.path.isdir(carpeta):
        return None
    huellas = {}
    for raiz, _dirs, archivos in os.walk(carpeta):
        for a in sorted(archivos):
            ruta = os.path.join(raiz, a)
            huellas[os.path.relpath(ruta, carpeta)] = hashlib.sha1(open(ruta, "rb").read()).hexdigest()
    return huellas


def _ajustes_reales():
    s = QtCore.QSettings("robot-cosechador", "cosechador")
    return {k: str(s.value(k)) for k in s.allKeys()}


def _desechables_de_este_proceso():
    """(creadas, borradas) hasta ahora. Las de ESTE proceso: mirar todas las del temporal del sistema hacía fallar la
    prueba cuando otra corrida (otra suite, una exploración larga) creaba o borraba las suyas al mismo tiempo."""
    return len(helpers_ventana.CREADAS), len(helpers_ventana.BORRADAS)


def test_ex07_no_toca_programas_ni_registro_ni_ajustes_de_la_persona():
    programas, logs = prog.CARPETA_PROGRAMAS, reg.CARPETA_LOGS
    antes = (_huella_de(programas), _huella_de(logs), _ajustes_reales())
    creadas, borradas = _desechables_de_este_proceso()
    resultado = ex.explorar("corto", 77, tolerar=True)         # CON GRABAR, GUARDAR, DESCARTAR Y EDITAR AL AZAR
    despues = (_huella_de(programas), _huella_de(logs), _ajustes_reales())
    nuevas, tiradas = helpers_ventana.CREADAS[creadas:], helpers_ventana.BORRADAS[borradas:]
    assert resultado.cobertura.gestos > 50
    assert antes[0] == despues[0], "cambió la carpeta programas/ de la persona"
    assert antes[1] == despues[1], "cambió la carpeta logs/ de la persona"
    assert antes[2] == despues[2], "cambiaron los ajustes (QSettings) de la persona"
    assert nuevas, "control: la exploración no usó ninguna carpeta desechable"
    assert not [c for c in nuevas if os.path.exists(c)], "quedó una carpeta desechable sin borrar"
    assert sorted(tiradas) == sorted(nuevas), "se borró una carpeta que esta exploración no creó (o una dos veces)"


def test_ex07_la_ventana_aislada_vive_toda_en_la_carpeta_desechable():
    carpeta = carpeta_desechable()
    w = ventana_aislada(ROBOT, carpeta)
    try:
        base = os.path.realpath(tempfile.gettempdir())
        assert os.path.realpath(w.carpeta_programas).startswith(base + os.sep)
        assert os.path.realpath(w.ajustes.fileName()).startswith(base + os.sep)
        assert w.registro.carpeta is None, "el registro de la ventana escribiría un archivo"
        assert os.path.realpath(w.carpeta_programas) != os.path.realpath(prog.CARPETA_PROGRAMAS)
    finally:
        cerrar(w)
        borrar_desechable(carpeta)
    assert not os.path.exists(carpeta)


def test_ex07_se_niega_a_trabajar_o_borrar_fuera_de_lo_desechable():
    with pytest.raises(RuntimeError):
        ventana_aislada(ROBOT, CARPETA_REPO)
    with pytest.raises(RuntimeError):
        borrar_desechable(os.path.join(CARPETA_REPO, "programas"))
    assert os.path.isdir(os.path.join(CARPETA_REPO, "programas"))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EX8: EL INFORME DE COBERTURA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_ex08_el_informe_dice_que_se_toco_cuantas_veces_y_que_nunca(corrida_corta):
    cob = corrida_corta.cobertura
    informe = cob.informe()
    assert cob.tocados and cob.controles
    for nombre, veces in cob.tocados.most_common(3):
        assert f"{nombre} {veces}" in informe, f"el informe no dice cuántas veces se tocó {nombre}"
    nunca = cob.nunca_tocados()
    assert all(n not in cob.tocados for n in nunca)
    assert set(nunca) | set(cob.tocados) | set(ex.EXCLUIDOS) >= set(cob.controles), "un control no está ni tocado ni en «nunca»"
    assert f"Controles NUNCA tocados ({len(nunca)})" in informe
    for nombre in nunca[:3]:
        assert nombre in informe
    for (postura, camino), n in cob.modos.items():
        assert f"{postura:<9} / {camino:<10} {n}" in informe, "el informe no lista un modo tocado"
    for excluido in ex.EXCLUIDOS:
        assert excluido in informe, "lo excluido a propósito debe declararse en el informe"
    assert sum(cob.tocados.values()) + cob.sin_efecto <= cob.gestos


def test_ex08_los_controles_se_descubren_no_se_listan_a_mano():
    carpeta = carpeta_desechable()
    w = ventana_aislada(ROBOT, carpeta)
    try:
        todos = ex.descubrir(w)
        for esperado in ("SLIDER_1", "SLIDER_X", "BTN_GRABAR", "SELECTOR_POSTURA", "PESTANAS_MOVIMIENTO", "TE_X", "LISTA_PUNTOS"):
            assert esperado in todos, f"{esperado} no se descubrió"
        assert len(todos) >= 40
    finally:
        cerrar(w)
        borrar_desechable(carpeta)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EX9: REDUCIR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_ex09_reduce_a_la_lista_mas_corta_que_reproduce():
    _, h, _ = _donde_lo_encuentra("rojo")
    with _sembrar_el_rojo():
        corto = ex.reducir(h)
        assert corto.clase == h.clase and len(corto.gestos) < len(h.gestos), (len(h.gestos), len(corto.gestos))
        assert ex.reproducir(corto.gestos) is not None, "la lista reducida ya no reproduce el hallazgo"
        # MÍNIMA: quitarle cualquier gesto, de uno en uno, y ya no hay hallazgo de esa clase
        for i in range(len(corto.gestos)):
            sin = corto.gestos[:i] + corto.gestos[i + 1:]
            r = ex.reproducir(sin) if sin else None
            assert r is None or r.clase != corto.clase, f"el gesto {i + 1} sobraba: {corto.gestos[i]}"


def test_ex09_un_hallazgo_que_no_se_repite_se_entrega_tal_cual():
    h = Hallazgo("rebote", "no se repite", (Gesto("esperar", "", 0.0, 0, 0.1),) * 3)
    assert ex.reducir(h, reproducir_con=lambda gestos: None) is h


def test_ex09_un_hallazgo_nuevo_no_se_reduce_hasta_volverlo_uno_conocido():
    """Lo que pasó el 25-nov-2026: un rebote NUEVO se «redujo» quitándole el gesto que lo hacía nuevo y quedó convertido en
    el rebote ya conocido: misma clase, otro defecto."""
    conocido = next(c for c in ex.CONOCIDOS if c.id.startswith("D7-"))
    clave, relleno = Gesto("elegir", "SELECTOR_POSTURA", 0.9, 0, 0.0), Gesto("esperar", "", 0.0, 0, 0.1)
    provoca = Gesto("arrastrar", "SLIDER_X", 0.3, 1, 0.0)

    def repetir(gestos):   # CON EL GESTO CLAVE, EL DEFECTO NUEVO; SIN ÉL, EL CONOCIDO; SIN EL QUE LO PROVOCA, NADA
        if provoca not in gestos:
            return None
        hasta = gestos[:gestos.index(provoca) + 1]
        return Hallazgo("rebote", "uno nuevo" if clave in gestos else f"({conocido.contiene})", hasta)

    h = Hallazgo("rebote", "uno nuevo", (relleno, clave, relleno, provoca, relleno))
    corto = ex.reducir(h, reproducir_con=repetir)
    assert corto.detalle == "uno nuevo" and corto.gestos == (clave, provoca), corto


def test_ex09_al_repetir_se_pueden_tolerar_los_conocidos_del_camino():
    """Para reducir un hallazgo nuevo hay que poder pasar de largo los conocidos, igual que la corrida que lo encontró."""
    d7 = list(next(h for h in HALLAZGOS if h[0].startswith("D7-"))[2])
    assert ex.reproducir(d7) is not None, "control: sin tolerar, el conocido sí detiene"
    assert ex.reproducir(d7, tolerar=True) is None


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EX10: LOS HALLAZGOS SE QUEDAN COMO PRUEBA FIJA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# (nombre, clase, gestos, motivo si el defecto sigue vivo en el código de hoy). Los pega, ya reducidos, `python tests/explorador.py`.
# Si el defecto sigue vivo: fallo esperado ESTRICTO (xfail strict) con su motivo. El día que se arregle, esta prueba se pone roja
# (XPASS) para que alguien quite la marca: así un arreglo no pasa sin que la prueba lo registre.
HALLAZGOS = (
    ("D5-calcular-con-inf-o-nan", "excepcion", (
        Gesto(tipo='escribir', control='TE_X', v=0.0, s=10, dt=0.0),
        Gesto(tipo='escribir', control='TE_Y', v=0.0, s=10, dt=0.0),
        Gesto(tipo='escribir', control='TE_Z', v=0.0, s=9, dt=0.0),
        Gesto(tipo='click', control='CALCULAR', v=0.0, s=778166, dt=0.008),
    ), ""),   # ARREGLADO EL 25-nov-2026 (tests/test_calcular.py, D5): SIN MOTIVO, LA PRUEBA FIJA EXIGE QUE YA NO SE REPITA
    ("D6-rebote-negar-con-la-punta-en-camino", "rebote", (
        Gesto(tipo='elegir', control='PESTANAS_MOVIMIENTO', v=0.5, s=0, dt=0.0),
        Gesto(tipo='arrastrar', control='SLIDER_Y', v=0.1901583561638126, s=402479, dt=1.5),
        Gesto(tipo='arrastrar', control='SLIDER_X', v=0.25891675029296335, s=536110, dt=0.008),
        Gesto(tipo='arrastrar', control='SLIDER_Y', v=0.9677999949201714, s=375441, dt=0.05),
        Gesto(tipo='elegir', control='SELECTOR_POSTURA', v=0.6424961212029794, s=0, dt=0.0),
        Gesto(tipo='arrastrar', control='SLIDER_Y', v=0.6837036588129731, s=624556, dt=0.008),
    ), ""),   # ARREGLADO EL 25-nov-2026 (tests/test_postura_en_camino.py, D6): AL PASAR A «NEGARSE» SE CANCELA LO PENDIENTE
    ("D7-rebote-reemplazar-con-la-punta-en-camino", "rebote", (
        Gesto(tipo='elegir', control='PESTANAS_MOVIMIENTO', v=0.5, s=0, dt=0.0),
        Gesto(tipo='elegir', control='SELECTOR_EN_CAMINO', v=0.2238158098647538, s=0, dt=0.0),
        Gesto(tipo='arrastrar', control='SLIDER_Z', v=0.5457821488113972, s=477770, dt=0.3),
        Gesto(tipo='arrastrar', control='SLIDER_X', v=0.8835869371580177, s=34855, dt=0.3),
        Gesto(tipo='arrastrar', control='SLIDER_Y', v=0.14673451377324587, s=607881, dt=1.5),
    ), "riesgo declarado de «Reemplazar»: con la punta en camino (aquí, 0.3 s entre soltar un slider y agarrar el otro) el rojo "
       "se mide desde el punto al que va y el tramo nuevo sale en diagonal desde donde va la punta; la diagonal cruza una zona "
       "que el robot no alcanza, el planeador se niega («pasa por … una zona que el robot no alcanza») y el slider vuelve. "
       "Lo encontró la exploración larga del 25-nov-2026. Esperando a que llegue, o en «En cola» y «Cancelar», no pasa"),
    ("D7n-rebote-reemplazar-en-negarse-con-la-punta-en-camino", "rebote", (
        Gesto(tipo='elegir', control='SELECTOR_POSTURA', v=0.9, s=0, dt=0.0),
        Gesto(tipo='elegir', control='PESTANAS_MOVIMIENTO', v=0.5, s=0, dt=0.0),
        Gesto(tipo='arrastrar', control='SLIDER_Y', v=0.38591912681792206, s=23154, dt=0.0),
        Gesto(tipo='elegir', control='SELECTOR_EN_CAMINO', v=0.16227008287739508, s=0, dt=0.05),
        Gesto(tipo='arrastrar', control='SLIDER_Z', v=0.734530518561858, s=528255, dt=0.05),
        Gesto(tipo='arrastrar', control='SLIDER_X', v=0.26604663315456467, s=620442, dt=0.008),
    ), "el riesgo declarado de «Reemplazar», con la postura en «negarse»: la diagonal desde donde va la punta pediría cambiar "
       "de postura, el planeador se niega («tendría que cambiar de postura») y el slider vuelve. Lo encontró la cuarta vuelta "
       "de la exploración larga del 25-nov-2026"),
)


def _prueba_fija(nombre, clase, gestos, motivo):
    marcas = [pytest.mark.xfail(strict=True, reason=motivo)] if motivo else []
    return pytest.param(clase, gestos, id=nombre, marks=marcas)


@pytest.mark.parametrize("clase,gestos", [_prueba_fija(*h) for h in HALLAZGOS])
def test_ex10_hallazgo_fijo(clase, gestos):
    hallazgo = ex.reproducir(list(gestos))
    assert hallazgo is None, f"el hallazgo sigue vivo:\n{hallazgo}"


def test_ex10_cada_defecto_conocido_tiene_su_prueba_fija():
    """Lo que el explorador deja pasar como «conocido» (CONOCIDOS) tiene que tener su prueba fija arriba, y al revés."""
    fijos = {h[0].split("-")[0] for h in HALLAZGOS if h[3]}       # LOS ABIERTOS: UNO ARREGLADO YA NO TIENE MOTIVO
    conocidos = {c.id.split("-")[0] for c in ex.CONOCIDOS}
    assert conocidos == fijos, f"conocidos sin prueba fija o al revés: {conocidos ^ fijos}"
    for c in ex.CONOCIDOS:
        assert c.motivo and c.contiene


def test_ex10_las_pruebas_fijas_reproducen_de_verdad_lo_que_dicen():
    """El control de que el instrumento no miente: cada defecto abierto SÍ se reproduce, con la clase que dice."""
    for nombre, clase, gestos, motivo in HALLAZGOS:
        if motivo:
            hallazgo = ex.reproducir(list(gestos))
            assert hallazgo is not None and hallazgo.clase == clase, f"{nombre}: ya no se reproduce como «{clase}»: {hallazgo}"
