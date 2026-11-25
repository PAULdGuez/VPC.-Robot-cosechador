"""Piezas compartidas para usar la ventana REAL desde las pruebas, sin pantalla y sin tocar nada de la persona (no son pruebas).

Aquí vive UNA sola versión de lo que antes estaba escrito dos veces con comportamiento distinto (el ayudante `correr`):

- La ventana aislada: carpeta de programas y ajustes dentro de una carpeta desechable, registro sin archivo, reloj falso
  y los diálogos contestados desde la prueba (un diálogo modal de verdad en `offscreen` cuelga el proceso para siempre).
- El temporizador del movimiento simulado con el reloj falso: `avanzar` y `correr` disparan los pasos en el instante en
  que el temporizador real los dispararía (cada paso, `timer.interval()` después del anterior; el primero, 1 ms después de
  que se encola el plan), así que lo que sale hacia la placa lleva la hora que tendría en la app.
- El espía de todo lo que sale hacia la placa. La guardia de salida es la DE VERDAD, la que lleva la ventana dentro (comparte el
  reloj falso con ella): lo que la guardia niega no sale, y `x.negadas` anota cada primera negativa.
"""
import os
import sys
import tempfile
import shutil
import collections
from contextlib import contextmanager

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ["QT_QPA_PLATFORM"] = "offscreen"      # NUNCA UNA VENTANA EN LA PANTALLA DE LA PERSONA

from PyQt5 import QtCore, QtGui, QtWidgets

from cosechador import registro as reg
from cosechador.hardware.guardia import GuardiaDeSalida, OrdenNegada
from cosechador.movimiento.grabadora import Grabadora
from helpers_guardia import EnlaceFalso, Reloj

# LA APLICACIÓN DE Qt NECESITA UNA REFERENCIA VIVA A NIVEL DE MÓDULO: SIN ELLA SE DESTRUYE Y EL PROCESO ABORTA
# («Must construct a QApplication before a QWidget»)
APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
if QtGui.QGuiApplication.platformName() != "offscreen":
    raise RuntimeError(f"Qt arrancó con la plataforma {QtGui.QGuiApplication.platformName()!r}: las pruebas solo corren "
                       "sin pantalla (QT_QPA_PLATFORM=offscreen).")

DESCARTAR_NOMBRE = "<descartar>"    # LA RESPUESTA DEL DIÁLOGO DEL NOMBRE QUE PIDE TIRAR LA GRABACIÓN
TOPE_PASOS = 20000                  # MÁS PASOS QUE ESTOS SIN TERMINAR = EL MOVIMIENTO NO TERMINA (LA COSECHA MÁS LARGA TIENE ~1000)
SEGUNDOS_TRAS_ABRIR = 1.0           # LO QUE TARDA UNA PERSONA, COMO POCO, EN TOCAR ALGO DESPUÉS DE QUE ABRE LA VENTANA


EXCEPCIONES = []                    # LO QUE REVENTÓ DENTRO DE UN SLOT DE Qt (UNA LISTA PARA TODAS LAS VENTANAS: SE ANOTA AL AZAR EN CUALQUIERA)


class NoTermina(Exception):
    """Un movimiento que sigue pidiendo pasos después de TOPE_PASOS."""


class Estado:
    """Todo lo que la ventana aislada anota de sí misma. Vive en `w.x`."""

    def __init__(self, carpeta):
        self.carpeta = carpeta
        self.reloj = Reloj(1000.0)
        self.vence = None                 # CUÁNDO DISPARARÍA EL TEMPORIZADOR REAL EL SIGUIENTE PASO (None: NO ESTÁ CORRIENDO)
        self.enviados = []                # [(hora, servos, origen del plan)] TODO LO QUE SALIÓ HACIA LA PLACA
        self.negadas = []                 # [(hora, Violacion, servos)] CADA VEZ QUE LA GUARDIA SE NEGÓ POR PRIMERA VEZ (LA QUE LA DETUVO)
        self.eventos = []                 # TODO LO QUE SE AVISÓ (sin el tope de 2000 del registro de la ventana)
        self.dialogos = []                # [(clase, texto, respuesta)] LO QUE SE PREGUNTÓ Y LO QUE SE CONTESTÓ
        self.nombres = collections.deque()      # RESPUESTAS A «¿CÓMO SE LLAMA?» (None = CANCELAR)
        self.confirmas = collections.deque()    # RESPUESTAS A LAS PREGUNTAS DE SÍ/NO
        self.excepciones = EXCEPCIONES    # (la lista de arriba)
        self.guardia = None               # LA GUARDIA DE VERDAD DE LA VENTANA (w.enlace)


CREADAS, BORRADAS = [], []   # LAS CARPETAS DESECHABLES QUE ESTE PROCESO CREÓ Y LAS QUE BORRÓ (test_ex07 LAS COTEJA)


def carpeta_desechable():
    """Una carpeta nueva y vacía, fuera del repo y de las carpetas de la persona."""
    carpeta = tempfile.mkdtemp(prefix="cosechador-exploracion-")
    CREADAS.append(carpeta)
    return carpeta


def borrar_desechable(carpeta):
    base = os.path.realpath(tempfile.gettempdir())
    real = os.path.realpath(carpeta)
    if not real.startswith(base + os.sep):   # NUNCA BORRA FUERA DE LA CARPETA TEMPORAL
        raise RuntimeError(f"No borro {carpeta!r}: no está dentro de {base!r}.")
    shutil.rmtree(real, ignore_errors=True)
    BORRADAS.append(carpeta)


def ventana_aislada(robot, carpeta):
    """La ventana REAL, armada como en la app pero encerrada en `carpeta` (que debe ser desechable)."""
    from cosechador.movimiento import programa as prog
    from cosechador.ui.ventana import VentanaPrincipal
    base = os.path.realpath(tempfile.gettempdir())
    if not os.path.realpath(carpeta).startswith(base + os.sep):
        raise RuntimeError(f"La carpeta de trabajo {carpeta!r} no es desechable (no está en {base!r}).")
    programas = os.path.join(carpeta, "programas")
    os.makedirs(programas, exist_ok=True)
    ajustes = QtCore.QSettings(os.path.join(carpeta, "ajustes.ini"), QtCore.QSettings.IniFormat)
    x = Estado(carpeta)
    # EL RELOJ FALSO ENTRA DESDE EL PRINCIPIO: LA GUARDIA DE LA VENTANA LO USA PARA LA POSE CON LA QUE ARRANCA
    w = VentanaPrincipal(robot, ajustes=ajustes, carpeta_programas=programas, registro=reg.Registro(carpeta=None), reloj=x.reloj)
    assert os.path.realpath(w.carpeta_programas) != os.path.realpath(prog.CARPETA_PROGRAMAS)
    w.x = x
    w.grabadora = Grabadora(reloj=x.reloj)
    w.registro.suscribir(x.eventos.append)
    _espiar_la_placa(w)
    _contestar_los_dialogos(w)
    return w


def _espiar_la_placa(w):
    x = w.x
    x.guardia = w.enlace
    # LA POSE CON LA QUE ARRANCÓ LA VENTANA YA SALIÓ... Y SALIÓ HACE RATO: NADIE TOCA UN CONTROL EN EL MISMO MILISEGUNDO EN QUE
    # ABRE LA VENTANA. SIN ESTO EL RELOJ FALSO NO AVANZA Y EL PRIMER PASO DEL PRIMER GESTO «SALE» 1 ms DESPUÉS DE LA POSE
    # INICIAL: LA GUARDIA LO NIEGA POR RITMO Y ES MENTIRA
    x.reloj.t += SEGUNDOS_TRAS_ABRIR
    crudo = w.enlace._enlace
    al_crudo = crudo.enviar

    def a_la_placa(*servos):
        x.enviados.append((x.reloj(), tuple(servos), w._origen_plan))   # LO QUE DE VERDAD LLEGÓ A LA PLACA
        return al_crudo(*servos)

    crudo.enviar = a_la_placa
    por_la_guardia = w.enlace.enviar

    def enviar(*servos):
        try:
            return por_la_guardia(*servos)
        except OrdenNegada as e:
            if e.nueva:       # SOLO LA PRIMERA NEGATIVA CUENTA: DESPUÉS LA GUARDIA DETENIDA REPITE LA MISMA
                x.negadas.append((x.reloj(), e.violacion, tuple(servos)))
            raise

    w.enlace.enviar = enviar


def teletransportar(w, servos):
    """Deja al robot en `servos` de un golpe, sin pasos de por medio, como si la app hubiera arrancado ahí. NO es algo que la
    app haga: es para las pruebas que necesitan una pose de salida cualquiera. La guardia (con razón) niega un brinco, así que
    aquí se le borra la memoria de lo último que salió: la primera orden de una guardia sin historia solo se revisa de forma,
    cantidad, números y rango. Es lo que le pasaría a la app al abrirse en esa pose. (La guardia NO tiene una forma pública de
    olvidar: hacerla para las pruebas sería dejarla para cualquiera.)"""
    como_recien_abierta(w)
    return w.poner_servos(servos)


def como_recien_abierta(w):
    """Borra la memoria de la guardia: lo siguiente que salga es, para ella, lo primero de una app recién abierta. Para las
    pruebas que llaman `_ir_a_home` a media sesión (la app solo lo hace al abrir)."""
    g = w.enlace
    g._ultima = g._hora_ultima = g._detenida = None
    g._deuda_ms = 0.0


def _contestar_los_dialogos(w):
    """Los TRES diálogos de la ventana (el del nombre, el de sobrescribir y las preguntas de sí/no) se contestan con lo
    que haya en las colas de `w.x`; sin respuesta, lo que no destruye (cancelar / no)."""
    x = w.x

    def pedir_nombre(sugerido, tipo="grabacion"):
        respuesta = x.nombres.popleft() if x.nombres else None
        if respuesta == DESCARTAR_NOMBRE and tipo != "grabacion":
            respuesta = None          # EL DIÁLOGO REAL SOLO TRAE «Descartar…» AL GUARDAR UNA GRABACIÓN; EN LOS DEMÁS, «Cancelar»
        x.dialogos.append(("nombre", f"{tipo}: {sugerido}", respuesta))
        if respuesta == DESCARTAR_NOMBRE:
            from cosechador.ui.ventana import DESCARTAR
            return DESCARTAR
        return respuesta

    def confirmar(titulo, pregunta, si, no):
        respuesta = bool(x.confirmas.popleft()) if x.confirmas else False
        x.dialogos.append(("confirmar", f"{titulo}: {pregunta}", respuesta))
        return respuesta

    def confirmar_sobrescribir(archivo, tipo="grabacion"):
        respuesta = bool(x.confirmas.popleft()) if x.confirmas else False
        x.dialogos.append(("sobrescribir", archivo, respuesta))
        return respuesta

    w._pedir_nombre, w._confirmar, w._confirmar_sobrescribir = pedir_nombre, confirmar, confirmar_sobrescribir


def cerrar(w):
    w.close()
    APP.processEvents()


@contextmanager
def sin_dialogos_de_verdad():
    """Mientras dura: ningún diálogo modal puede abrirse (levanta en vez de colgar el proceso) y una excepción dentro de un
    slot de Qt NO aborta el proceso (PyQt5 lo hace si nadie atrapa): se anota en EXCEPCIONES."""
    def no_abre(*_a, **_k):
        raise AssertionError("se intentó abrir un diálogo modal de verdad: le falta su gancho a la ventana")
    parches = [(QtWidgets.QDialog, "exec_"), (QtWidgets.QMessageBox, "exec_"), (QtWidgets.QFileDialog, "getSaveFileName"),
               (QtWidgets.QFileDialog, "getOpenFileName"), (QtWidgets.QFileDialog, "getExistingDirectory"),
               (QtWidgets.QInputDialog, "getText"), (QtWidgets.QMessageBox, "question"),
               (QtWidgets.QMessageBox, "warning"), (QtWidgets.QMessageBox, "information"),
               (QtWidgets.QMessageBox, "critical")]
    guardado = [(clase, nombre, clase.__dict__.get(nombre)) for clase, nombre in parches]
    gancho_anterior = sys.excepthook

    def gancho(tipo, valor, tb):
        EXCEPCIONES.append((tipo, valor, tb))

    for clase, nombre in parches:
        setattr(clase, nombre, staticmethod(no_abre) if nombre.startswith(("get", "quest", "warn", "info", "crit")) else no_abre)
    sys.excepthook = gancho
    EXCEPCIONES.clear()
    try:
        yield
    finally:
        sys.excepthook = gancho_anterior
        for clase, nombre, original in guardado:
            if original is None:
                try:
                    delattr(clase, nombre)
                except AttributeError:
                    pass
            else:
                setattr(clase, nombre, original)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EL TEMPORIZADOR, CON EL RELOJ FALSO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _sincronizar_vencimiento(w):
    """Si el temporizador acaba de arrancar (la app lo arranca a 1 ms), su primer disparo es 1 ms después de ahora; si
    se detuvo, ya no hay disparo pendiente."""
    t, x = w.temporizador_movimiento, w.x
    if not t.isActive():
        x.vence = None
    elif x.vence is None:
        x.vence = x.reloj.t + t.interval() / 1000


def _disparar(w):
    """Un disparo del temporizador, a su hora."""
    x = w.x
    x.reloj.t = max(x.reloj.t, x.vence)
    w._siguiente_paso()
    t = w.temporizador_movimiento
    # setInterval REARRANCA EL TEMPORIZADOR: EL SIGUIENTE DISPARO ES UN INTERVALO DESPUÉS DE ESTE
    x.vence = x.reloj.t + t.interval() / 1000 if t.isActive() else None


def avanzar(w, segundos):
    """Deja pasar `segundos` de tiempo falso: el temporizador dispara los pasos que le toquen en ese rato."""
    x = w.x
    _sincronizar_vencimiento(w)
    fin = x.reloj.t + segundos
    while w.temporizador_movimiento.isActive() and x.vence <= fin + 1e-12:
        _disparar(w)
    x.reloj.t = max(x.reloj.t, fin)
    _sincronizar_vencimiento(w)


def correr(w, tope=TOPE_PASOS):
    """Deja que el movimiento TERMINE (llega o se acaba el plan). Levanta NoTermina si no acaba en `tope` pasos."""
    _sincronizar_vencimiento(w)
    pasos = 0
    while w.temporizador_movimiento.isActive():
        _disparar(w)
        pasos += 1
        if pasos > tope:
            raise NoTermina(f"el movimiento sigue pidiendo pasos después de {tope} (cola: {len(w.cola)})")
