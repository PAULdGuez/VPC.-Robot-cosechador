"""Ventana principal: conecta los controles del .ui con el robot, la visión y el enlace.

Aquí no se calcula nada: los movimientos los planea cosechador.movimiento y esta ventana solo
ejecuta el plan paso a paso con un temporizador (sin bucles que bloqueen la interfaz).
"""
import dataclasses
import datetime as dt
import math
import os
import time
from collections import deque

import html
import json

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtGui import QImage, QPixmap
from pyvistaqt import QtInteractor

from cosechador import registro as reg
from cosechador.hardware.enlace import abrir_enlace
from cosechador.hardware.guardia import GuardiaDeSalida, OrdenNegada
from cosechador.movimiento import programa as programas
from cosechador.movimiento.grabadora import Grabadora, programa_de_grabacion
from cosechador.movimiento.trayectorias import (duracion_ms, hasta_donde_llega, plan_cosecha, plan_grabacion,
                                                plan_ir_y_volver, plan_puntos, posicion, tramo_a_punto,
                                                tramo_articular, tramo_continuo)
from cosechador.robot.cinematica import servo_a_dh
from cosechador.robot.cinematica_inversa import TOLERANCIA_TRAMOS_MM, alcanzable, tramos_alcanzables
from cosechador.robot.espacio_trabajo import barrer, contorno_del_corte, giro_de_la_base
from cosechador.ui import tema
from cosechador.ui.escena3d import Escena
from cosechador.ui.visor import Visor
from cosechador.vision.camara import pedunculo_en_base
from cosechador.vision.deteccion import detectar_aguacates

RUTA_UI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "COSECHADOR_AGUACATE.ui")

ICONO_NIVEL = {"info": "●", "ok": "✔", "aviso": "⚠", "error": "✖"}
COLOR_NIVEL = {"info": "#2c6fbb", "ok": "#1e8e3e", "aviso": "#b7791f", "error": "#c0392b"}
FILTROS_REGISTRO = (("Todos", reg.NIVELES), ("Avisos y errores", ("aviso", "error")), ("Solo errores", ("error",)))

AMARILLO, ROJO = (0, 255, 255), (0, 0, 255)
# LAS COLUMNAS QUE SE CONTRAEN A UNA FRANJA: (SU FONDO, SU FRANJA, EL AJUSTE DONDE SE RECUERDA CÓMO QUEDÓ)
COLUMNAS = {"vision": ("FONDO_VISION", "FRANJA_VISION", "vision/contraida"),
            "acciones": ("FONDO_ACCIONES", "FRANJA_ACCIONES", "acciones/contraida")}
AJUSTE_NUMEROS = "rejilla/numeros"  # SI LOS NÚMEROS DE LA REJILLA 3D ESTÁN ENCENDIDOS


def cv_a_pixmap(imagen_cv):
    rgb = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2RGB)
    alto, ancho, canales = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, ancho, alto, canales * ancho, QImage.Format_RGB888))


def abrir_camara(fuente):
    """Abre la cámara (URL de la app del celular o índice de webcam). None si no hay o no abre."""
    if not fuente:
        return None
    try:
        fuente = int(fuente)  # ÍNDICE DE WEBCAM ("0", "1"...); SI NO ES NÚMERO, ES UNA URL
    except ValueError:
        pass
    video = cv2.VideoCapture(fuente)
    return video if video.isOpened() else None


POSTURAS = (("despacio", "Al cambiar de postura: girar despacio"), ("negar", "Al cambiar de postura: negarse"))
PESTANA_SERVOS, PESTANA_XYZ, PESTANA_PROGRAMA = 0, 1, 2
MM_POR_MUESTRA_JOG = 5.0  # EN MODO XYZ, LA RECTA SE MUESTREA CADA 5 mm
NOMBRES_EJES = "XYZ"
# QUÉ HACE LA PUNTA SI SE MUEVE OTRO SLIDER DE «PUNTA X Y Z» MIENTRAS VA EN CAMINO: (CLAVE, TEXTO DEL SELECTOR)
EN_CAMINO = (("reemplazar", "Reemplazar"), ("en_cola", "En cola"), ("cancelar", "Cancelar"))
EN_CAMINO_AL_ABRIR = "en_cola"
AJUSTE_EN_CAMINO = "jog/en_camino"  # DONDE SE RECUERDA EL MODO ELEGIDO
AYUDA_EN_CAMINO = ("Qué hace la punta si se mueve otro slider mientras va en camino:\n"
                   "• Reemplazar: deja lo que hacía y va en línea recta al punto que marcan los tres sliders.\n"
                   "• En cola: termina el tramo que lleva y después recorre los siguientes, en el orden pedido.\n"
                   "• Cancelar: se detiene donde va y recorre solo el tramo nuevo.")
SIN_EJE = -1  # EN EL ÚLTIMO TRAMO DEL JOG: NO ES DE NINGÚN SLIDER (CUALQUIER ORDEN NUEVA «EN COLA» VA DETRÁS DE LO PENDIENTE)
PASOS_POR_REPINTADO_JOG = 3  # CON LA PUNTA EN CAMINO, EL ROJO DE LOS SLIDERS SE VUELVE A CALCULAR CADA TANTOS PASOS
MISMA_RECTA_MM = 0.05        # DOS RECTAS QUE DIFIEREN MENOS QUE ESTO PINTAN EL MISMO ROJO: NO SE VUELVE A CALCULAR
TEXTO_GRABAR, TEXTO_DETENER_GRABACION = "⏺ Grabar", "⏹ Detener grabación"
TEXTO_VER_VOLUMEN, TEXTO_OCULTAR_VOLUMEN = "VOLUMEN DE TRABAJO", "OCULTAR VOLUMEN"
# PEDIR NOMBRE Y GUARDAR, SEGÚN EL TIPO DE PROGRAMA: (TÍTULO DEL DIÁLOGO, ETIQUETA DEL CAMPO, ORIGEN EN EL REGISTRO)
TEXTOS_GUARDAR = {"grabacion": ("Guardar grabación", "Nombre de la grabación:", "Grabación"),
                  "puntos": ("Guardar programa", "Nombre del programa:", "Programa")}
DESCARTAR = object()        # LO QUE DEVUELVE EL DIÁLOGO DEL NOMBRE CUANDO SE PIDE TIRAR LA GRABACIÓN
RESULTADO_DESCARTAR = 2     # CÓDIGO DEL DIÁLOGO (0 = RECHAZADO, 1 = ACEPTADO)
ANCHO_DEL_NOMBRE = 320      # px DEL CAMPO: CABE ENTERO EL NOMBRE QUE SE SUGIERE (grabacion-AAAA-MM-DD-HHMM)
TEXTO_GUARDIA_TITULO = "MOVIMIENTO DETENIDO"
TEXTO_GUARDIA_RESTO = "No se mandó esa orden: el robot sigue donde estaba. Revise y pulse Reiniciar para volver a moverlo."
PAUSA_AVISO_GUARDIA_S = 0.5  # DETENIDA, FUERA DE UN ARRASTRE (RUEDA, FLECHAS) NO HAY MÁS DE UN AVISO CADA TANTO: NUNCA UNA RACHA
TEXTO_NUEVO_PROGRAMA = "+ Nuevo programa"  # EL ÚLTIMO RENGLÓN DEL SELECTOR, EL ÚNICO QUE NO ES UN ARCHIVO
# EDITOR DE PUNTOS
MOTIVO_NO_SE_EDITA = "Una grabación son muestras por tiempo:\nno se edita por puntos."
INVITACION_SIN_PUNTOS = ("Sin puntos.\n\nLleva el robot a una pose con las pestañas Servos o Punta X Y Z "
                         "y pulsa «+ Punto»: se guarda la pose actual.")
PASO_VELOCIDAD_PUNTO = 5  # LA VELOCIDAD DE UN PUNTO SE ELIGE DE 5 EN 5 %
MARCA_IMPOSIBLE = "imposible"  # LO QUE LLEVA EN UserRole EL RENGLÓN DE UN PUNTO AL QUE NO SE PUEDE LLEGAR


def muestras_del_jog(distancia_mm):
    """En cuántas muestras se parte una recta de la punta: una cada MM_POR_MUESTRA_JOG. La usan la orden del jog y el rojo
    de «negarse», para que los dos le pregunten lo mismo al planeador."""
    return max(2, int(distancia_mm / MM_POR_MUESTRA_JOG) + 1)


def _marca(slider, valor):
    """El valor con el que un slider marca `valor`: un entero dentro de su rango."""
    return min(max(int(round(valor)), slider.minimum()), slider.maximum())


def _cuantos_puntos(n):
    """«sin puntos», «1 punto», «4 puntos»: el conteo en palabras. Lo dicen igual la pestaña Programa y el editor."""
    return "sin puntos" if not n else "1 punto" if n == 1 else f"{n} puntos"


class _RenglonDelRegistro(QtWidgets.QStyledItemDelegate):
    """El alto de cada renglón se calcula SIEMPRE con el ancho real de la lista.

    QListView lo calculaba con un ancho viejo (antes de acomodar la ventana, o antes de que apareciera la
    barra de desplazamiento): unos mensajes salían cortados con «…» y otros dejaban una línea en blanco.
    """

    def sizeHint(self, opcion, indice):
        rect = QtCore.QRect(opcion.rect)
        # MENOS EL MARGEN INTERNO DEL RENGLÓN (4 + 4 px EN tema.py) Y 2 px DE HOLGURA: Qt MIDE EL ALTO SIN
        # DESCONTARLO, PERO AL DIBUJAR SÍ LO DESCUENTA, Y EL MENSAJE QUE "CABÍA" SALÍA CORTADO CON «…»
        rect.setWidth(self.parent().viewport().width() - 10)
        opcion.rect = rect
        return super().sizeHint(opcion, indice)


class _RenglonDePunto(_RenglonDelRegistro):
    """El punto imposible sigue en rojo AUNQUE esté elegido.

    La hoja del tema pone en negro todo renglón elegido (QListWidget::item:selected) y el rojo se perdía justo
    al elegir el punto para arreglarlo. Aquí el fondo de «elegido» se pinta a mano y el texto se queda con el
    color del renglón. Mismo rojo y mismo fondo del tema: no es un estilo nuevo.
    """

    def paint(self, pintor, opcion, indice):
        elegido = int(opcion.state) & int(QtWidgets.QStyle.State_Selected)
        if elegido and indice.data(QtCore.Qt.UserRole) == MARCA_IMPOSIBLE:
            pintor.fillRect(opcion.rect, QtGui.QColor(tema.TOKENS["seleccion"]))
            opcion = QtWidgets.QStyleOptionViewItem(opcion)
            opcion.state = QtWidgets.QStyle.State(int(opcion.state) & ~int(QtWidgets.QStyle.State_Selected))
        super().paint(pintor, opcion, indice)


class VentanaPrincipal(QtWidgets.QMainWindow):

    def __init__(self, robot, enlace=None, fuente_camara=None, ajustes=None, carpeta_programas=None, registro=None,
                 reloj=time.monotonic):
        super().__init__()
        uic.loadUi(RUTA_UI, self)
        tema.aplicar(self)  # COLORES, TAMAÑOS Y ESPACIOS: TODO EN cosechador/ui/tema.py
        self.robot = robot
        self.registro = registro if registro is not None else reg.Registro()
        self._origen_plan = "Movimiento"
        self._configurar_registro()
        # AJUSTES DE LA PERSONA (P. EJ. EL ÚLTIMO PROGRAMA ELEGIDO): QSettings ES LO NATIVO DE Qt
        self.ajustes = ajustes if ajustes is not None else QtCore.QSettings("robot-cosechador", "cosechador")
        self.carpeta_programas = carpeta_programas or programas.CARPETA_PROGRAMAS
        # ESCENA 3D INCRUSTADA EN EL RECUADRO "PANTALLA" DEL .ui, DENTRO DEL VISOR CON PESTAÑAS (3D, LATERAL, SUPERIOR, FICHA)
        self.escena = Escena(QtInteractor(self), robot)
        self.visor = Visor(self.PANTALLA, self.escena, robot, self.ajustes)

        # ESP32 REAL SI HAY ROBOT_PUERTO; SI NO, LA VIRTUAL. SIEMPRE DENTRO DE LA GUARDIA DE SALIDA: ES EL ÚNICO CAMINO A LA
        # PLACA (SIN PARÁMETRO NI AJUSTE QUE LA QUITE). EL RELOJ SOLO SE INYECTA PARA QUE LAS PRUEBAS COMPARTAN EL SUYO FALSO:
        # LA GUARDIA LO LEE SIEMPRE DE `self.reloj`, ASÍ QUE CAMBIARLO DESPUÉS TAMBIÉN VALE
        crudo = enlace or abrir_enlace(rangos=[(j.servo.minimo, j.servo.maximo) for j in robot.articulaciones])
        self.reloj = reloj
        self.enlace = GuardiaDeSalida(crudo, robot, reloj=lambda: self.reloj())
        self._gesto_avisado = False          # YA SE AVISÓ «SIGUE DETENIDO» EN ESTE ARRASTRE: NO SE REPITE HASTA SOLTAR EL SLIDER
        self._hora_aviso_guardia = -math.inf  # CUÁNDO FUE EL ÚLTIMO AVISO DE «SIGUE DETENIDO» (LO QUE NO ES UN ARRASTRE)
        self._configurar_guardia()
        for respuesta in self.enlace.dispositivo.leer_respuestas():
            self.avisar(respuesta, "info", "ESP32")
        # LA BARRA DE ARRIBA DICE QUÉ ROBOT ESTÁ CARGADO Y A QUÉ PLACA SE LE MANDAN LOS ÁNGULOS
        self.ESTADO_ROBOT.setText(robot.nombre)
        color = "#f1c40f" if "virtual" in self.enlace.nombre.lower() else "#2ecc71"
        self.ESTADO_CONEXION.setText(f"<span style='color:{color}'>●</span> {html.escape(self.enlace.nombre)}")

        # ESTADO
        self.servos = None
        self.puntos_cosecha = list(robot.puntos_demo)
        self.guardar_en_siguiente_frame = False
        self.imagen_fija = None
        self.cola = deque()
        self._destino_jog = None      # PUNTO EXACTO DONDE TERMINA EL JOG DE LA PUNTA QUE VA EN CAMINO (None: NO HAY)
        self._tramo_jog = None        # SU ÚLTIMO TRAMO: (EJE, CUÁNTOS PASOS, SERVOS DE SALIDA, PUNTO DE SALIDA)
        self._pasos_jog = 0           # PASOS DADOS DESDE QUE SE PUSO AL DÍA EL ROJO POR UNA ORDEN: SE REPINTA CADA POCOS
        self._rectas_pintadas = [None, None, None]  # LA RECTA SOBRE LA QUE SE PINTÓ EL ROJO DE CADA SLIDER DE LA PUNTA
        self.grabadora = Grabadora()  # ANTES DEL PRIMER poner_servos: AHÍ SE ENGANCHA LA GRABACIÓN
        self.tipo_elegido = None      # DEL PROGRAMA ELEGIDO EN EL SELECTOR: "puntos" | "grabacion" (None: NINGUNO, O ILEGIBLE)
        self.editado = None           # EL Programa ABIERTO EN EL EDITOR DE PUNTOS (None: EDITOR CERRADO)
        self.archivo_editado = None   # SU ARCHIVO, DENTRO DE carpeta_programas
        self._vision_contraida_antes = False
        self.ocupado = False          # UN MOVIMIENTO COMPLETO EN CURSO (PROGRAMA, PASO, VISTA PREVIA, COSECHA, CALCULAR)
        self.en_vista_previa = False  # LO QUE HAY EN LA COLA SOLO SE DIBUJA: A LA PLACA NO SE LE MANDA
        self.temporizador_movimiento = QtCore.QTimer(self)
        self.temporizador_movimiento.setTimerType(QtCore.Qt.PreciseTimer)  # LAS GRABACIONES RESPETAN SUS TIEMPOS
        self.temporizador_movimiento.timeout.connect(self._siguiente_paso)

        self._configurar_selectores()
        self._configurar_esquina()
        self._configurar_sliders()
        self._configurar_editor()
        self._configurar_programas()
        self._configurar_vision()
        self._configurar_acciones()
        self._configurar_visor()
        self.CHK_TRAYECTORIA.toggled.connect(self.escena.mostrar_trayectoria)
        self.BTN_LIMPIAR_TRAYECTORIA.clicked.connect(lambda: self.escena.limpiar_trayectoria(self.posicion))
        for boton, nombre in ((self.BTN_VISTA_ISO, "iso"), (self.BTN_VISTA_ARRIBA, "arriba"),
                              (self.BTN_VISTA_FRENTE, "frente"), (self.BTN_VISTA_LADO, "lado")):
            boton.clicked.connect(lambda _=False, n=nombre: self.escena.vista(n))
        self.BTN_ENCUADRAR.clicked.connect(lambda: self.escena.encuadrar())
        self.CALCULAR.clicked.connect(self.ir_y_volver)
        self.COSECHAR.clicked.connect(self.cosechar)
        self.VER_VOLUMEN.clicked.connect(self.ver_volumen)
        self.LARGO_HERRAMIENTA.setText(f"{robot.efector.largo:g}")  # EL LARGO LO DICE EL ARCHIVO DEL ROBOT: AQUÍ SOLO SE ENSEÑA

        self._ir_a_home()
        self.escena.reiniciar_recorrido(self.posicion)
        self.escena.encuadrar()

        self.video = abrir_camara(fuente_camara if fuente_camara is not None else os.environ.get("ROBOT_CAMARA"))
        self.temporizador_video = QtCore.QTimer(self)
        self.temporizador_video.timeout.connect(self._actualizar_frame)
        if self.video is not None:
            self.temporizador_video.start(30)
        else:
            self.ORIGINAL.setText("Sin cámara.\nDefine ROBOT_CAMARA o usa\nCARGAR IMAGEN.")

        self.avisar(f"{robot.nombre} · conectado a: {self.enlace.nombre}", "ok", "ESP32",
                    robot=robot.nombre, placa=self.enlace.nombre, registro=self.registro.archivo)
        # QUE NINGÚN CAMPO ARRANQUE "ACTIVO": EL FOCO INICIAL VA AL FONDO (Y UN CLIC EN EL FONDO SUELTA EL CAMPO)
        self.centralwidget.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.centralwidget.setFocus()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ESTADO DEL BRAZO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    @property
    def posicion(self):
        return tuple(posicion(self.robot, self.servos))

    @property
    def en_movimiento(self):
        return self.temporizador_movimiento.isActive()

    def avisar(self, texto, nivel="info", origen="Sistema", **datos):
        """Todo aviso queda en el registro (con nivel, origen y datos) y el último se ve abajo."""
        evento = self.registro.registrar(nivel, origen, texto, **datos)
        print(f"{evento.hora:%H:%M:%S} {nivel.upper():5} {origen}: {texto}")
        self.statusBar().showMessage(f"{ICONO_NIVEL[nivel]} {texto}")

    def poner_servos(self, servos, duracion_s=None):
        """Manda a la placa esos ángulos de servo, los dibuja y devuelve True. Si la guardia de salida no los deja salir,
        no se dibuja nada, `self.servos` no cambia, nada se graba y devuelve False: quien llama no debe seguir su plan.

        duracion_s: lo que dura este paso (para la grabación); None = un paso a velocidad 100 %.
        """
        pedido = tuple(servos)
        try:
            comando, respuestas = self.enlace.enviar(*pedido)  # LA ORDEN VA ANTES DE DIBUJAR: SI SE NIEGA, EL ROBOT NO SE MOVIÓ
        except OrdenNegada as negada:                          # EL ÚNICO LUGAR QUE ATIENDE LA NEGATIVA DE LA GUARDIA
            self._guardia_se_nego(negada, pedido)
            return False
        self.servos = pedido
        self._dibujar(self.servos)
        self.poner_etiquetas()
        # SE GRABA AQUÍ, JUSTO DESPUÉS DE MANDARLA: LO GRABADO ES LO QUE RECIBIÓ LA PLACA, NO LO QUE DICEN LOS SLIDERS
        self.grabadora.anotar(self.servos, duracion_s if duracion_s is not None else self.robot.movimiento.ms_por_paso / 1000)
        for respuesta in respuestas:
            self.avisar(respuesta, "aviso", "ESP32", comando=comando, placa=self.enlace.nombre)
        return True

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GUARDIA DE SALIDA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def _configurar_guardia(self):
        """La franja roja bajo el título está en el .ui, oculta: aquí solo se conecta su botón y se le pone el texto fijo."""
        self.BTN_REINICIAR_GUARDIA.clicked.connect(self._reiniciar_guardia)
        self.TITULO_GUARDIA.setText(TEXTO_GUARDIA_TITULO)

    def _guardia_se_nego(self, negada, pedido):
        """La guardia no dejó salir la orden: lo que faltaba del movimiento se cancela, los sliders y las etiquetas vuelven a
        donde está el robot (que no se movió), aparece la franja y se avisa. La primera negativa es un error en el registro;
        mientras siga detenida, cada intento nuevo de mover avisa (una vez por gesto) que sigue detenida."""
        violacion = negada.violacion
        if self.servos is not None:                # (SI NI LA POSE DE ARRANQUE SALIÓ, NO HAY DONDE VOLVER)
            self._fin_del_movimiento()             # COLA VACÍA, TEMPORIZADOR PARADO, SLIDERS DONDE ESTÁ EL ROBOT
            self.poner_etiquetas()
        else:
            self.cola.clear()
            self.temporizador_movimiento.stop()
        self.TEXTO_GUARDIA.setText(f"{violacion.motivo} {TEXTO_GUARDIA_RESTO}")
        self.FRANJA_GUARDIA.show()
        if negada.nueva:
            self.avisar(f"{violacion.motivo} No se mandó esa orden: el robot sigue donde estaba.", "error", "Guardia",
                        clave=violacion.clave, motivo=violacion.motivo, servos=pedido, servo=violacion.servo,
                        pedido=violacion.pedido, tope=violacion.tope, detalle=violacion.detalle)
            self._gesto_avisado = self._arrastrando()
        else:
            self._avisar_sigue_detenida(violacion)

    def _avisar_sigue_detenida(self, violacion):
        """Un intento de mover con la guardia detenida. Un aviso por gesto: un arrastre completo de un slider es UN gesto
        (no uno por pixel); fuera de un arrastre (rueda, flechas, botones) no hay más de uno cada PAUSA_AVISO_GUARDIA_S."""
        ahora = self.reloj()
        if self._arrastrando():
            if self._gesto_avisado:
                return
            self._gesto_avisado = True
        elif ahora - self._hora_aviso_guardia < PAUSA_AVISO_GUARDIA_S:
            return
        self._hora_aviso_guardia = ahora
        self.avisar(f"El movimiento sigue detenido: {violacion.motivo} Pulse Reiniciar para volver a moverlo.", "aviso",
                    "Guardia", clave=violacion.clave)

    def _gesto_de_slider_termina(self):
        """Se soltó un slider: el siguiente arrastre es otro gesto. Si la guardia se negó con la perilla agarrada, los
        sliders no se tocaron (no se le quita a la persona lo que sostiene): ahora que la soltó, vuelven al robot."""
        self._gesto_avisado = False
        if self.enlace.detenida:
            self._sincronizar_sliders()

    def _reiniciar_guardia(self):
        """Lo único que levanta el candado, y solo lo pulsa una persona: no mueve el robot ni reenvía la orden negada."""
        self.enlace.reiniciar()
        self.FRANJA_GUARDIA.hide()
        self._gesto_avisado = False
        self.avisar("Guardia reiniciada: no se mandó nada y el robot sigue donde estaba. Ya se puede volver a mover.", "info",
                    "Guardia")

    def _dibujar(self, servos):
        """El robot de la simulación con esos ángulos de servo. Solo el dibujo: a la placa no le llega nada."""
        self.escena.dibujar(servo_a_dh(self.robot, servos))
        self.visor.poner_servos(servos)  # LA LATERAL Y LA SUPERIOR SE MUEVEN CON LA 3D, POR CUALQUIER MEDIO QUE MUEVA EL ROBOT
        self.escena.agregar_al_recorrido(tuple(posicion(self.robot, servos)))

    def _ir_a_home(self):
        servos = self.robot.postura_home
        if servos is None:  # SIN SOLVER (O HOME INALCANZABLE): CADA SERVO AL CENTRO DE SU RANGO
            servos = tuple(j.servo.centro for j in self.robot.articulaciones)
        if self.poner_servos(servos):
            self._sincronizar_sliders()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EJECUCIÓN DE PLANES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def ejecutar(self, plan, bloquear=True, origen="Movimiento", solo_simulacion=False, conservar=0):
        """Encola los pasos del plan; el temporizador los ejecuta uno por uno.

        bloquear=False (jog con los sliders): no se bloquean los controles y un movimiento nuevo
        reemplaza lo que faltaba del anterior, salvo sus primeros `conservar` pasos (el modo «En cola» de la punta).
        solo_simulacion=True (vista previa): los pasos se dibujan y a la placa no se le manda nada.
        """
        if self.en_vista_previa and not solo_simulacion:
            self._fin_del_movimiento()  # LO QUE FALTABA DE UNA VISTA PREVIA NUNCA SE MANDA A LA PLACA
        self.en_vista_previa = solo_simulacion
        self._destino_jog = self._tramo_jog = None  # TODO MOVIMIENTO NUEVO OLVIDA EL JOG PENDIENTE (EL JOG LO VUELVE A ANOTAR)
        if origen != "Jog":
            self._aviso_jog_dado = None             # EL ROBOT SE MUEVE POR OTRO LADO: EL AVISO DEL JOG VUELVE A VALER
        if not bloquear:
            while len(self.cola) > conservar:
                self.cola.pop()
        self.cola.extend(plan)
        # LO QUE VA A RECORRER LA PUNTA, ANTES DE MOVERSE. UN MOVIMIENTO «COMPLETO» (PROGRAMA, COSECHA,
        # CALCULAR) EMPIEZA ESTELA NUEVA; EL JOG Y LOS SLIDERS LA SIGUEN ALARGANDO.
        camino = [self.posicion] + [posicion(self.robot, p.servos) for p in self.cola if p.servos is not None]
        self.escena.mostrar_plan(camino)
        if bloquear:
            self.escena.reiniciar_recorrido(self.posicion)
        self._origen_plan = origen
        self._ocupado(bloquear)
        self.BTN_DETENER.setEnabled(True)
        if not self.temporizador_movimiento.isActive():
            self.temporizador_movimiento.start(1)  # EL PRIMER PASO, YA

    def _siguiente_paso(self):
        if not self.cola:
            self._fin_del_movimiento()
            return
        paso = self.cola.popleft()
        # EL SIGUIENTE PASO LLEGA CUANDO TERMINA ESTE: SU DURACIÓN, ESCALADA POR LA VELOCIDAD GENERAL
        ms = paso.ms if paso.ms is not None else (self.robot.movimiento.ms_por_paso if paso.servos is not None else 0)
        intervalo = max(1, int(round(ms * 100 / self.velocidad)))
        if paso.servos is not None and self.en_vista_previa:
            self._dibujar(paso.servos)  # NI PLACA, NI GRABACIÓN, NI self.servos: EL ROBOT NO SE MOVIÓ
        elif paso.servos is not None:
            if not self.poner_servos(paso.servos, intervalo / 1000):
                return  # LA GUARDIA SE NEGÓ: YA SE CANCELÓ LO PENDIENTE Y NO HAY PLAN QUE SEGUIR
            if self._destino_jog is not None:  # LA PUNTA VA EN CAMINO POR UN SLIDER: SU ROJO SE PONE AL DÍA CADA POCOS PASOS
                self._pasos_jog += 1
                if self._pasos_jog % PASOS_POR_REPINTADO_JOG == 0:
                    self._pintar_alcance(solo_lo_cambiado=True)
        if paso.aviso is not None:
            self.avisar(paso.aviso, paso.nivel, self._origen_plan)
        self.temporizador_movimiento.setInterval(intervalo)

    def detener(self):
        """■ Detener: el robot se queda donde iba y no se manda nada más a la placa."""
        if not self.en_movimiento and not self.cola:
            return
        era_vista_previa = self.en_vista_previa
        self._fin_del_movimiento()
        if era_vista_previa:
            self.avisar("Vista previa detenida: el robot no se movió.", "aviso", self._origen_plan)
            return
        self.avisar(f"Detenido en {tuple(int(round(c)) for c in self.posicion)}.", "aviso", self._origen_plan,
                    posicion_mm=self.posicion, servos=self.servos)

    def _fin_del_movimiento(self):
        """Lo que pasa siempre que algo deja de moverse, porque terminó o porque se detuvo."""
        self.cola.clear()
        self.temporizador_movimiento.stop()
        if self.en_vista_previa:  # EL DIBUJO VUELVE A DONDE ESTÁ EL ROBOT DE VERDAD, QUE NO SE MOVIÓ
            self.en_vista_previa = False
            self.escena.dibujar(servo_a_dh(self.robot, self.servos))
            self.escena.reiniciar_recorrido(self.posicion)
        self._sincronizar_sliders()
        self._ocupado(False)
        self.BTN_DETENER.setEnabled(False)

    def _ocupado(self, si):
        """Bloquea los controles de movimiento mientras el robot se mueve (evita clics encimados)."""
        self.ocupado = si
        for control in (self.CALCULAR, self.COSECHAR, self.VER_VOLUMEN, self.BTN_REPRODUCIR,
                        self.SELECTOR_PROGRAMA, self.BTN_GRABAR, *self.sliders_servo, *self.sliders_xyz):
            control.setEnabled(not si)
        if self.SELECTOR_PROGRAMA.currentData() is None:  # SIN PROGRAMA ELEGIDO NO HAY QUÉ REPRODUCIR
            self.BTN_REPRODUCIR.setEnabled(False)
        if self.grabadora.grabando:  # MIENTRAS SE GRABA, SOLO SE MUEVE CON LOS SLIDERS
            for control in (self.CALCULAR, self.COSECHAR, self.BTN_REPRODUCIR, self.SELECTOR_PROGRAMA):
                control.setEnabled(False)
        self._actualizar_editor()  # LOS BOTONES DEL EDITOR SE APAGAN Y SE ENCIENDEN CON LO MISMO

    def _libre(self):
        """True si no corre un movimiento completo ni se está grabando: solo entonces se edita un programa.
        El movimiento de los sliders no cuenta: enseñar un punto es mover el robot y guardar."""
        return not self.ocupado and not self.grabadora.grabando

    def _grabando_avisa(self):
        """True (y avisa) si hay una grabación en curso: entonces no se reproduce, calcula ni cosecha."""
        if self.grabadora.grabando:
            self.avisar("Mientras se graba, el robot solo se mueve con los sliders.", "aviso", "Grabación")
        return self.grabadora.grabando

    def _puede_planear(self):
        if self._grabando_avisa():
            return False
        if self.robot.cinematica_inversa is None:
            self.avisar(f"'{self.robot.nombre}' no tiene solver de cinemática inversa: solo se mueve con los sliders.",
                        "aviso", "Movimiento")
            return False
        return not self.en_movimiento

    def ir_y_volver(self):
        """Botón CALCULAR: ir al punto escrito en X, Y, Z y regresar a HOME."""
        if not self._puede_planear():
            return
        try:
            destino = tuple(float(campo.text()) for campo in (self.TE_X, self.TE_Y, self.TE_Z))
            if not all(math.isfinite(c) for c in destino):  # «inf» Y «nan» SON NÚMEROS PARA float(), NO PARA EL ROBOT
                raise ValueError(destino)
        except ValueError:
            self.avisar("Escribe números en X, Y y Z (en mm).", "aviso", "Movimiento",
                        escrito=[campo.text() for campo in (self.TE_X, self.TE_Y, self.TE_Z)])
            return
        self.ejecutar(plan_ir_y_volver(self.robot, self.servos, destino, self.postura), origen="CALCULAR")

    def cosechar(self):
        """Botón COSECHAR: visitar cada pedúnculo guardado y regresar a HOME."""
        if not self._puede_planear():
            return
        plan, _ = plan_cosecha(self.robot, self.servos, self.puntos_cosecha, self.postura)
        self.ejecutar(plan, origen="Cosecha")

    def ver_volumen(self):
        """Botón del volumen de trabajo: lo dibuja; con el volumen a la vista, lo oculta."""
        if self.escena.actores_volumen:
            self.escena.ocultar_volumen()
            self.visor.ocultar_volumen()
            self.PROCESO.setValue(0)
        else:
            self._dibujar_volumen()
        self.VER_VOLUMEN.setText(TEXTO_OCULTAR_VOLUMEN if self.escena.actores_volumen else TEXTO_VER_VOLUMEN)

    def _dibujar_volumen(self):
        def avance(hechos, total):
            self.PROCESO.setRange(0, total)
            self.PROCESO.setValue(hechos)
            QtWidgets.QApplication.processEvents()
        lazos = contorno_del_corte(self.robot, progreso=avance)
        self.visor.poner_volumen(lazos)  # (SIN CONTORNO, LA LATERAL Y LA SUPERIOR SE QUEDAN CON EL ROBOT)
        if lazos is None:  # EL CORTE DE ESTE ROBOT NO SE PUEDE GIRAR (O NO ENCIERRA ÁREA): LA NUBE DE PUNTOS
            self.escena.mostrar_volumen(barrer(self.robot, paso=5, progreso=avance))
        else:
            self.escena.mostrar_superficie(lazos, giro_de_la_base(self.robot))
        self.escena.encuadrar(zoom=1.0)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% VELOCIDAD Y POSTURA (FUERA DE LAS PESTAÑAS) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def _configurar_selectores(self):
        for clave, texto in POSTURAS:
            self.SELECTOR_POSTURA.addItem(texto, clave)
        self.SELECTOR_POSTURA.setCurrentIndex([c for c, _ in POSTURAS].index(self.robot.movimiento.al_cambiar_postura))
        self.SELECTOR_POSTURA.currentIndexChanged.connect(self._postura_cambiada)
        self.SLIDER_VELOCIDAD.valueChanged.connect(self._velocidad_cambiada)
        self._velocidad_cambiada()
        if self.robot.cinematica_inversa is None:  # SIN SOLVER NO HAY JOG XYZ NI TRAYECTORIAS
            self.PESTANAS_MOVIMIENTO.setTabEnabled(PESTANA_XYZ, False)

    def _configurar_esquina(self):
        """La esquina de la barra de pestañas: el aviso de que se está grabando (a la vista en las tres pestañas, porque
        se graba moviendo desde las otras dos) y el selector «En camino» (solo junto a los sliders de la punta)."""
        self.aviso_grabando = QtWidgets.QLabel("⏺ Grabando")
        self.aviso_grabando.setProperty("rol", "grabando")
        self.aviso_grabando.setToolTip("Se detiene en la pestaña Programa")
        self.aviso_grabando.hide()
        self.etiqueta_en_camino = QtWidgets.QLabel("En camino")
        self.selector_en_camino = QtWidgets.QComboBox()
        self.selector_en_camino.setProperty("rol", "en-barra")
        for clave, texto in EN_CAMINO:
            self.selector_en_camino.addItem(texto, clave)
        guardado = self.selector_en_camino.findData(str(self.ajustes.value(AJUSTE_EN_CAMINO, EN_CAMINO_AL_ABRIR)))
        if guardado < 0:  # UN MODO QUE ESTA VERSIÓN NO CONOCE
            guardado = self.selector_en_camino.findData(EN_CAMINO_AL_ABRIR)
        self.selector_en_camino.setCurrentIndex(guardado)
        esquina = QtWidgets.QWidget()
        fila = QtWidgets.QHBoxLayout(esquina)
        fila.setContentsMargins(0, 0, 0, 0)
        fila.setSpacing(8)
        for control, nombre in ((self.aviso_grabando, "AVISO_GRABANDO"), (self.etiqueta_en_camino, "ETIQUETA_EN_CAMINO"),
                                (self.selector_en_camino, "SELECTOR_EN_CAMINO")):  # (SUS NOMBRES PARA EL AUDITOR)
            control.setObjectName(nombre)
            fila.addWidget(control)
        for control in (self.etiqueta_en_camino, self.selector_en_camino):
            control.setToolTip(AYUDA_EN_CAMINO)
        self.PESTANAS_MOVIMIENTO.setCornerWidget(esquina, QtCore.Qt.TopRightCorner)
        self.selector_en_camino.currentIndexChanged.connect(self._en_camino_cambiado)
        self.PESTANAS_MOVIMIENTO.currentChanged.connect(self._acomodar_esquina)
        self._acomodar_esquina()

    def _acomodar_esquina(self, *_):
        """El selector «En camino» solo acompaña a los sliders de la punta. Al tamaño mínimo de la ventana no caben el
        aviso de grabación, la etiqueta y el selector (faltan 25 px): mientras se graba se quita la etiqueta, y qué es
        ese selector lo sigue diciendo su ayuda."""
        en_xyz = self.PESTANAS_MOVIMIENTO.currentIndex() == PESTANA_XYZ
        self.selector_en_camino.setVisible(en_xyz)
        self.etiqueta_en_camino.setVisible(en_xyz and self.aviso_grabando.isHidden())

    @property
    def en_camino(self):
        """Qué hace la punta con una orden nueva mientras va en camino: "reemplazar" | "en_cola" | "cancelar"."""
        return self.selector_en_camino.currentData()

    def _en_camino_cambiado(self):
        self.ajustes.setValue(AJUSTE_EN_CAMINO, self.en_camino)
        self._pintar_alcance(solo_lo_cambiado=True)  # CON LA PUNTA EN CAMINO, CADA MODO MIDE EL ROJO DESDE OTRO PUNTO

    def _postura_cambiada(self, *_):
        """Cambió el modo «Al cambiar de postura». Al pasar a «negarse», lo pendiente que se aceptó en «girar despacio» ya no
        vale: se cancela desde el primer cambio de postura que no ha empezado. Y el rojo se pinta otra vez, completo: en
        «negarse» cubre también lo que pide cambiar de postura."""
        if self.postura == "negar":
            self._cancelar_desde_el_cambio_de_postura()
        self._pintar_alcance()

    def _cancelar_desde_el_cambio_de_postura(self):
        """Quita de lo pendiente el primer cambio de postura que no ha empezado y todo lo que venía detrás. Lo anterior se
        termina: la punta llega hasta donde iba a empezar el giro y ahí se queda, que es hasta donde «negarse» la habría
        dejado llegar. Un giro que ya empezó se termina (dejarlo a medias sería peor). En vista previa no hay nada que
        cancelar: el robot no se mueve."""
        corte = next((i for i, paso in enumerate(self.cola) if paso.cambio_de_postura), None)
        if corte is None or self.en_vista_previa:
            return
        cancelados = len(self.cola) - corte
        for _ in range(cancelados):
            self.cola.pop()
        servos_fin = self._servos_al_terminar()
        fin = tuple(float(c) for c in posicion(self.robot, servos_fin))
        if self._destino_jog is not None:  # ERA DE LOS SLIDERS DE LA PUNTA: SU DESTINO Y SU ÚLTIMO TRAMO SE ACORTAN A LO QUE QUEDA
            eje, pasos, servos_salida, salida = self._tramo_jog
            self._tramo_jog = ((eje, pasos - cancelados, servos_salida, salida) if pasos > cancelados
                               else (SIN_EJE, 0, servos_fin, fin))
            self._destino_jog = fin
            if not any(s.isSliderDown() for s in self.sliders_xyz):
                self._sincronizar(self.sliders_xyz, fin)
        self.escena.mostrar_plan([self.posicion] + [posicion(self.robot, p.servos) for p in self.cola if p.servos is not None])
        self.avisar("El modo ahora es «negarse»: se canceló el cambio de postura que estaba pendiente y lo que venía detrás. "
                    f"La punta se queda en {tuple(int(round(c)) for c in fin)}.", "aviso", "Postura",
                    cancelados=cancelados, queda_mm=fin)

    @property
    def postura(self):
        return self.SELECTOR_POSTURA.currentData()

    @property
    def velocidad(self):
        """Porcentaje de la velocidad máxima segura (10 a 100 %)."""
        return self.SLIDER_VELOCIDAD.value()

    def _velocidad_cambiada(self):
        # MENOS VELOCIDAD = MÁS TIEMPO ENTRE PASOS (LO CALCULA _siguiente_paso). EL SALTO POR PASO NO
        # CAMBIA: AL 100 % SE VA A LO MÁS RÁPIDO QUE AGUANTA EL SERVO Y NUNCA MÁS. APLICA DESDE EL PASO
        # SIGUIENTE, AUNQUE HAYA UN MOVIMIENTO EN CURSO.
        self.VS_VELOCIDAD.setText(f"{self.velocidad} %")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PESTAÑAS DE MOVIMIENTO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def _configurar_sliders(self):
        """Pestaña Servos: un slider por servo (grados). Pestaña Punta X Y Z: un slider por eje (mm)."""
        todos = (self.SLIDER_1, self.SLIDER_2, self.SLIDER_3)
        todas = (self.VS_1, self.VS_2, self.VS_3)
        n = min(self.robot.gdl, len(todos))
        if self.robot.gdl > len(todos):
            print(f"Aviso: la interfaz tiene {len(todos)} sliders y el robot {self.robot.gdl} articulaciones; "
                  f"las demás se quedan en el centro de su rango.")
        self.sliders_servo, self.valores_servo = todos[:n], todas[:n]
        for slider in todos[n:]:
            slider.setEnabled(False)
        for slider, etiqueta, j in zip(self.sliders_servo, (self.ETIQUETA_SERVO_1, self.ETIQUETA_SERVO_2, self.ETIQUETA_SERVO_3), self.robot.articulaciones):
            slider.setRange(int(j.servo.minimo), int(j.servo.maximo))
            etiqueta.setToolTip(j.nombre)
            slider.valueChanged.connect(self._slider_servo_movido)
            slider.sliderReleased.connect(self._gesto_de_slider_termina)

        self.sliders_xyz = (self.SLIDER_X, self.SLIDER_Y, self.SLIDER_Z)
        self.valores_xyz = (self.VS_X, self.VS_Y, self.VS_Z)
        self._poner_rangos_xyz()
        self._jog_avisado = False  # YA SE AVISÓ UN TOPE O UN «NO SE MUEVE» EN ESTE ARRASTRE: NO SE REPITE
        self._jog_de_paso_avisado = False  # YA SE AVISÓ EL PUNTO DE PASO EN ESTE ARRASTRE (LLEVA SU PROPIA CUENTA)
        self._aviso_jog_dado = None  # EL ÚLTIMO AVISO DEL JOG (SLIDER Y MOTIVO): NO SE REPITE HASTA QUE ALGO SE MUEVA
        self._jog_arrastrando = False
        for eje, slider in enumerate(self.sliders_xyz):
            slider.color_fuera = QtGui.QColor(COLOR_NIVEL["error"])  # EL ROJO DE LOS ERRORES, EL MISMO DEL REGISTRO
            slider.valueChanged.connect(lambda _valor, eje=eje: self._slider_xyz_movido(eje))
            slider.sliderPressed.connect(lambda: self._jog_arrastre(True))
            slider.sliderReleased.connect(lambda: self._jog_arrastre(False))
            slider.sliderReleased.connect(self._gesto_de_slider_termina)

    def _poner_rangos_xyz(self):
        """Los sliders de la punta llegan hasta donde llega la punta de la herramienta. Con las señales calladas:
        si el rango se encoge y recorta el valor de un slider, eso no es una orden de moverse."""
        alcance = int(math.ceil(self.robot.alcance_de_la_punta))
        for slider, (lo, hi) in zip(self.sliders_xyz, ((-alcance, alcance), (-alcance, alcance), (0, alcance))):
            slider.blockSignals(True)
            slider.setRange(lo, hi)
            slider.blockSignals(False)

    def poner_etiquetas(self):
        for etiqueta, valor in zip(self.valores_servo, self.servos):
            etiqueta.setText(f"{valor:.0f}°")
        for etiqueta, valor in zip(self.valores_xyz, self.posicion):
            etiqueta.setText(f"{round(valor) + 0:.0f}")  # (+ 0 QUITA EL "-0")

    def _arrastrando(self):
        return any(slider.isSliderDown() for slider in (*self.sliders_servo, *self.sliders_xyz))

    def _slider_servo_movido(self):
        # LOS SLIDERS DE SERVOS SON LOS ÁNGULOS DE CADA SERVO (LO MISMO QUE RECIBE LA PLACA)
        # VAN POCO A POCO (≤ salto_max_por_paso): UN CLIC AL EXTREMO YA NO BRINCA 0° → 180° DE GOLPE
        destino = list(self.servos)
        for i, slider in enumerate(self.sliders_servo):
            destino[i] = slider.value()
        self.ejecutar(tramo_articular(self.robot, self.servos, destino), bloquear=False, origen="Servos")

    def _slider_xyz_movido(self, eje):
        """Un slider de la punta cambió: no se deja entrar al rojo (se detiene en la frontera) y se mueve la punta."""
        slider = self.sliders_xyz[eje]
        desde = self._referencia_del_jog()
        self._pintar_alcance(solo_lo_cambiado=True, ejes=(eje,))  # EL SLIDER FRENA CON EL ROJO DE ESTE MOMENTO
        pedido = slider.value()
        tope = slider.acotar(pedido, desde[eje])
        topado = tope != pedido
        if topado:
            slider.blockSignals(True)
            slider.setValue(tope)
            slider.blockSignals(False)
            # EN «NEGARSE» EL ROJO TAMBIÉN ES LO QUE PIDE CAMBIAR DE POSTURA: SI UN MILÍMETRO MÁS ALLÁ SÍ SE ALCANZA, EL TOPE
            # ES POR ESO Y EL AVISO LO DICE (Y DICE CÓMO SÍ SE PASA), EN VEZ DE AFIRMAR QUE NO SE ALCANZA
            mas_alla = list(desde)
            mas_alla[eje] = tope + (1 if pedido > tope else -1)
            por_postura = self.postura == "negar" and alcanzable(self.robot, mas_alla)
            nombre = NOMBRES_EJES[eje]
            self._avisar_jog(f"La punta no pasa de {nombre} = {tope} mm sin cambiar de postura y el modo es «negarse»: "
                             f"con «girar despacio» sí pasa." if por_postura else
                             f"La punta no llega más allá de {nombre} = {tope} mm desde esta pose: "
                             f"el resto del slider (en rojo) no se alcanza.", ("tope", eje, tope),
                             eje=nombre, tope_mm=tope, desde_mm=desde, por_postura=por_postura)
        # DURANTE UN ARRASTRE EL AVISO NO SE REARMA (EL PLANEADOR ACEPTA Y NIEGA DESTINOS ALTERNADOS Y SE REPETIRÍA): SOLO
        # SOLTAR EL SLIDER. SIN ARRASTRE (TECLADO, RUEDA), UN MOVIMIENTO DENTRO DEL TRAMO LO REARMA. Y EL MISMO AVISO
        # (_aviso_jog_dado) SOLO VUELVE A VALER SI ALGO SE MOVIÓ DENTRO DEL TRAMO: SOLTAR Y VOLVER A AGARRAR NO BASTA
        if self._jog_cartesiano(eje, desde) and not topado:
            self._aviso_jog_dado = None
            if not self._jog_arrastrando:
                self._jog_avisado = self._jog_de_paso_avisado = False

    def _jog_arrastre(self, empieza):
        """Se agarra o se suelta un slider de la punta: arrastre nuevo, aviso nuevo. Durante el arrastre el rojo de los
        otros dos sliders no se recalcula en cada pixel: se pone al día cada pocos pasos y aquí, al soltar."""
        self._jog_arrastrando = empieza
        self._jog_avisado = self._jog_de_paso_avisado = False
        if not empieza:
            self._pintar_alcance(solo_lo_cambiado=True)

    def _referencia_del_jog(self):
        """El punto del que se mide una orden nueva de la punta: su destino, el freno de cada slider y su rojo. Con la
        punta quieta, o en «Cancelar» (el tramo nuevo sale de donde va la punta), es donde está la punta. En «Reemplazar»
        y «En cola», con la punta en camino, es el punto al que va: el que marcan los tres sliders, con sus decimales."""
        return self.posicion if self._jog_desde_la_punta() else self._destino_jog

    def _jog_desde_la_punta(self):
        """True si una orden nueva de la punta se mide desde donde está; False si se mide desde el punto al que va."""
        return self._destino_jog is None or self.en_camino == "cancelar"

    def _servos_al_terminar(self):
        """Los servos con los que el robot termina lo que tiene pendiente (los de ahora, si no hay nada pendiente)."""
        return next((p.servos for p in reversed(self.cola) if p.servos is not None), self.servos)

    def _detras_de_lo_pendiente(self, eje, desde):
        """De dónde sale un tramo de la punta que va DETRÁS de lo pendiente, y cuántos pasos de lo pendiente se quedan:
        (conservar, servos, salida). None si no hay un jog pendiente, o si el mismo slider ya empezó su tramo (no hay un
        «detrás» que le toque). Un solo lugar lo decide: lo usan «En cola» y el punto de paso de «Reemplazar»."""
        if self._tramo_jog is None:
            return None
        eje_ultimo, pasos_ultimo, servos_ultimo, salida_ultimo = self._tramo_jog
        if eje_ultimo != eje:                        # OTRO SLIDER: SU TRAMO VA DETRÁS DE TODO LO PENDIENTE
            return len(self.cola), self._servos_al_terminar(), desde
        if len(self.cola) >= pasos_ultimo:           # EL MISMO SLIDER Y SU TRAMO NO HA EMPEZADO: SE TRAZA OTRA VEZ DESDE DONDE SALÍA
            return len(self.cola) - pasos_ultimo, servos_ultimo, salida_ultimo
        return None

    def _avisar_jog(self, texto, clave, **datos):
        """Un aviso del jog por arrastre: tras el primero se calla hasta que se suelte el slider. Sin arrastre (teclado,
        rueda) se calla hasta que algo se mueva. Y el mismo aviso (`clave`: qué slider y por qué) no se repite aunque se
        suelte el slider y se vuelva a agarrar: insistir contra el rojo avisa una vez, hasta que algo se mueva.

        El aviso del punto de paso (`de_paso`) lleva su propia cuenta: sale una vez por arrastre aunque ya haya salido
        otro. Cuando el arrastre topa con el rojo, el aviso del tope se pide antes, y sin esto la punta iba de paso sin
        decirlo. Al revés no: si el de paso ya salió, el del tope de ese arrastre se calla."""
        if datos.get("de_paso"):
            if self._jog_de_paso_avisado:
                return
            self._jog_avisado = self._jog_de_paso_avisado = True
        elif self._jog_avisado or clave == self._aviso_jog_dado:
            return
        else:
            self._jog_avisado, self._aviso_jog_dado = True, clave
        self.avisar(texto, "aviso", "Jog", **datos)

    def _jog_cartesiano(self, eje, desde):
        """Pestaña Punta X Y Z: se movió el slider `eje`. La punta va en línea recta a `desde` (el punto del que se mide
        la orden, ver _referencia_del_jog) con ese eje donde lo marca su slider; la CI mueve los servos. Con la punta en
        camino, el selector «En camino» decide qué pasa con lo que faltaba. True si el movimiento salió.

        Los otros dos ejes se toman de `desde`, con sus decimales, y no de sus sliders: un slider guarda milímetros enteros
        (y bajo la base el de Z marca 0 porque no baja más); tomar su valor saca el destino de la recta sobre la que se
        pintó el rojo, y justo en la frontera el planeador se niega."""
        slider = self.sliders_xyz[eje]
        destino = list(desde)
        destino[eje] = float(slider.value())
        # DE DÓNDE SALE EL TRAMO NUEVO Y CUÁNTOS PASOS DE LO PENDIENTE SE QUEDAN. «REEMPLAZAR» Y «CANCELAR»: SALE DE DONDE
        # VA LA PUNTA Y NO SE QUEDA NADA
        conservar, servos, salida = 0, self.servos, self.posicion
        detras = self._detras_de_lo_pendiente(eje, desde) if self.en_camino == "en_cola" else None
        if detras is not None:
            conservar, servos, salida = detras
        # EN «EN COLA», EL MISMO SLIDER CON SU TRAMO YA EMPEZADO SALE DE DONDE VA LA PUNTA, COMO EN LOS OTROS DOS MODOS
        distancia = math.dist(salida, destino)
        if distancia < 0.5 and self._destino_jog is None:
            return False                             # LA PUNTA YA ESTÁ AHÍ
        if distancia < 0.5:
            plan, motivo = [], None                  # EL SLIDER VOLVIÓ A DONDE SALE SU TRAMO: ESE TRAMO YA NO HACE FALTA
        else:
            plan, motivo = tramo_continuo(self.robot, servos, destino, self.postura, pasos=muestras_del_jog(distancia))
        if plan is None and self.en_camino == "reemplazar" and self._destino_jog is not None:
            # D7: DESDE DONDE VA LA PUNTA NO SE PUEDE IR DIRECTO (LO QUE ALCANZA LA PUNTA NO ES CONVEXO), PERO EL ROJO DEL
            # SLIDER SE MIDIÓ DESDE EL PUNTO AL QUE VA. LA ORDEN NO SE PIERDE: LA PUNTA TERMINA LO PENDIENTE Y LUEGO VA
            for de_paso in (self._detras_de_lo_pendiente(eje, desde), (len(self.cola), self._servos_al_terminar(), desde)):
                if de_paso is None or not de_paso[0]:   # SIN NADA PENDIENTE QUE CONSERVAR ES LA MISMA ORDEN DIRECTA
                    continue
                plan, _ = tramo_continuo(self.robot, de_paso[1], destino, self.postura,
                                         pasos=muestras_del_jog(math.dist(de_paso[2], destino)))
                if plan is not None:
                    conservar, servos, salida = de_paso
                    self._avisar_jog(f"No se puede ir directo a {tuple(int(c) for c in destino)} desde donde va la punta: "
                                     "termina el tramo que lleva y luego va.", ("de_paso", eje),
                                     destino_mm=destino, desde_mm=list(salida), de_paso=True, postura=self.postura)
                    break
        if plan is None:
            self._avisar_jog(f"No se mueve: el camino hacia {tuple(int(c) for c in destino)} {motivo}"
                             if not motivo.startswith("Para") else f"No se mueve. {motivo}",
                             ("negado", eje, motivo.startswith("Para")),
                             destino_mm=destino, desde_mm=salida, postura=self.postura)
            # EL SLIDER REGRESA A DONDE ESTABA SU ORDEN (O LA PUNTA); LO QUE YA IBA EN CAMINO SIGUE IGUAL
            self._sincronizar((slider,), ((self._destino_jog or self.posicion)[eje],))
            return False
        if self.en_camino == "cancelar":             # LA PUNTA SE DETIENE DONDE VA: LOS OTROS DOS SLIDERS REGRESAN AHÍ
            self._sincronizar(*zip(*((s, c) for s, c in zip(self.sliders_xyz, self.posicion) if s is not slider)))
        self.ejecutar(plan, bloquear=False, origen="Jog", conservar=conservar)
        self._destino_jog, self._tramo_jog = tuple(destino), (eje, len(plan), servos, tuple(salida))
        # EL ROJO DE LOS OTROS DOS SLIDERS SE PONE AL DÍA AQUÍ; EN UN ARRASTRE NO (SERÍA EN CADA PIXEL): AHÍ LO HACEN
        # _siguiente_paso CADA POCOS PASOS Y _jog_arrastre AL SOLTAR. POR ESO EL CONTADOR SOLO VUELVE A CERO AQUÍ: UN RATÓN
        # MANDA VARIAS ÓRDENES ENTRE PASO Y PASO, Y SI CADA UNA LO REGRESARA A CERO NO LLEGARÍA NUNCA
        if not self._jog_arrastrando:
            self._pasos_jog = 0
            self._pintar_alcance(solo_lo_cambiado=True)
        return True

    @staticmethod
    def _sincronizar(sliders, valores):
        for slider, valor in zip(sliders, valores):
            slider.blockSignals(True)
            slider.setValue(_marca(slider, valor))
            slider.blockSignals(False)

    def _sincronizar_sliders(self):
        """Las dos pestañas muestran siempre la misma pose: servos en grados y punta en mm.
        Un grupo de sliders que la persona está arrastrando no se toca. Los tramos que la punta alcanza desde esta
        pose se recalculan aquí: es lo que pasa siempre que algo termina de moverse (y al arrancar, en HOME).
        Por eso aquí se olvida el jog pendiente de la punta: los sliders vuelven a decir dónde está, no a dónde va."""
        self._destino_jog = self._tramo_jog = None
        self._pintar_alcance()
        if not any(s.isSliderDown() for s in self.sliders_servo):
            self._sincronizar(self.sliders_servo, self.servos)
        if not any(s.isSliderDown() for s in self.sliders_xyz):
            self._sincronizar(self.sliders_xyz, self.posicion)

    def _pintar_alcance(self, solo_lo_cambiado=False, ejes=(0, 1, 2)):
        """Cada slider de la punta pinta hasta dónde se le puede llevar en este momento: el tramo que la punta alcanza
        sobre su eje, medido desde el punto del que saldría un tramo nuevo (_referencia_del_jog). Sin solver de
        cinemática inversa no se sabe qué se alcanza: no se pinta ni se frena nada.

        solo_lo_cambiado: se salta el slider cuya recta es la que ya tiene pintada. Con la punta en camino se pregunta
        muy seguido y cada recta cuesta unos 6 ms."""
        desde = self._referencia_del_jog()
        for eje in ejes:
            slider = self.sliders_xyz[eje]
            if self.robot.cinematica_inversa is None:
                slider.poner_tramos(None)
                continue
            # EL TRAMO SE BUSCA TAMBIÉN DONDE ESTÁ LA PUNTA AUNQUE EL SLIDER NO LLEGUE AHÍ (BAJO LA BASE, EL DE Z EMPIEZA EN
            # 0): SI NO, EL SLIDER NO SABE EN QUÉ TRAMO ESTÁ LA PUNTA Y NO FRENA
            aqui = desde[eje]
            lo, hi = min(slider.minimum(), aqui), max(slider.maximum(), aqui)
            recta = (*(c for otro, c in enumerate(desde) if otro != eje), lo, hi)  # LAS OTRAS DOS COORDENADAS Y DÓNDE SE BUSCA
            pintada = self._rectas_pintadas[eje]
            if solo_lo_cambiado and pintada is not None and max(abs(a - b) for a, b in zip(recta, pintada)) < MISMA_RECTA_MM:
                continue
            self._rectas_pintadas[eje] = recta
            tramos = tramos_alcanzables(self.robot, desde, eje, lo, hi)
            if self.postura == "negar":  # EL TRAMO DONDE ESTÁ LA PERILLA SE RECORTA A LO QUE NO PIDE CAMBIAR DE POSTURA
                tramos = [self._sin_cambiar_de_postura(eje, desde, tramo)
                          if tramo[0] - TOLERANCIA_TRAMOS_MM <= aqui <= tramo[1] + TOLERANCIA_TRAMOS_MM else tramo
                          for tramo in tramos]
            slider.poner_tramos(tramos)

    def _sin_cambiar_de_postura(self, eje, desde, tramo):
        """En «negarse»: el trozo de `tramo` (el de un slider, donde está su perilla) al que la punta llega en línea recta
        sin cambiar de postura. El resto se pinta en rojo igual que lo que no se alcanza, porque el planeador se va a negar:
        lo decide su mismo recorrido (hasta_donde_llega), con las mismas muestras que usa la orden del jog.

        Se mide sobre el eje, desde la referencia del jog y con los servos que el robot tendrá ahí. Con la punta en camino
        en «Reemplazar» el tramo nuevo sale en diagonal desde donde va la punta: ahí esto es una aproximación (como el rojo
        de lo que no se alcanza) y el planeador conserva la última palabra."""
        servos = self.servos if self._jog_desde_la_punta() else self._servos_al_terminar()
        bordes = []
        for extremo, entero_de_afuera in zip(tramo, (math.floor, math.ceil)):
            destino = list(desde)
            destino[eje] = extremo
            punto, servos_ahi = hasta_donde_llega(self.robot, servos, destino, "negar",
                                                  pasos=muestras_del_jog(abs(extremo - desde[eje])))
            borde = float(punto[eje])
            # EL BORDE SALE AFINADO A TOLERANCIA_TRAMOS_MM Y EL SLIDER ES ENTERO: SI EL ENTERO QUE SIGUE TAMBIÉN SE ACEPTA
            # (Y = 0, EL PLANO DEL FRENTE, ES EL CASO DE TODOS LOS DÍAS), EL BORDE ES ESE ENTERO Y NO UN MILÍMETRO ANTES
            destino[eje] = float(entero_de_afuera(borde))
            if (destino[eje] != borde and tramo[0] <= destino[eje] <= tramo[1]
                    and tramo_continuo(self.robot, servos_ahi, destino, "negar", pasos=2)[0] is not None):
                borde = destino[eje]
            bordes.append(borde)
        return tuple(bordes)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONSOLA DE REGISTRO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def _configurar_registro(self):
        for texto, niveles in FILTROS_REGISTRO:
            self.FILTRO_REGISTRO.addItem(texto, niveles)
        self.FILTRO_REGISTRO.currentIndexChanged.connect(self._repintar_registro)
        self.LISTA_REGISTRO.currentItemChanged.connect(self._mostrar_detalle)
        self.BTN_COPIAR_REGISTRO.clicked.connect(self.copiar_registro)
        self.BTN_EXPORTAR_REGISTRO.clicked.connect(self.exportar_registro)
        self.LISTA_REGISTRO.setWordWrap(False)  # UN EVENTO, UN RENGLÓN: EL TEXTO COMPLETO SE LEE EN EL DETALLE
        self.LISTA_REGISTRO.setTextElideMode(QtCore.Qt.ElideRight)
        self.LISTA_REGISTRO.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        # BARRA VERTICAL SIEMPRE PRESENTE: ASÍ EL ANCHO PARA EL TEXTO NO CAMBIA CUANDO LA LISTA SE LLENA
        self.LISTA_REGISTRO.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.LISTA_REGISTRO.setItemDelegate(_RenglonDelRegistro(self.LISTA_REGISTRO))
        self.LISTA_REGISTRO.setResizeMode(QtWidgets.QListView.Adjust)
        self.DETALLE_REGISTRO.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.DETALLE_REGISTRO.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)
        self.DETALLE_REGISTRO.setViewportMargins(0, 0, 10, 0)  # QUE EL TEXTO NO QUEDE BAJO LA BARRA
        self.DETALLE_REGISTRO.setPlaceholderText("Elige un evento para ver sus datos.")
        self._nuevos_sin_ver = 0
        self.BTN_NUEVOS_REGISTRO.hide()
        self.BTN_NUEVOS_REGISTRO.clicked.connect(self._ir_al_ultimo)
        self.LISTA_REGISTRO.verticalScrollBar().valueChanged.connect(self._registro_desplazado)
        self.registro.suscribir(self._evento_registrado)
        self.BTN_DETENER.setEnabled(False)  # SOLO SE ENCIENDE MIENTRAS ALGO SE MUEVE

    def _item_de(self, evento):
        item = QtWidgets.QListWidgetItem(f"{evento.hora:%H:%M:%S}  {ICONO_NIVEL[evento.nivel]} "
                                         f"{' '.join(evento.mensaje.split())}")  # UN RENGLÓN: EL ORIGEN Y LO DEMÁS, EN EL DETALLE
        item.setForeground(QtGui.QColor(COLOR_NIVEL[evento.nivel]) if evento.nivel != "info" else QtGui.QColor("#1f2d3a"))
        item.setData(QtCore.Qt.UserRole, evento)
        item.setToolTip(f"{evento.origen}: {evento.mensaje}")
        return item

    def _al_final(self):
        barra = self.LISTA_REGISTRO.verticalScrollBar()
        return barra.value() >= barra.maximum() - 2

    def _evento_registrado(self, evento):
        """Lo más nuevo va abajo y la lista lo sigue sola, como una terminal. Si la persona subió a
        leer algo, no se le mueve la lista: aparece «↓ N nuevos» para regresar."""
        if evento.nivel not in self.FILTRO_REGISTRO.currentData():
            return
        seguir = self._al_final()
        self.LISTA_REGISTRO.addItem(self._item_de(evento))
        while self.LISTA_REGISTRO.count() > reg.MAXIMO_EN_MEMORIA:
            self.LISTA_REGISTRO.takeItem(0)
        if seguir:
            self.LISTA_REGISTRO.scrollToBottom()
        else:
            self._nuevos_sin_ver += 1
            self.BTN_NUEVOS_REGISTRO.setText(f"↓ {self._nuevos_sin_ver} nuevo{'s' if self._nuevos_sin_ver > 1 else ''}")
            self.BTN_NUEVOS_REGISTRO.show()

    def _registro_desplazado(self, _valor):
        if self._al_final() and self._nuevos_sin_ver:
            self._nuevos_sin_ver = 0
            self.BTN_NUEVOS_REGISTRO.hide()

    def _ir_al_ultimo(self):
        self.LISTA_REGISTRO.scrollToBottom()
        self._registro_desplazado(None)

    def _repintar_registro(self):
        self.LISTA_REGISTRO.clear()
        for evento in self.registro.filtrar(self.FILTRO_REGISTRO.currentData()):
            self.LISTA_REGISTRO.addItem(self._item_de(evento))
        self._ir_al_ultimo()

    def _mostrar_detalle(self, item, _anterior=None):
        if item is None:
            self.DETALLE_REGISTRO.clear()
            return
        e = item.data(QtCore.Qt.UserRole)
        filas = "".join(f"<tr><td style='color:#555;padding-right:8px'>{html.escape(k)}</td>"
                        f"<td><code>{html.escape(json.dumps(v, ensure_ascii=False))}</code></td></tr>"
                        for k, v in e.a_dict()["datos"].items())
        self.DETALLE_REGISTRO.setHtml(
            f"<b style='color:{COLOR_NIVEL[e.nivel]}'>{ICONO_NIVEL[e.nivel]} {e.nivel.upper()}</b> · {html.escape(e.origen)}"
            f" · {e.hora:%Y-%m-%d %H:%M:%S.%f}"[:-3] + f"<br>{html.escape(e.mensaje)}"
            + (f"<table style='margin-top:4px'>{filas}</table>" if filas else "<br><i style='color:#888'>Sin datos</i>"))

    def eventos_visibles(self):
        """En orden cronológico (como en la lista y en el archivo)."""
        return [self.LISTA_REGISTRO.item(i).data(QtCore.Qt.UserRole) for i in range(self.LISTA_REGISTRO.count())]

    def copiar_registro(self):
        """Copia el evento elegido (o todos los visibles) como JSON Lines al portapapeles."""
        item = self.LISTA_REGISTRO.currentItem()
        eventos = [item.data(QtCore.Qt.UserRole)] if item is not None else self.eventos_visibles()
        QtWidgets.QApplication.clipboard().setText("\n".join(e.a_jsonl() for e in eventos))
        self.statusBar().showMessage(f"Copiados {len(eventos)} eventos (JSON Lines).")

    def exportar_registro(self):
        ruta, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Exportar registro", "registro.jsonl", "JSON Lines (*.jsonl)")
        if ruta:
            reg.exportar_jsonl(self.eventos_visibles(), ruta)
            self.avisar(f"Registro exportado: {len(self.eventos_visibles())} eventos", "ok", "Sistema", ruta=ruta)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PROGRAMAS (PESTAÑA PROGRAMA) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def _configurar_programas(self):
        self.BTN_REPRODUCIR.clicked.connect(self.reproducir)
        self.BTN_DETENER.clicked.connect(self.detener)
        self.BTN_GRABAR.clicked.connect(self.alternar_grabacion)
        # EL BOTÓN MIDE LO DE SU TEXTO MÁS LARGO: AL CAMBIAR DE TEXTO, NADA DE LA FILA SE RECORRE
        anchos = []
        for texto in (TEXTO_DETENER_GRABACION, TEXTO_GRABAR):
            self.BTN_GRABAR.setText(texto)
            anchos.append(self.BTN_GRABAR.sizeHint().width())
        self.BTN_GRABAR.setMinimumWidth(max(anchos))
        self.SELECTOR_PROGRAMA.currentIndexChanged.connect(self._selector_cambio)
        self.recargar_programas()

    def recargar_programas(self):
        """Lista los programas de la carpeta y deja elegido el último que se usó."""
        ultimo = self.ajustes.value("programa/ultimo", "")
        self.SELECTOR_PROGRAMA.blockSignals(True)
        self.SELECTOR_PROGRAMA.clear()
        nombres = programas.listar(self.carpeta_programas)
        for nombre in nombres:
            self.SELECTOR_PROGRAMA.addItem(nombre, nombre)
        self.SELECTOR_PROGRAMA.addItem(TEXTO_NUEVO_PROGRAMA, None)
        # «NUEVO» SOLO QUEDA ELEGIDO SI LA PERSONA LO ELIGE: SIN PROGRAMAS GUARDADOS, NO QUEDA ELEGIDO NINGUNO
        self.SELECTOR_PROGRAMA.setCurrentIndex(nombres.index(ultimo) if ultimo in nombres else 0 if nombres else -1)
        self.SELECTOR_PROGRAMA.blockSignals(False)
        self._programa_elegido()

    def _nuevo_elegido(self):
        """True si en el selector está elegido «+ Nuevo programa» (el único renglón que no es un archivo)."""
        return self.SELECTOR_PROGRAMA.currentIndex() >= 0 and self.SELECTOR_PROGRAMA.currentData() is None

    def _programa_nuevo(self):
        """El programa por puntos que todavía no tiene ni puntos ni nombre: se le pide al guardar el primero."""
        return programas.Programa("", "puntos", self.robot.nombre, self.robot.gdl)

    def _selector_cambio(self):
        """La persona eligió otro renglón del selector. «+ Nuevo programa» abre el editor, vacío."""
        self._programa_elegido()
        if self._nuevo_elegido() and not self.editando:
            self.abrir_editor()

    def _programa_elegido(self):
        """Pinta el resumen del programa elegido en el selector. El editor de puntos, si está abierto, lo sigue."""
        nombre, p = self.SELECTOR_PROGRAMA.currentData(), None
        self.BTN_REPRODUCIR.setEnabled(nombre is not None)
        if self._nuevo_elegido():
            p = self._programa_nuevo()
            self.INFO_PROGRAMA.setText("Programa nuevo · sin puntos")
        elif nombre is None:
            self.INFO_PROGRAMA.setText("No hay programas guardados")
            self.INFO_PROGRAMA.setToolTip(self.carpeta_programas)
        else:
            self.ajustes.setValue("programa/ultimo", nombre)
            try:
                p = programas.cargar(os.path.join(self.carpeta_programas, nombre), self.robot)
                self.INFO_PROGRAMA.setText(self._resumen(p))
            except programas.ErrorPrograma as e:
                self.INFO_PROGRAMA.setText("⚠ No se puede leer (ver registro)")
                self.avisar(str(e), "error", "Programa", archivo=nombre)
        self.tipo_elegido = p.tipo if p is not None else None
        if self.editando and self.tipo_elegido == "puntos":
            self._editar(p, nombre)
        else:
            self.cerrar_editor()  # UNA GRABACIÓN (O NINGÚN PROGRAMA) NO TIENE PUNTOS QUE ENSEÑAR
        self._actualizar_editor()

    @staticmethod
    def _resumen(p):
        """Lo que dice la pestaña Programa del programa elegido."""
        if p.tipo == "puntos":
            return f"Por puntos · {_cuantos_puntos(len(p.puntos))}"
        return f"Grabación · {len(p.muestras)} muestras · {p.duracion:.1f} s"

    def programa_elegido(self):
        """El Programa elegido en el selector, o None (y avisa por qué)."""
        nombre = self.SELECTOR_PROGRAMA.currentData()
        if nombre is None:
            self.avisar("No hay programa elegido.", "aviso", "Programa")
            return None
        try:
            return programas.cargar(os.path.join(self.carpeta_programas, nombre), self.robot)
        except programas.ErrorPrograma as e:
            self.avisar(str(e), "error", "Programa", archivo=nombre)
            return None

    def reproducir(self):
        """▶ Reproducir: el plan se calcula COMPLETO antes de mover; si no puede terminar, no arranca."""
        if self.en_movimiento or self._grabando_avisa():
            return
        p = self.programa_elegido()
        if p is None:
            return
        if p.tipo == "puntos":
            if not self._puede_planear():
                return
            plan, problemas = plan_puntos(self.robot, self.servos, p, self.postura)
        else:
            plan, problemas = plan_grabacion(self.robot, self.servos, p, self.CHK_QUITAR_PAUSAS.isChecked())
        if self._no_arranca(p, problemas, "Programa"):
            return
        segundos = duracion_ms(self.robot, plan) / 1000 * 100 / self.velocidad
        self.avisar(f"Reproduciendo «{p.nombre}» (~{segundos:.1f} s)", "info", "Programa",
                    programa=p.nombre, tipo=p.tipo, duracion_s=segundos, velocidad=self.velocidad, postura=self.postura)
        self.ejecutar(plan, origen="Programa")

    def _no_arranca(self, programa, problemas, origen):
        """True (y avisa) si el plan del programa tiene problemas: lo que no puede terminar, no arranca."""
        if problemas:
            extra = f" (y {len(problemas) - 1} más)" if len(problemas) > 1 else ""
            self.avisar(f"No arranca: {problemas[0][1]}{extra}", "aviso", origen,
                        programa=programa.nombre, problemas=[m for _, m in problemas])
        return bool(problemas)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GRABACIÓN (PESTAÑA PROGRAMA) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # LA LÓGICA VIVE EN cosechador/movimiento/grabadora.py; AQUÍ SOLO SE CONECTA CON LOS BOTONES Y LOS DIÁLOGOS.
    def alternar_grabacion(self):
        """⏺ Grabar / ⏹ Detener grabación."""
        if self.grabadora.grabando:
            self.terminar_grabacion()
        elif self.BTN_GRABAR.isEnabled():  # APAGADO MIENTRAS SE REPRODUCE, CALCULA O COSECHA
            self.empezar_grabacion()

    def empezar_grabacion(self):
        self.grabadora.empezar(self.servos)
        self._boton_grabar(grabando=True)
        self._ocupado(False)
        self.INFO_PROGRAMA.setText("⏺ Grabando…")
        self.avisar("Grabando: lo que se mueva con los sliders queda guardado.", "info", "Grabación", servos=self.servos)

    def terminar_grabacion(self):
        """⏹ Detener grabación. El nombre se pide CON LA GRABADORA VIVA: «Seguir grabando» no pierde nada, y para
        tirar lo grabado hay que pedirlo y confirmarlo (un «Cancelar» que destruye le costó a Paul 170 muestras)."""
        muestras = self.grabadora.muestras
        if muestras is None:
            self._soltar_grabacion()
            self.avisar("No se guardó: no hubo movimiento mientras se grababa.", "aviso", "Grabación")
            return
        cuantas, segundos = self.grabadora.resumen()
        while True:
            guardado = self._guardar_con_nombre(f"grabacion-{dt.datetime.now():%Y-%m-%d-%H%M}", "grabacion",
                                                lambda nombre: programa_de_grabacion(muestras, nombre, self.robot))
            if guardado is None:
                self.grabadora.reanudar()  # EL RATO CON EL DIÁLOGO ABIERTO NO ES PARTE DE LA GRABACIÓN
                self.avisar("Sigue grabando: no se perdió nada.", "info", "Grabación", muestras=cuantas)
                return
            if guardado is not DESCARTAR:
                break
            if self._confirmar("Descartar grabación", f"Se pierden {cuantas} muestras ({segundos:.1f} s). "
                                                      "¿Descartar la grabación?", "Descartar", "Conservar"):
                self._soltar_grabacion()
                self.avisar(f"Grabación descartada: {cuantas} muestras ({segundos:.1f} s).", "aviso", "Grabación",
                            muestras=cuantas, duracion_s=segundos)
                return
        self._soltar_grabacion()
        archivo, p = guardado
        self.ajustes.setValue("programa/ultimo", archivo)
        self.recargar_programas()  # APARECE EN EL SELECTOR Y QUEDA ELEGIDA
        self.avisar(f"Grabación guardada: {archivo} ({len(muestras)} muestras, {p.duracion:.1f} s)", "ok", "Grabación",
                    ruta=os.path.join(self.carpeta_programas, archivo), muestras=len(muestras), duracion_s=p.duracion)

    def _guardar_con_nombre(self, sugerido, tipo, hacer_programa):
        """Pide el nombre y guarda el programa que devuelve hacer_programa(nombre).

        Las reglas del nombre son UNAS para todo programa: si no sirve se vuelve a pedir, y si ya existe un
        archivo con ese nombre se pregunta antes de reemplazarlo. Devuelve (archivo, programa), o None si la
        persona cancela, o DESCARTAR si pide tirar la grabación.
        """
        origen = TEXTOS_GUARDAR[tipo][2]
        while True:
            nombre = self._pedir_nombre(sugerido, tipo)
            if nombre is None or nombre is DESCARTAR:
                return nombre
            try:
                archivo = programas.nombre_de_archivo(nombre)
            except programas.ErrorPrograma as e:
                self.avisar(str(e), "aviso", origen, escrito=nombre)
                continue
            ruta = os.path.join(self.carpeta_programas, archivo)
            if os.path.exists(ruta) and not self._confirmar_sobrescribir(archivo, tipo):
                sugerido = nombre
                continue
            p = hacer_programa(nombre.strip())
            try:
                os.makedirs(self.carpeta_programas, exist_ok=True)
                programas.guardar(p, ruta)
            except OSError as e:
                self.avisar(f"No se pudo guardar {archivo}: {e}", "error", origen, ruta=ruta)
                continue
            return archivo, p

    def _soltar_grabacion(self):
        """La grabación terminó (guardada, descartada o vacía): la grabadora se cierra y la pantalla vuelve a su estado."""
        self.grabadora.terminar()
        self._boton_grabar(grabando=False)
        self._ocupado(False)
        self._programa_elegido()

    def _boton_grabar(self, grabando):
        """Aspecto de «grabando» (rol de tema.py): el botón en rojo y el aviso que se ve desde las tres pestañas."""
        self.aviso_grabando.setVisible(grabando)
        self._acomodar_esquina()
        self.BTN_GRABAR.setText(TEXTO_DETENER_GRABACION if grabando else TEXTO_GRABAR)
        self.BTN_GRABAR.setProperty("rol", "grabando" if grabando else "")
        self.BTN_GRABAR.style().unpolish(self.BTN_GRABAR)  # Qt SOLO RELEE LA HOJA AL CAMBIAR UNA PROPIEDAD SI SE LE PIDE
        self.BTN_GRABAR.style().polish(self.BTN_GRABAR)

    def _pedir_nombre(self, sugerido, tipo="grabacion"):
        """El nombre que escribe la persona; None si cancela; DESCARTAR si pide tirar la grabación."""
        dialogo, campo, _ = self._dialogo_del_nombre(sugerido, tipo)
        dialogo.exec_()
        return self._respuesta_del_nombre(dialogo, campo)

    def _dialogo_del_nombre(self, sugerido, tipo):
        """El diálogo del nombre, sin abrirlo: (diálogo, campo, botones). Una grabación trae, además de «Guardar»,
        «Seguir grabando» (no se pierde nada) y «Descartar…» (pregunta antes); un programa por puntos, «Cancelar»."""
        titulo, etiqueta, _ = TEXTOS_GUARDAR[tipo]
        dialogo = QtWidgets.QDialog(self)
        dialogo.setWindowTitle(titulo)
        caja = QtWidgets.QVBoxLayout(dialogo)
        caja.addWidget(QtWidgets.QLabel(etiqueta))
        campo = QtWidgets.QLineEdit(sugerido)
        campo.setMinimumWidth(ANCHO_DEL_NOMBRE)
        campo.selectAll()
        caja.addWidget(campo)
        guardar = QtWidgets.QPushButton("Guardar")
        volver = QtWidgets.QPushButton("Seguir grabando" if tipo == "grabacion" else "Cancelar")
        guardar.clicked.connect(dialogo.accept)
        volver.clicked.connect(dialogo.reject)
        botones, fila = [guardar, volver], QtWidgets.QHBoxLayout()
        if tipo == "grabacion":  # LO QUE DESTRUYE VA APARTE, A LA IZQUIERDA
            botones.append(QtWidgets.QPushButton("Descartar…"))
            botones[-1].clicked.connect(lambda: dialogo.done(RESULTADO_DESCARTAR))
            fila.addWidget(botones[-1])
            fila.addSpacing(32)
        fila.addStretch(1)
        fila.addWidget(volver)
        fila.addWidget(guardar)
        for boton in botones:  # «Intro» SIEMPRE ES «Guardar», TENGA EL FOCO QUIEN LO TENGA: NUNCA DESCARTA
            boton.setAutoDefault(False)
        guardar.setDefault(True)
        caja.addLayout(fila)
        return dialogo, campo, botones

    @staticmethod
    def _respuesta_del_nombre(dialogo, campo):
        if dialogo.result() == RESULTADO_DESCARTAR:
            return DESCARTAR
        return campo.text() if dialogo.result() == QtWidgets.QDialog.Accepted else None

    def _confirmar_sobrescribir(self, archivo, tipo="grabacion"):
        return self._confirmar(TEXTOS_GUARDAR[tipo][0], f"Ya existe «{archivo}». ¿Reemplazarlo?", "Reemplazar", "Cancelar")

    def _confirmar(self, titulo, pregunta, si, no):
        """Para lo que no tiene vuelta atrás: dos botones con su verbo (si, no); el que no destruye va por omisión."""
        caja, boton_si, _ = self._caja_de_confirmar(titulo, pregunta, si, no)
        caja.exec_()
        return caja.clickedButton() is boton_si

    def _caja_de_confirmar(self, titulo, pregunta, si, no):
        """La pregunta, sin abrirla: (caja, botón que confirma, botón que no). «Intro» y «Esc» son el que no."""
        caja = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Question, titulo, pregunta, parent=self)
        boton_si = caja.addButton(si, QtWidgets.QMessageBox.DestructiveRole)
        boton_no = caja.addButton(no, QtWidgets.QMessageBox.RejectRole)
        caja.setDefaultButton(boton_no)
        caja.setEscapeButton(boton_no)
        return caja, boton_si, boton_no

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EDITOR DE PUNTOS (COLUMNA PUNTOS) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def _configurar_editor(self):
        """La columna PUNTOS nace oculta: aparece con «✎ Editar», en el lugar de VISIÓN."""
        self.FONDO_PUNTOS.hide()
        self._ayuda_editar, self._ayuda_quitar = self.BTN_EDITAR_PUNTOS.toolTip(), self.BTN_QUITAR_PUNTO.toolTip()
        lista = self.LISTA_PUNTOS  # MISMA RECETA QUE LA LISTA DEL REGISTRO (_configurar_registro)
        lista.setWordWrap(True)
        lista.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        lista.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        lista.setItemDelegate(_RenglonDePunto(lista))
        lista.setResizeMode(QtWidgets.QListView.Adjust)
        lista.itemSelectionChanged.connect(self._punto_elegido_cambio)
        self.SELECTOR_POSTURA.currentIndexChanged.connect(self._revisar_puntos)
        # QListWidget NO TRAE TEXTO DE «LISTA VACÍA» EN Qt 5: LA INVITACIÓN VA ENCIMA, SIN ESTORBARLE AL RATÓN
        self.nota_sin_puntos = QtWidgets.QLabel(INVITACION_SIN_PUNTOS)
        self.nota_sin_puntos.setProperty("rol", "nota")
        self.nota_sin_puntos.setAlignment(QtCore.Qt.AlignCenter)
        self.nota_sin_puntos.setWordWrap(True)
        self.nota_sin_puntos.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        diseno = QtWidgets.QVBoxLayout(lista.viewport())
        diseno.setContentsMargins(8, 4, 8, 4)
        diseno.addWidget(self.nota_sin_puntos)
        for movimiento in programas.MOVIMIENTOS:  # LOS DEL FORMATO, NO ESCRITOS A MANO
            self.SELECTOR_TIPO_PUNTO.addItem(movimiento.capitalize(), movimiento)
        # EL SLIDER CUENTA DE 5 EN 5: CADA UNIDAD SUYA ES UN 5 %, DENTRO DEL RANGO QUE VALIDA programa.py
        self.SLIDER_VELOCIDAD_PUNTO.setRange(math.ceil(programas.VELOCIDAD_MIN / PASO_VELOCIDAD_PUNTO),
                                             programas.VELOCIDAD_MAX // PASO_VELOCIDAD_PUNTO)
        self.SLIDER_VELOCIDAD_PUNTO.setPageStep(1)
        hueco = self.VS_VELOCIDAD_PUNTO.sizePolicy()
        hueco.setRetainSizeWhenHidden(True)  # SIN PUNTO ELEGIDO EL VALOR SE ESCONDE Y LA FILA NO SE MUEVE
        self.VS_VELOCIDAD_PUNTO.setSizePolicy(hueco)
        self.BTN_EDITAR_PUNTOS.clicked.connect(self.abrir_editor)
        self.BTN_CERRAR_PUNTOS.clicked.connect(self.cerrar_editor)
        self.BTN_PUNTO_NUEVO.clicked.connect(self.agregar_punto)
        self.BTN_REENSENAR.clicked.connect(self.reensenar_punto)
        self.BTN_QUITAR_PUNTO.clicked.connect(self.quitar_punto)
        self.BTN_SUBIR_PUNTO.clicked.connect(lambda: self.mover_punto(-1))
        self.BTN_BAJAR_PUNTO.clicked.connect(lambda: self.mover_punto(+1))
        self.BTN_PASO_ATRAS.clicked.connect(lambda: self.paso(-1))
        self.BTN_PASO_ADELANTE.clicked.connect(lambda: self.paso(+1))
        self.BTN_VISTA_PREVIA.clicked.connect(self.vista_previa)
        self.SELECTOR_TIPO_PUNTO.currentIndexChanged.connect(
            lambda: self._cambiar_el_elegido(movimiento=self.SELECTOR_TIPO_PUNTO.currentData()))
        # AL ARRASTRAR SE VE EL VALOR, Y SE GUARDA UNA SOLA VEZ: AL SOLTAR (CON EL TECLADO, EN CADA PASO)
        self.SLIDER_VELOCIDAD_PUNTO.setTracking(False)
        self.SLIDER_VELOCIDAD_PUNTO.sliderMoved.connect(
            lambda valor: self.VS_VELOCIDAD_PUNTO.setText(f"{valor * PASO_VELOCIDAD_PUNTO} %"))
        self.SLIDER_VELOCIDAD_PUNTO.valueChanged.connect(
            lambda valor: self._cambiar_el_elegido(velocidad=valor * PASO_VELOCIDAD_PUNTO))

    @property
    def editando(self):
        return self.editado is not None

    def fila_elegida(self):
        """La fila del punto elegido en la lista (la primera es 0), o None.

        Se pregunta por la SELECCIÓN, no por el renglón «actual»: Qt vuelve actual el primer renglón con solo
        darle el foco a la lista, sin que nadie haya elegido nada.
        """
        filas = self.LISTA_PUNTOS.selectionModel().selectedRows()
        return filas[0].row() if filas else None

    def abrir_editor(self):
        """✎ Editar: la columna PUNTOS, con los puntos del programa elegido. Una grabación no se edita por puntos."""
        if self._nuevo_elegido():
            self._editar(self._programa_nuevo(), None)
            return
        p = self.programa_elegido()
        if p is None:
            return
        if p.tipo != "puntos":
            self.avisar(MOTIVO_NO_SE_EDITA.replace("\n", " "), "aviso", "Programa", programa=p.nombre)
            return
        self._editar(p, self.SELECTOR_PROGRAMA.currentData())

    def _editar(self, programa, archivo):
        """Pone `programa` en el editor, y lo abre si estaba cerrado."""
        fila = self.fila_elegida() if self.editando and archivo == self.archivo_editado else None
        if not self.editando:
            # VISIÓN LE CEDE SU LUGAR. NO SE TOCA LO QUE LA PERSONA TIENE RECORDADO: AL CERRAR, VUELVE COMO ESTABA
            self._vision_contraida_antes = self.FONDO_VISION.isHidden()
            self.contraer_vision(True, recordar=False)
            self.FONDO_PUNTOS.show()
        self.editado, self.archivo_editado = programa, archivo
        self._pintar_puntos(fila if fila is not None and fila < len(programa.puntos) else None)

    def cerrar_editor(self):
        """✕ Cerrar: la pantalla queda como estaba antes de abrir el editor."""
        if not self.editando:
            return
        self.editado = self.archivo_editado = None
        self.FONDO_PUNTOS.hide()
        self.contraer_vision(self._vision_contraida_antes, recordar=False)
        self._actualizar_editor()

    def _abrir_vision(self):
        """Clic en la franja «▸ VISIÓN». PUNTOS ocupa el lugar de VISIÓN: con el editor abierto, primero se cierra."""
        self.cerrar_editor()
        self.contraer_vision(False)

    def _pintar_puntos(self, elegir=None):
        """La lista y su nota, desde self.editado. elegir: la fila que queda elegida (None = ninguna)."""
        puntos, lista = self.editado.puntos, self.LISTA_PUNTOS
        lista.blockSignals(True)  # SE AVISA UNA SOLA VEZ, AL FINAL
        lista.clear()
        imposibles = self._imposibles(self.editado)  # SE REVISA EN CADA PINTADA: TRAS CADA CAMBIO, PUES
        for numero, punto in enumerate(puntos, start=1):
            item = QtWidgets.QListWidgetItem(f"{numero}   {punto.movimiento.capitalize()} · {punto.velocidad} %")
            if numero in imposibles:  # EN ROJO, COMO LOS ERRORES DEL REGISTRO, Y CON SU MOTIVO DEBAJO
                motivo = imposibles[numero]
                item.setText(f"{item.text()}   ⚠\n{motivo[:1].upper()}{motivo[1:]}")
                item.setForeground(QtGui.QColor(tema.TOKENS["rojo"]))
                item.setData(QtCore.Qt.UserRole, MARCA_IMPOSIBLE)
                item.setToolTip(f"<qt>Punto {numero} ({punto.movimiento}): {html.escape(motivo)}</qt>")
            lista.addItem(item)
        lista.setCurrentRow(-1 if elegir is None else elegir)
        lista.blockSignals(False)
        self.nota_sin_puntos.setVisible(not puntos)
        nota = f"{self.archivo_editado or 'Programa nuevo'} · {_cuantos_puntos(len(puntos))}"
        # UN NOMBRE DE ARCHIVO LARGO NO ENSANCHA LA COLUMNA: SE RECORTA POR EN MEDIO Y VA COMPLETO EN LA AYUDA
        self.NOTA_PUNTOS.ensurePolished()
        margenes = self.FONDO_PUNTOS.layout().contentsMargins()
        ancho = self.FONDO_PUNTOS.minimumWidth() - margenes.left() - margenes.right()
        self.NOTA_PUNTOS.setText(self.NOTA_PUNTOS.fontMetrics().elidedText(nota, QtCore.Qt.ElideMiddle, ancho))
        self.NOTA_PUNTOS.setToolTip(nota)
        self._punto_elegido_cambio()

    def _imposibles(self, programa):
        """{número de punto: por qué no se puede llegar a él}, para marcarlo en la lista.

        Lo decide plan_puntos, lo mismo que decide si el programa arranca: aquí no se calcula nada más. Se
        revisa desde el primer punto y no desde donde está el robot ahora: la marca es del programa (el tramo
        hasta el primer punto se revisa al reproducir, al dar un paso y en la vista previa).
        """
        if not programa.puntos or self.robot.cinematica_inversa is None:
            return {}
        _, problemas = plan_puntos(self.robot, programa.puntos[0].servos, programa, self.postura)
        return {numero: texto.removeprefix(f"Punto {numero} ({programa.puntos[numero - 1].movimiento}): ")
                for numero, texto in problemas}

    def _revisar_puntos(self):
        """Cambió la postura: un punto que se podía alcanzar puede ya no poderse, y al revés."""
        if self.editando:
            self._pintar_puntos(self.fila_elegida())

    def _punto_elegido_cambio(self):
        """La fila «del punto elegido» muestra su tipo y su velocidad, o se apaga si no hay ninguno elegido."""
        if not self.editando:
            return
        fila = self.fila_elegida()
        punto = self.editado.puntos[fila] if fila is not None else None
        self.SUB_PUNTO_ELEGIDO.setText(f"PUNTO {fila + 1}" if punto else "NINGÚN PUNTO ELEGIDO")
        for control in (self.SELECTOR_TIPO_PUNTO, self.SLIDER_VELOCIDAD_PUNTO):
            control.blockSignals(True)  # AQUÍ SOLO SE PINTA: NO ES UN CAMBIO DE LA PERSONA
        self.SELECTOR_TIPO_PUNTO.setCurrentIndex(self.SELECTOR_TIPO_PUNTO.findData(punto.movimiento) if punto else -1)
        self.SLIDER_VELOCIDAD_PUNTO.setValue(round(punto.velocidad / PASO_VELOCIDAD_PUNTO) if punto
                                             else self.SLIDER_VELOCIDAD_PUNTO.minimum())
        for control in (self.SELECTOR_TIPO_PUNTO, self.SLIDER_VELOCIDAD_PUNTO):
            control.blockSignals(False)
        self.VS_VELOCIDAD_PUNTO.setText(f"{punto.velocidad} %" if punto else "")
        self.VS_VELOCIDAD_PUNTO.setVisible(punto is not None)
        self._actualizar_editor()

    def _actualizar_editor(self):
        """UN solo lugar decide qué está encendido en el editor: según el programa y el punto elegidos, y
        apagado todo lo que edita o mueve mientras se reproduce, se graba o corre la vista previa."""
        libre = self._libre()
        self.BTN_EDITAR_PUNTOS.setEnabled(libre and self.tipo_elegido == "puntos")
        self.BTN_EDITAR_PUNTOS.setToolTip(MOTIVO_NO_SE_EDITA if self.tipo_elegido == "grabacion" else self._ayuda_editar)
        if not self.editando:
            return
        cuantos, fila = len(self.editado.puntos), self.fila_elegida()
        elegido = libre and fila is not None
        self.BTN_PUNTO_NUEVO.setEnabled(libre)
        self.BTN_REENSENAR.setEnabled(elegido)
        self.BTN_QUITAR_PUNTO.setEnabled(elegido and cuantos > 1)  # EL FORMATO PIDE AL MENOS UN PUNTO
        self.BTN_QUITAR_PUNTO.setToolTip("Un programa necesita al menos un punto" if cuantos == 1 else self._ayuda_quitar)
        self.BTN_SUBIR_PUNTO.setEnabled(elegido and fila > 0)
        self.BTN_BAJAR_PUNTO.setEnabled(elegido and fila < cuantos - 1)
        for control in (self.ETIQUETA_TIPO_PUNTO, self.SELECTOR_TIPO_PUNTO, self.ETIQUETA_VELOCIDAD_PUNTO,
                        self.SLIDER_VELOCIDAD_PUNTO):
            control.setEnabled(elegido)
        # SIN PUNTO ELEGIDO, EL PASO ADELANTE VA AL PRIMERO Y EL PASO ATRÁS AL ÚLTIMO
        self.BTN_PASO_ATRAS.setEnabled(libre and cuantos > 0 and fila != 0)
        self.BTN_PASO_ADELANTE.setEnabled(libre and cuantos > 0 and fila != cuantos - 1)
        self.BTN_VISTA_PREVIA.setEnabled(libre and cuantos > 0)

    def _puede_editar(self):
        return self.editando and self._libre()

    def _cambiar_puntos(self, puntos, elegir=None):
        """EL ÚNICO lugar por donde cambia el programa que se edita: se guarda solo (no hay botón de guardar)
        y la lista se vuelve a pintar. Devuelve False si no se pudo guardar: entonces nada cambia."""
        nuevo = dataclasses.replace(self.editado, puntos=tuple(puntos))
        if self.archivo_editado is None:
            return self._guardar_el_nuevo(nuevo, elegir)
        ruta = os.path.join(self.carpeta_programas, self.archivo_editado)
        try:
            programas.guardar(nuevo, ruta)
        except OSError as e:
            self.avisar(f"No se pudo guardar {self.archivo_editado}: {e}", "error", "Editor", ruta=ruta)
            self._pintar_puntos(self.fila_elegida())  # LA FILA VUELVE A DECIR LO QUE SÍ ESTÁ GUARDADO
            return False
        self.editado = nuevo
        self._pintar_puntos(elegir)
        self.INFO_PROGRAMA.setText(self._resumen(nuevo))
        return True

    def _guardar_el_nuevo(self, programa, elegir):
        """El primer punto de un programa nuevo: aquí se le pide el nombre, con las reglas de toda grabación
        (_guardar_con_nombre). Devuelve False si la persona no le da nombre: entonces el punto no se guarda."""
        guardado = self._guardar_con_nombre(f"programa-{dt.datetime.now():%Y-%m-%d-%H%M}", "puntos",
                                            lambda nombre: dataclasses.replace(programa, nombre=nombre))
        if guardado is None:
            self.avisar("El punto no se guardó: el programa nuevo se quedó sin nombre.", "aviso", "Editor")
            return False
        archivo, _ = guardado
        self.ajustes.setValue("programa/ultimo", archivo)
        self.recargar_programas()  # APARECE EN EL SELECTOR Y QUEDA ELEGIDO; EL EDITOR, QUE SIGUE AL SELECTOR, YA ES EL SUYO
        if self.editando:
            self._pintar_puntos(elegir)
        self.avisar(f"Programa guardado: {archivo}", "ok", "Programa", ruta=os.path.join(self.carpeta_programas, archivo))
        return True

    def agregar_punto(self):
        """+ Punto: la pose actual del robot, después del punto elegido (o al final si no hay ninguno elegido)."""
        if not self._puede_editar():
            return
        puntos, fila = list(self.editado.puntos), self.fila_elegida()
        lugar = len(puntos) if fila is None else fila + 1
        puntos.insert(lugar, programas.Punto(tuple(self.servos)))
        if self._cambiar_puntos(puntos, elegir=lugar):
            self.avisar(f"Punto {lugar + 1} guardado con la pose actual.", "ok", "Editor",
                        archivo=self.archivo_editado, servos=self.servos, posicion_mm=self.posicion)

    def reensenar_punto(self):
        """Re-enseñar: el punto elegido toma la pose actual del robot. Conserva su tipo y su velocidad."""
        fila = self.fila_elegida()
        if not self._puede_editar() or fila is None:
            return
        if not self._confirmar("Re-enseñar", f"¿Reemplazar el punto {fila + 1} con la pose actual del robot?",
                               "Reemplazar", "Cancelar"):
            return
        puntos = list(self.editado.puntos)
        puntos[fila] = dataclasses.replace(puntos[fila], servos=tuple(self.servos))
        if self._cambiar_puntos(puntos, elegir=fila):
            self.avisar(f"Punto {fila + 1} re-enseñado con la pose actual.", "ok", "Editor",
                        archivo=self.archivo_editado, servos=self.servos, posicion_mm=self.posicion)

    def quitar_punto(self):
        """Quitar: el punto elegido. El único punto de un programa no se quita (el formato pide al menos uno)."""
        fila = self.fila_elegida()
        if not self._puede_editar() or fila is None or len(self.editado.puntos) < 2:
            return
        puntos = list(self.editado.puntos)
        del puntos[fila]
        if self._cambiar_puntos(puntos, elegir=min(fila, len(puntos) - 1)):
            self.avisar(f"Punto {fila + 1} quitado.", "ok", "Editor", archivo=self.archivo_editado)

    def mover_punto(self, lugares):
        """▲ Subir (-1) y ▼ Bajar (+1): el punto elegido cambia de lugar con su vecino, y sigue elegido."""
        fila = self.fila_elegida()
        if not self._puede_editar() or fila is None or not 0 <= fila + lugares < len(self.editado.puntos):
            return
        puntos, destino = list(self.editado.puntos), fila + lugares
        puntos[fila], puntos[destino] = puntos[destino], puntos[fila]
        if self._cambiar_puntos(puntos, elegir=destino):
            self.avisar(f"Punto {fila + 1} movido al lugar {destino + 1}.", "ok", "Editor", archivo=self.archivo_editado)

    def _cambiar_el_elegido(self, **cambios):
        """El tipo (movimiento=) o la velocidad (velocidad=) del punto elegido, desde su fila."""
        fila = self.fila_elegida()
        if not self._puede_editar() or fila is None:
            return
        puntos = list(self.editado.puntos)
        cambiado = dataclasses.replace(puntos[fila], **cambios)
        if cambiado == puntos[fila]:
            return
        puntos[fila] = cambiado
        if self._cambiar_puntos(puntos, elegir=fila):
            self.avisar(f"Punto {fila + 1}: {cambiado.movimiento}, {cambiado.velocidad} %.", "ok", "Editor",
                        archivo=self.archivo_editado)

    def paso(self, sentido):
        """Paso adelante (+1) y Paso atrás (-1): el robot va al punto siguiente o al anterior, y ahí se detiene.

        Se llega con el tipo y la velocidad DEL PUNTO AL QUE SE LLEGA, también hacia atrás: lo decide
        tramo_a_punto, igual que al reproducir. El punto queda elegido en la lista: ahí va el robot.
        Sin punto elegido, adelante va al primero y atrás al último.
        """
        if not self._puede_editar() or not self._puede_planear():
            return
        cuantos, fila = len(self.editado.puntos), self.fila_elegida()
        destino = fila + sentido if fila is not None else 0 if sentido > 0 else cuantos - 1
        if not 0 <= destino < cuantos:
            return
        punto = self.editado.puntos[destino]
        tramo, motivo = tramo_a_punto(self.robot, self.servos, punto, self.postura)
        if tramo is None:  # NO SE MUEVE, Y LA LISTA SIGUE DICIENDO DÓNDE ESTÁ
            self.avisar(f"No se mueve: Punto {destino + 1} ({punto.movimiento}): {motivo}", "aviso", "Paso a paso",
                        archivo=self.archivo_editado, desde_mm=self.posicion, postura=self.postura)
            return
        self.LISTA_PUNTOS.setCurrentRow(destino)
        self.avisar(f"Paso al punto {destino + 1} de {cuantos} ({punto.movimiento}, {punto.velocidad} %)", "info",
                    "Paso a paso", archivo=self.archivo_editado, velocidad=self.velocidad, postura=self.postura)
        self.ejecutar(tramo, origen="Paso a paso")

    def vista_previa(self):
        """Vista previa: recorre el programa SOLO en la simulación, desde donde está el robot.

        Es el mismo plan que haría Reproducir, pero a la placa no se le manda nada; al terminar (o al detenerla),
        el dibujo vuelve a donde está el robot de verdad.
        """
        if not self._puede_editar() or not self.editado.puntos or not self._puede_planear():
            return
        plan, problemas = plan_puntos(self.robot, self.servos, self.editado, self.postura)
        if self._no_arranca(self.editado, problemas, "Vista previa"):
            return
        segundos = duracion_ms(self.robot, plan) / 1000 * 100 / self.velocidad
        self.avisar(f"Vista previa de «{self.editado.nombre}» (~{segundos:.1f} s): el robot no se mueve.", "info",
                    "Vista previa", archivo=self.archivo_editado, duracion_s=segundos, velocidad=self.velocidad)
        self.ejecutar(plan, origen="Vista previa", solo_simulacion=True)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% VISIÓN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def _configurar_vision(self):
        verde = self.robot.vision.verde.minimo
        self.SLIDER_TONO.setRange(0, 255)
        self.SLIDER_SATURACION.setRange(0, 255)
        self.SLIDER_TONO.setValue(int(verde[0]))
        self.SLIDER_SATURACION.setValue(int(verde[1]))
        self.VS_TONO.setText(str(self.SLIDER_TONO.value()))
        self.VS_SATURACION.setText(str(self.SLIDER_SATURACION.value()))
        self.SLIDER_TONO.valueChanged.connect(lambda v: self.VS_TONO.setText(str(v)))
        self.SLIDER_SATURACION.valueChanged.connect(lambda v: self.VS_SATURACION.setText(str(v)))
        self.GUARDAR.clicked.connect(self.guardar_detecciones)
        self.CARGA.clicked.connect(self.cargar_imagen)
        self.BTN_CONTRAER_VISION.clicked.connect(lambda: self.contraer_vision(True))
        self.FRANJA_VISION.clicked.connect(self._abrir_vision)
        self.contraer_vision(str(self.ajustes.value(COLUMNAS["vision"][2], "false")).lower() == "true", recordar=False)

    def contraer_columna(self, nombre, contraida, recordar=True):
        """Oculta una columna con franja (VISIÓN a la izquierda, ACCIONES a la derecha): queda la franja que la reabre y la
        simulación toma el espacio. Se recuerda al reabrir. UN solo mecanismo para las dos columnas."""
        fondo, franja, ajuste = COLUMNAS[nombre]
        getattr(self, fondo).setVisible(not contraida)
        getattr(self, franja).setVisible(contraida)
        if recordar:
            self.ajustes.setValue(ajuste, "true" if contraida else "false")

    def contraer_vision(self, contraida, recordar=True):
        """Oculta la columna de visión a la izquierda (queda la franja «▸ VISIÓN»). Se recuerda al reabrir."""
        self.contraer_columna("vision", contraida, recordar)

    def contraer_acciones(self, contraida, recordar=True):
        """Oculta la columna ACCIONES a la derecha (queda la franja «◂ ACCIONES»): el visor crece a lo ancho."""
        self.contraer_columna("acciones", contraida, recordar)

    def _configurar_acciones(self):
        self.BTN_CONTRAER_ACCIONES.clicked.connect(lambda: self.contraer_acciones(True))
        self.FRANJA_ACCIONES.clicked.connect(lambda: self.contraer_acciones(False))
        self.contraer_acciones(str(self.ajustes.value(COLUMNAS["acciones"][2], "false")).lower() == "true", recordar=False)

    def _configurar_visor(self):
        """Los botones de cámara solo mueven la vista 3D (se apagan si no está a la vista) y «Números» recuerda cómo quedó."""
        self.visor.vista_3d_a_la_vista.connect(self._camara_habilitada)
        self._camara_habilitada(self.visor.vista_3d_visible())
        numeros = str(self.ajustes.value(AJUSTE_NUMEROS, "true")).lower() != "false"
        self.CHK_NUMEROS_REJILLA.setChecked(numeros)
        self.escena.mostrar_numeros(numeros)
        self.CHK_NUMEROS_REJILLA.toggled.connect(self._numeros_cambiado)

    def _numeros_cambiado(self, visible):
        self.escena.mostrar_numeros(visible)
        self.ajustes.setValue(AJUSTE_NUMEROS, "true" if visible else "false")

    def _camara_habilitada(self, a_la_vista):
        for boton in (self.BTN_VISTA_ISO, self.BTN_VISTA_ARRIBA, self.BTN_VISTA_FRENTE, self.BTN_VISTA_LADO, self.BTN_ENCUADRAR):
            boton.setEnabled(a_la_vista)

    def guardar_detecciones(self):
        # EN EL SIGUIENTE FRAME, LOS PEDÚNCULOS DETECTADOS REEMPLAZAN LA LISTA DE COSECHA
        self.guardar_en_siguiente_frame = True

    def cargar_imagen(self):
        """Procesa una foto del disco: sirve para probar la visión sin la cámara del celular."""
        ruta, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Imagen de aguacates", "",
                                                        "Imágenes (*.png *.jpg *.jpeg *.bmp)")
        if ruta:
            self.usar_imagen(ruta)

    def usar_imagen(self, ruta):
        imagen = cv2.imread(ruta)
        if imagen is None:
            self.avisar(f"No se pudo leer la imagen: {ruta}", "error", "Visión", ruta=ruta)
            return
        self.imagen_fija = imagen
        self.temporizador_video.start(30)  # SE REPROCESA EN CADA TICK PARA QUE LOS SLIDERS DE COLOR RESPONDAN

    def _actualizar_frame(self):
        if self.imagen_fija is not None:
            frame = self.imagen_fija
        else:
            ok, frame = self.video.read()
            if not ok:
                return
        self.procesar_imagen(frame)

    def procesar_imagen(self, frame):
        h_min, s_min = self.SLIDER_TONO.value(), self.SLIDER_SATURACION.value()
        self.VS_TONO.setText(str(h_min))
        self.VS_SATURACION.setText(str(s_min))
        resultado, detecciones, _, _ = detectar_aguacates(frame, self.robot.vision, h_min, s_min)

        # SI LA CÁMARA VA EN EL EFECTOR, SU POSE ES LA DEL BRAZO EN ESTE MOMENTO
        alto, ancho = frame.shape[:2]
        q = servo_a_dh(self.robot, self.servos)
        pedunculos = []
        for det in detecciones:
            p = pedunculo_en_base(det, ancho, alto, self.robot, q)
            pedunculos.append(tuple(float(c) for c in p))
            color = AMARILLO if self.robot.cinematica_inversa and alcanzable(self.robot, p) else ROJO
            u, v = (int(c) for c in det["pedunculo_px"])
            cv2.putText(resultado, f"({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}) mm", (u, v),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if self.guardar_en_siguiente_frame:
            self.puntos_cosecha = pedunculos
            self.guardar_en_siguiente_frame = False
            self.avisar(f"Guardados {len(pedunculos)} pedúnculos para cosechar.", "ok", "Visión",
                        pedunculos_mm=pedunculos)

        self.ORIGINAL.setPixmap(cv_a_pixmap(frame))
        self.RESULTADOS.setPixmap(cv_a_pixmap(resultado))
        self.L_CONTADOR.setText(str(len(detecciones)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for lista in (self.LISTA_REGISTRO, self.LISTA_PUNTOS):
            lista.scheduleDelayedItemsLayout()  # LOS RENGLONES SE VUELVEN A MEDIR CON EL ANCHO NUEVO

    def closeEvent(self, event):
        self.temporizador_video.stop()
        self.temporizador_movimiento.stop()
        self.escena.plotter.close()
        if self.video is not None:
            self.video.release()
        self.enlace.close()
        event.accept()
