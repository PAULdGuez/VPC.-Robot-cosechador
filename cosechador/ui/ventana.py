"""Ventana principal: conecta los controles del .ui con el robot, la visión y el enlace.

Aquí no se calcula nada: los movimientos los planea cosechador.movimiento y esta ventana solo
ejecuta el plan paso a paso con un temporizador (sin bucles que bloqueen la interfaz).
"""
import dataclasses
import datetime as dt
import math
import os
from collections import deque

import html
import json

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtGui import QImage, QPixmap
from pyvistaqt import QtInteractor

from cosechador import registro as reg
from cosechador.hardware.enlace import abrir_enlace
from cosechador.movimiento import programa as programas
from cosechador.movimiento.grabadora import Grabadora, programa_de_grabacion
from cosechador.movimiento.trayectorias import (duracion_ms, plan_cosecha, plan_grabacion, plan_ir_y_volver,
                                                plan_puntos, posicion, tramo_a_punto, tramo_articular,
                                                tramo_continuo)
from cosechador.robot.cinematica import servo_a_dh
from cosechador.robot.cinematica_inversa import alcanzable
from cosechador.robot.espacio_trabajo import barrer, contorno_del_corte, giro_de_la_base
from cosechador.ui import tema
from cosechador.ui.escena3d import Escena
from cosechador.vision.camara import pedunculo_en_base
from cosechador.vision.deteccion import detectar_aguacates

RUTA_UI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "COSECHADOR_AGUACATE.ui")

ICONO_NIVEL = {"info": "●", "ok": "✔", "aviso": "⚠", "error": "✖"}
COLOR_NIVEL = {"info": "#2c6fbb", "ok": "#1e8e3e", "aviso": "#b7791f", "error": "#c0392b"}
FILTROS_REGISTRO = (("Todos", reg.NIVELES), ("Avisos y errores", ("aviso", "error")), ("Solo errores", ("error",)))

AMARILLO, ROJO = (0, 255, 255), (0, 0, 255)


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
TEXTO_GRABAR, TEXTO_DETENER_GRABACION = "⏺ Grabar", "⏹ Detener grabación"
TEXTO_VER_VOLUMEN, TEXTO_OCULTAR_VOLUMEN = "VOLUMEN DE TRABAJO", "OCULTAR VOLUMEN"
# PEDIR NOMBRE Y GUARDAR, SEGÚN EL TIPO DE PROGRAMA: (TÍTULO DEL DIÁLOGO, ETIQUETA DEL CAMPO, ORIGEN EN EL REGISTRO)
TEXTOS_GUARDAR = {"grabacion": ("Guardar grabación", "Nombre de la grabación:", "Grabación"),
                  "puntos": ("Guardar programa", "Nombre del programa:", "Programa")}
TEXTO_NUEVO_PROGRAMA = "+ Nuevo programa"  # EL ÚLTIMO RENGLÓN DEL SELECTOR, EL ÚNICO QUE NO ES UN ARCHIVO
# EDITOR DE PUNTOS
MOTIVO_NO_SE_EDITA = "Una grabación son muestras por tiempo:\nno se edita por puntos."
INVITACION_SIN_PUNTOS = ("Sin puntos.\n\nLleva el robot a una pose con las pestañas Servos o Punta X Y Z "
                         "y pulsa «+ Punto»: se guarda la pose actual.")
PASO_VELOCIDAD_PUNTO = 5  # LA VELOCIDAD DE UN PUNTO SE ELIGE DE 5 EN 5 %
MARCA_IMPOSIBLE = "imposible"  # LO QUE LLEVA EN UserRole EL RENGLÓN DE UN PUNTO AL QUE NO SE PUEDE LLEGAR


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

    def __init__(self, robot, enlace=None, fuente_camara=None, ajustes=None, carpeta_programas=None, registro=None):
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
        # ESCENA 3D INCRUSTADA EN EL RECUADRO "PANTALLA" DEL .ui
        self.escena = Escena(QtInteractor(self), robot)
        diseno = QtWidgets.QVBoxLayout(self.PANTALLA)
        diseno.setContentsMargins(0, 0, 0, 0)
        diseno.addWidget(self.escena.plotter.interactor)

        # ESP32 REAL SI HAY ROBOT_PUERTO; SI NO, LA VIRTUAL
        self.enlace = enlace or abrir_enlace(rangos=[(j.servo.minimo, j.servo.maximo) for j in robot.articulaciones])
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
        self._configurar_sliders()
        self._configurar_editor()
        self._configurar_programas()
        self._configurar_vision()
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
        """Dibuja el robot con esos ángulos de servo y se los manda a la placa.

        duracion_s: lo que dura este paso (para la grabación); None = un paso a velocidad 100 %.
        """
        self.servos = tuple(servos)
        self._dibujar(self.servos)
        self.poner_etiquetas()
        comando, respuestas = self.enlace.enviar(*self.servos)
        # SE GRABA AQUÍ, JUSTO DESPUÉS DE MANDARLA: LO GRABADO ES LO QUE RECIBIÓ LA PLACA, NO LO QUE DICEN LOS SLIDERS
        self.grabadora.anotar(self.servos, duracion_s if duracion_s is not None else self.robot.movimiento.ms_por_paso / 1000)
        for respuesta in respuestas:
            self.avisar(respuesta, "aviso", "ESP32", comando=comando, placa=self.enlace.nombre)

    def _dibujar(self, servos):
        """El robot de la simulación con esos ángulos de servo. Solo el dibujo: a la placa no le llega nada."""
        self.escena.dibujar(servo_a_dh(self.robot, servos))
        self.escena.agregar_al_recorrido(tuple(posicion(self.robot, servos)))

    def _ir_a_home(self):
        servos = self.robot.postura_home
        if servos is None:  # SIN SOLVER (O HOME INALCANZABLE): CADA SERVO AL CENTRO DE SU RANGO
            servos = tuple(j.servo.centro for j in self.robot.articulaciones)
        self.poner_servos(servos)
        self._sincronizar_sliders()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EJECUCIÓN DE PLANES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def ejecutar(self, plan, bloquear=True, origen="Movimiento", solo_simulacion=False):
        """Encola los pasos del plan; el temporizador los ejecuta uno por uno.

        bloquear=False (jog con los sliders): no se bloquean los controles y un movimiento nuevo
        reemplaza lo que faltaba del anterior.
        solo_simulacion=True (vista previa): los pasos se dibujan y a la placa no se le manda nada.
        """
        if self.en_vista_previa and not solo_simulacion:
            self._fin_del_movimiento()  # LO QUE FALTABA DE UNA VISTA PREVIA NUNCA SE MANDA A LA PLACA
        self.en_vista_previa = solo_simulacion
        if not bloquear:
            self.cola.clear()
        # LO QUE VA A RECORRER LA PUNTA, ANTES DE MOVERSE. UN MOVIMIENTO «COMPLETO» (PROGRAMA, COSECHA,
        # CALCULAR) EMPIEZA ESTELA NUEVA; EL JOG Y LOS SLIDERS LA SIGUEN ALARGANDO.
        camino = [self.posicion] + [posicion(self.robot, p.servos) for p in plan if p.servos is not None]
        self.escena.mostrar_plan(camino)
        if bloquear:
            self.escena.reiniciar_recorrido(self.posicion)
        self._origen_plan = origen
        self.cola.extend(plan)
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
            self.poner_servos(paso.servos, intervalo / 1000)
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
        self.SLIDER_VELOCIDAD.valueChanged.connect(self._velocidad_cambiada)
        self._velocidad_cambiada()
        if self.robot.cinematica_inversa is None:  # SIN SOLVER NO HAY JOG XYZ NI TRAYECTORIAS
            self.PESTANAS_MOVIMIENTO.setTabEnabled(PESTANA_XYZ, False)

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

        self.sliders_xyz = (self.SLIDER_X, self.SLIDER_Y, self.SLIDER_Z)
        self.valores_xyz = (self.VS_X, self.VS_Y, self.VS_Z)
        self._poner_rangos_xyz()
        for slider in self.sliders_xyz:
            slider.valueChanged.connect(self._jog_cartesiano)

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

    def _jog_cartesiano(self):
        """Pestaña Punta X Y Z: la punta va en línea recta al punto de los sliders; la CI mueve los servos."""
        destino = [float(slider.value()) for slider in self.sliders_xyz]
        distancia = math.dist(self.posicion, destino)
        if distancia < 0.5:
            return
        muestras = max(2, int(distancia / MM_POR_MUESTRA_JOG) + 1)
        plan, motivo = tramo_continuo(self.robot, self.servos, destino, self.postura, pasos=muestras)
        if plan is None:
            self.avisar(f"No se mueve: el camino hacia {tuple(int(c) for c in destino)} {motivo}"
                        if not motivo.startswith("Para") else f"No se mueve. {motivo}", "aviso", "Jog",
                        destino_mm=destino, desde_mm=self.posicion, postura=self.postura)
            self._sincronizar(self.sliders_xyz, self.posicion)  # EL SLIDER REGRESA A DONDE ESTÁ LA PUNTA
            return
        self.ejecutar(plan, bloquear=False, origen="Jog")

    @staticmethod
    def _sincronizar(sliders, valores):
        for slider, valor in zip(sliders, valores):
            slider.blockSignals(True)
            slider.setValue(int(round(valor)))
            slider.blockSignals(False)

    def _sincronizar_sliders(self):
        """Las dos pestañas muestran siempre la misma pose: servos en grados y punta en mm.
        Un grupo de sliders que la persona está arrastrando no se toca."""
        if not any(s.isSliderDown() for s in self.sliders_servo):
            self._sincronizar(self.sliders_servo, self.servos)
        if not any(s.isSliderDown() for s in self.sliders_xyz):
            self._sincronizar(self.sliders_xyz, self.posicion)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONSOLA DE REGISTRO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def _configurar_registro(self):
        for texto, niveles in FILTROS_REGISTRO:
            self.FILTRO_REGISTRO.addItem(texto, niveles)
        self.FILTRO_REGISTRO.currentIndexChanged.connect(self._repintar_registro)
        self.LISTA_REGISTRO.currentItemChanged.connect(self._mostrar_detalle)
        self.BTN_COPIAR_REGISTRO.clicked.connect(self.copiar_registro)
        self.BTN_EXPORTAR_REGISTRO.clicked.connect(self.exportar_registro)
        self.LISTA_REGISTRO.setWordWrap(True)  # LOS MENSAJES LARGOS PASAN AL SIGUIENTE RENGLÓN
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
        item = QtWidgets.QListWidgetItem(f"{evento.hora:%H:%M:%S}  {ICONO_NIVEL[evento.nivel]} {evento.nivel.upper()} · "
                                         f"{evento.origen}\n{evento.mensaje}")
        item.setForeground(QtGui.QColor(COLOR_NIVEL[evento.nivel]) if evento.nivel != "info" else QtGui.QColor("#1f2d3a"))
        item.setData(QtCore.Qt.UserRole, evento)
        item.setToolTip(evento.mensaje)
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
        # SE GRABA MOVIENDO DESDE LAS OTRAS DOS PESTAÑAS: EL AVISO VA EN LA ESQUINA DE LAS PESTAÑAS, A LA VISTA EN LAS TRES
        self.aviso_grabando = QtWidgets.QLabel("⏺ Grabando")
        self.aviso_grabando.setProperty("rol", "grabando")
        self.aviso_grabando.setToolTip("Se detiene en la pestaña Programa")
        self.PESTANAS_MOVIMIENTO.setCornerWidget(self.aviso_grabando, QtCore.Qt.TopRightCorner)
        self.aviso_grabando.hide()
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
        muestras = self.grabadora.terminar()
        self._boton_grabar(grabando=False)
        self._ocupado(False)
        self._programa_elegido()
        if muestras is None:
            self.avisar("No se guardó: no hubo movimiento mientras se grababa.", "aviso", "Grabación")
            return
        guardado = self._guardar_con_nombre(f"grabacion-{dt.datetime.now():%Y-%m-%d-%H%M}", "grabacion",
                                            lambda nombre: programa_de_grabacion(muestras, nombre, self.robot))
        if guardado is None:
            self.avisar("Grabación descartada (no se le dio nombre).", "aviso", "Grabación", muestras=len(muestras))
            return
        archivo, p = guardado
        self.ajustes.setValue("programa/ultimo", archivo)
        self.recargar_programas()  # APARECE EN EL SELECTOR Y QUEDA ELEGIDA
        self.avisar(f"Grabación guardada: {archivo} ({len(muestras)} muestras, {p.duracion:.1f} s)", "ok", "Grabación",
                    ruta=os.path.join(self.carpeta_programas, archivo), muestras=len(muestras), duracion_s=p.duracion)

    def _guardar_con_nombre(self, sugerido, tipo, hacer_programa):
        """Pide el nombre y guarda el programa que devuelve hacer_programa(nombre).

        Las reglas del nombre son UNAS para todo programa: si no sirve se vuelve a pedir, y si ya existe un
        archivo con ese nombre se pregunta antes de reemplazarlo. Devuelve (archivo, programa), o None si la
        persona cancela.
        """
        origen = TEXTOS_GUARDAR[tipo][2]
        while True:
            nombre = self._pedir_nombre(sugerido, tipo)
            if nombre is None:
                return None
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

    def _boton_grabar(self, grabando):
        """Aspecto de «grabando» (rol de tema.py): el botón en rojo y el aviso que se ve desde las tres pestañas."""
        self.aviso_grabando.setVisible(grabando)
        self.BTN_GRABAR.setText(TEXTO_DETENER_GRABACION if grabando else TEXTO_GRABAR)
        self.BTN_GRABAR.setProperty("rol", "grabando" if grabando else "")
        self.BTN_GRABAR.style().unpolish(self.BTN_GRABAR)  # Qt SOLO RELEE LA HOJA AL CAMBIAR UNA PROPIEDAD SI SE LE PIDE
        self.BTN_GRABAR.style().polish(self.BTN_GRABAR)

    def _pedir_nombre(self, sugerido, tipo="grabacion"):
        """El nombre que escribe la persona, o None si cancela."""
        titulo, etiqueta, _ = TEXTOS_GUARDAR[tipo]
        texto, ok = QtWidgets.QInputDialog.getText(self, titulo, etiqueta, QtWidgets.QLineEdit.Normal, sugerido)
        return texto if ok else None

    def _confirmar_sobrescribir(self, archivo, tipo="grabacion"):
        return self._confirmar(TEXTOS_GUARDAR[tipo][0], f"Ya existe «{archivo}». ¿Reemplazarlo?")

    def _confirmar(self, titulo, pregunta):
        """Sí o no, con «No» por omisión: para lo que reemplaza algo sin vuelta atrás."""
        respuesta = QtWidgets.QMessageBox.question(
            self, titulo, pregunta, QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        return respuesta == QtWidgets.QMessageBox.Yes

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
        if not self._confirmar("Re-enseñar", f"¿Reemplazar el punto {fila + 1} con la pose actual del robot?"):
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
        self.contraer_vision(str(self.ajustes.value("vision/contraida", "false")).lower() == "true", recordar=False)

    def contraer_vision(self, contraida, recordar=True):
        """Oculta la columna de visión a la izquierda (queda la franja «▸ VISIÓN»). La simulación toma
        el espacio; el panel derecho tiene ancho fijo, así que no se mueve. Se recuerda al reabrir."""
        self.FONDO_VISION.setVisible(not contraida)
        self.FRANJA_VISION.setVisible(contraida)
        if recordar:
            self.ajustes.setValue("vision/contraida", "true" if contraida else "false")

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
