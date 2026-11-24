"""Ventana principal: conecta los controles del .ui con el robot, la visión y el enlace.

Aquí no se calcula nada: los movimientos los planea cosechador.movimiento y esta ventana solo
ejecuta el plan paso a paso con un temporizador (sin bucles que bloqueen la interfaz).
"""
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
                                                plan_puntos, posicion, tramo_articular, tramo_continuo)
from cosechador.robot.cinematica import servo_a_dh
from cosechador.robot.cinematica_inversa import SinSolver, alcanzable, angulos_servo
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
        self.temporizador_movimiento = QtCore.QTimer(self)
        self.temporizador_movimiento.setTimerType(QtCore.Qt.PreciseTimer)  # LAS GRABACIONES RESPETAN SUS TIEMPOS
        self.temporizador_movimiento.timeout.connect(self._siguiente_paso)

        self._configurar_selectores()
        self._configurar_sliders()
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
        self.escena.dibujar(servo_a_dh(self.robot, self.servos))
        self.escena.agregar_al_recorrido(self.posicion)
        self.poner_etiquetas()
        comando, respuestas = self.enlace.enviar(*self.servos)
        # SE GRABA AQUÍ, JUSTO DESPUÉS DE MANDARLA: LO GRABADO ES LO QUE RECIBIÓ LA PLACA, NO LO QUE DICEN LOS SLIDERS
        self.grabadora.anotar(self.servos, duracion_s if duracion_s is not None else self.robot.movimiento.ms_por_paso / 1000)
        for respuesta in respuestas:
            self.avisar(respuesta, "aviso", "ESP32", comando=comando, placa=self.enlace.nombre)

    def _ir_a_home(self):
        try:
            servos = angulos_servo(self.robot, self.robot.home)
        except SinSolver:
            servos = None
        if servos is None:  # SIN SOLVER (O HOME INALCANZABLE): CADA SERVO AL CENTRO DE SU RANGO
            servos = tuple(j.servo.centro for j in self.robot.articulaciones)
        self.poner_servos(servos)
        self._sincronizar_sliders()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EJECUCIÓN DE PLANES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def ejecutar(self, plan, bloquear=True, origen="Movimiento"):
        """Encola los pasos del plan; el temporizador los ejecuta uno por uno.

        bloquear=False (jog con los sliders): no se bloquean los controles y un movimiento nuevo
        reemplaza lo que faltaba del anterior.
        """
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
            self.temporizador_movimiento.stop()
            self._sincronizar_sliders()
            self._ocupado(False)
            self.BTN_DETENER.setEnabled(False)
            return
        paso = self.cola.popleft()
        # EL SIGUIENTE PASO LLEGA CUANDO TERMINA ESTE: SU DURACIÓN, ESCALADA POR LA VELOCIDAD GENERAL
        ms = paso.ms if paso.ms is not None else (self.robot.movimiento.ms_por_paso if paso.servos is not None else 0)
        intervalo = max(1, int(round(ms * 100 / self.velocidad)))
        if paso.servos is not None:
            self.poner_servos(paso.servos, intervalo / 1000)
        if paso.aviso is not None:
            self.avisar(paso.aviso, paso.nivel, self._origen_plan)
        self.temporizador_movimiento.setInterval(intervalo)

    def detener(self):
        """■ Detener: el robot se queda donde iba y no se manda nada más a la placa."""
        if not self.en_movimiento and not self.cola:
            return
        self.cola.clear()
        self.temporizador_movimiento.stop()
        self._sincronizar_sliders()
        self._ocupado(False)
        self.BTN_DETENER.setEnabled(False)
        self.avisar(f"Detenido en {tuple(int(round(c)) for c in self.posicion)}.", "aviso", self._origen_plan,
                    posicion_mm=self.posicion, servos=self.servos)

    def _ocupado(self, si):
        """Bloquea los controles de movimiento mientras el robot se mueve (evita clics encimados)."""
        for control in (self.CALCULAR, self.COSECHAR, self.VER_VOLUMEN, self.BTN_REPRODUCIR, self.SELECTOR_PROGRAMA,
                        self.BTN_GRABAR, *self.sliders_servo, *self.sliders_xyz):
            control.setEnabled(not si)
        if self.SELECTOR_PROGRAMA.currentData() is None:  # SIN PROGRAMA ELEGIDO NO HAY QUÉ REPRODUCIR
            self.BTN_REPRODUCIR.setEnabled(False)
        if self.grabadora.grabando:  # MIENTRAS SE GRABA, SOLO SE MUEVE CON LOS SLIDERS
            for control in (self.CALCULAR, self.COSECHAR, self.BTN_REPRODUCIR, self.SELECTOR_PROGRAMA):
                control.setEnabled(False)

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
        alcance = int(math.ceil(self.robot.alcance))
        for slider, (lo, hi) in zip(self.sliders_xyz, ((-alcance, alcance), (-alcance, alcance), (0, alcance))):
            slider.setRange(lo, hi)
            slider.valueChanged.connect(self._jog_cartesiano)

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
        self.SELECTOR_PROGRAMA.currentIndexChanged.connect(self._programa_elegido)
        self.recargar_programas()

    def recargar_programas(self):
        """Lista los programas de la carpeta y deja elegido el último que se usó."""
        ultimo = self.ajustes.value("programa/ultimo", "")
        self.SELECTOR_PROGRAMA.blockSignals(True)
        self.SELECTOR_PROGRAMA.clear()
        for nombre in programas.listar(self.carpeta_programas):
            self.SELECTOR_PROGRAMA.addItem(nombre, nombre)
        indice = self.SELECTOR_PROGRAMA.findData(ultimo)
        self.SELECTOR_PROGRAMA.setCurrentIndex(indice if indice >= 0 else 0)
        self.SELECTOR_PROGRAMA.blockSignals(False)
        self._programa_elegido()

    def _programa_elegido(self):
        nombre = self.SELECTOR_PROGRAMA.currentData()
        self.BTN_REPRODUCIR.setEnabled(nombre is not None)
        if nombre is None:
            self.INFO_PROGRAMA.setText("No hay programas guardados")
            self.INFO_PROGRAMA.setToolTip(self.carpeta_programas)
            return
        self.ajustes.setValue("programa/ultimo", nombre)
        try:
            p = programas.cargar(os.path.join(self.carpeta_programas, nombre), self.robot)
        except programas.ErrorPrograma as e:
            self.INFO_PROGRAMA.setText("⚠ No se puede leer (ver registro)")
            self.avisar(str(e), "error", "Programa", archivo=nombre)
            return
        if p.tipo == "puntos":
            self.INFO_PROGRAMA.setText(f"Por puntos · {len(p.puntos)} puntos")
        else:
            self.INFO_PROGRAMA.setText(f"Grabación · {len(p.muestras)} muestras · {p.duracion:.1f} s")

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
        if problemas:
            extra = f" (y {len(problemas) - 1} más)" if len(problemas) > 1 else ""
            self.avisar(f"No arranca: {problemas[0][1]}{extra}", "aviso", "Programa",
                        programa=p.nombre, problemas=[m for _, m in problemas])
            return
        segundos = duracion_ms(self.robot, plan) / 1000 * 100 / self.velocidad
        self.avisar(f"Reproduciendo «{p.nombre}» (~{segundos:.1f} s)", "info", "Programa",
                    programa=p.nombre, tipo=p.tipo, duracion_s=segundos, velocidad=self.velocidad, postura=self.postura)
        self.ejecutar(plan, origen="Programa")

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
        sugerido = f"grabacion-{dt.datetime.now():%Y-%m-%d-%H%M}"
        while True:
            nombre = self._pedir_nombre(sugerido)
            if nombre is None:
                self.avisar("Grabación descartada (no se le dio nombre).", "aviso", "Grabación", muestras=len(muestras))
                return
            try:
                archivo = programas.nombre_de_archivo(nombre)
            except programas.ErrorPrograma as e:
                self.avisar(str(e), "aviso", "Grabación", escrito=nombre)
                continue
            ruta = os.path.join(self.carpeta_programas, archivo)
            if os.path.exists(ruta) and not self._confirmar_sobrescribir(archivo):
                sugerido = nombre
                continue
            p = programa_de_grabacion(muestras, nombre.strip(), self.robot)
            try:
                os.makedirs(self.carpeta_programas, exist_ok=True)
                programas.guardar(p, ruta)
            except OSError as e:
                self.avisar(f"No se pudo guardar {archivo}: {e}", "error", "Grabación", ruta=ruta)
                continue
            break
        self.ajustes.setValue("programa/ultimo", archivo)
        self.recargar_programas()  # APARECE EN EL SELECTOR Y QUEDA ELEGIDA
        self.avisar(f"Grabación guardada: {archivo} ({len(muestras)} muestras, {p.duracion:.1f} s)", "ok", "Grabación",
                    ruta=ruta, muestras=len(muestras), duracion_s=p.duracion)

    def _boton_grabar(self, grabando):
        """Aspecto de «grabando» (rol de tema.py): el botón en rojo y el aviso que se ve desde las tres pestañas."""
        self.aviso_grabando.setVisible(grabando)
        self.BTN_GRABAR.setText(TEXTO_DETENER_GRABACION if grabando else TEXTO_GRABAR)
        self.BTN_GRABAR.setProperty("rol", "grabando" if grabando else "")
        self.BTN_GRABAR.style().unpolish(self.BTN_GRABAR)  # Qt SOLO RELEE LA HOJA AL CAMBIAR UNA PROPIEDAD SI SE LE PIDE
        self.BTN_GRABAR.style().polish(self.BTN_GRABAR)

    def _pedir_nombre(self, sugerido):
        """El nombre que escribe la persona, o None si cancela."""
        texto, ok = QtWidgets.QInputDialog.getText(self, "Guardar grabación", "Nombre de la grabación:",
                                                   QtWidgets.QLineEdit.Normal, sugerido)
        return texto if ok else None

    def _confirmar_sobrescribir(self, archivo):
        respuesta = QtWidgets.QMessageBox.question(
            self, "Guardar grabación", f"Ya existe «{archivo}». ¿Reemplazarlo?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        return respuesta == QtWidgets.QMessageBox.Yes

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
        self.FRANJA_VISION.clicked.connect(lambda: self.contraer_vision(False))
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
        self.LISTA_REGISTRO.scheduleDelayedItemsLayout()  # LOS RENGLONES SE VUELVEN A MEDIR CON EL ANCHO NUEVO

    def closeEvent(self, event):
        self.temporizador_video.stop()
        self.temporizador_movimiento.stop()
        self.escena.plotter.close()
        if self.video is not None:
            self.video.release()
        self.enlace.close()
        event.accept()
