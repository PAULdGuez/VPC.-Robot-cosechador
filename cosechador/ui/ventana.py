"""Ventana principal: conecta los controles del .ui con el robot, la visión y el enlace.

Aquí no se calcula nada: los movimientos los planea cosechador.movimiento y esta ventana solo
ejecuta el plan paso a paso con un temporizador (sin bucles que bloqueen la interfaz).
"""
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
from cosechador.movimiento.trayectorias import (duracion_ms, plan_cosecha, plan_grabacion, plan_ir_y_volver,
                                                plan_puntos, posicion, tramo_articular, tramo_continuo)
from cosechador.robot.cinematica import servo_a_dh
from cosechador.robot.cinematica_inversa import SinSolver, alcanzable, angulos_servo
from cosechador.robot.espacio_trabajo import barrer
from cosechador.ui.escena3d import Escena
from cosechador.vision.camara import pedunculo_en_base
from cosechador.vision.deteccion import detectar_aguacates

RUTA_UI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "COSECHADOR_AGUACATE.ui")
FLECHA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iconos", "flecha-abajo.svg")

ICONO_NIVEL = {"info": "●", "ok": "✔", "aviso": "⚠", "error": "✖"}
COLOR_NIVEL = {"info": "#2c6fbb", "ok": "#1e8e3e", "aviso": "#b7791f", "error": "#c0392b"}
# EN macOS LA BARRA DE DESPLAZAMIENTO NATIVA NO RECIBE EL ESTILO DE SU LISTA (QUEDA DE 16 px Y CON UNA LÍNEA
# NEGRA QUE TAPA TEXTO): SE LE PONE DIRECTO A LA BARRA.
ESTILO_BARRA = """
QScrollBar:vertical { width: 8px; background: transparent; margin: 2px 1px; border: none; }
QScrollBar::handle:vertical { background: #c4c4c4; border: none; border-radius: 3px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: #9aa5b1; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; border: none; background: none; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
"""
FILTROS_REGISTRO = (("Todos", reg.NIVELES), ("Avisos y errores", ("aviso", "error")), ("Solo errores", ("error",)))

# MISMO LENGUAJE VISUAL QUE LOS BOTONES: BORDE NEGRO, ESQUINAS REDONDAS, FLECHA PROPIA
ESTILO_SELECTOR = """
QComboBox {{ background: white; border: 2px solid black; border-radius: 8px; padding: 2px 30px 2px 10px;
             font-size: 13px; color: #1f2d3a; min-height: 20px; }}
QComboBox:hover {{ background: #f3f6f9; }}
QComboBox:disabled {{ color: #999; border-color: #bbb; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; width: 26px;
                        border-left: 1px solid #ccc; }}
QComboBox::down-arrow {{ image: url("{flecha}"); width: 12px; height: 8px; }}
QComboBox QAbstractItemView {{ background: white; border: 2px solid black; selection-background-color: #2c3e50;
                               selection-color: white; outline: 0; padding: 4px; }}
"""

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


class VentanaPrincipal(QtWidgets.QMainWindow):

    def __init__(self, robot, enlace=None, fuente_camara=None, ajustes=None, carpeta_programas=None, registro=None):
        super().__init__()
        uic.loadUi(RUTA_UI, self)
        self.robot = robot
        self.registro = registro if registro is not None else reg.Registro()
        self._origen_plan = "Movimiento"
        self._configurar_registro()
        # EN macOS EL ESTILO NATIVO CALCULA LOS BOTONES MÁS CHICOS DE LO QUE SE DIBUJAN: CON BORDE PROPIO SE
        # ENCIMABAN 2 px AUNQUE EL LAYOUT PIDIERA 10 px DE SEPARACIÓN. QUE EL LAYOUT USE EL TAMAÑO REAL:
        for boton in self.findChildren(QtWidgets.QPushButton):
            boton.setAttribute(QtCore.Qt.WA_LayoutUsesWidgetRect)
        # EL ESTILO VA EN CADA SELECTOR: EN Qt EL ESTILO PROPIO DE UN CONTROL GANA AL DE LA VENTANA
        for selector in self.findChildren(QtWidgets.QComboBox):
            selector.setStyleSheet(ESTILO_SELECTOR.format(flecha=FLECHA.replace(os.sep, "/")))
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

        # ESTADO
        self.servos = None
        self.puntos_cosecha = list(robot.puntos_demo)
        self.guardar_en_siguiente_frame = False
        self.imagen_fija = None
        self.cola = deque()
        self.temporizador_movimiento = QtCore.QTimer(self)
        self.temporizador_movimiento.setTimerType(QtCore.Qt.PreciseTimer)  # LAS GRABACIONES RESPETAN SUS TIEMPOS
        self.temporizador_movimiento.timeout.connect(self._siguiente_paso)

        self._configurar_selectores()
        self._configurar_sliders()
        self._configurar_programas()
        self._configurar_vision()
        self.CALCULAR.clicked.connect(self.ir_y_volver)
        self.COSECHAR.clicked.connect(self.cosechar)
        self.VER_VOLUMEN.clicked.connect(self.ver_volumen)

        self._ir_a_home()
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

    def poner_servos(self, servos):
        """Dibuja el robot con esos ángulos de servo y se los manda a la placa."""
        self.servos = tuple(servos)
        self.escena.dibujar(servo_a_dh(self.robot, self.servos))
        self.poner_etiquetas()
        comando, respuestas = self.enlace.enviar(*self.servos)
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
        if paso.servos is not None:
            self.poner_servos(paso.servos)
        if paso.aviso is not None:
            self.avisar(paso.aviso, paso.nivel, self._origen_plan)
        # EL SIGUIENTE PASO LLEGA CUANDO TERMINA ESTE: SU DURACIÓN, ESCALADA POR LA VELOCIDAD GENERAL
        ms = paso.ms if paso.ms is not None else (self.robot.movimiento.ms_por_paso if paso.servos is not None else 0)
        self.temporizador_movimiento.setInterval(max(1, int(round(ms * 100 / self.velocidad))))

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
                        *self.sliders_servo, *self.sliders_xyz):
            control.setEnabled(not si)

    def _puede_planear(self):
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
        self.escena.mostrar_volumen(barrer(self.robot, paso=5, progreso=avance))
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
        for slider, etiqueta, j in zip(self.sliders_servo, (self.label_12, self.label_7, self.label_9), self.robot.articulaciones):
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
        self.LISTA_REGISTRO.setResizeMode(QtWidgets.QListView.Adjust)
        self.DETALLE_REGISTRO.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.DETALLE_REGISTRO.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)
        for area in (self.LISTA_REGISTRO, self.DETALLE_REGISTRO):
            area.verticalScrollBar().setStyleSheet(ESTILO_BARRA)
        self.DETALLE_REGISTRO.setViewportMargins(0, 0, 10, 0)  # QUE EL TEXTO NO QUEDE BAJO LA BARRA
        self.registro.suscribir(self._evento_registrado)
        self.BTN_DETENER.setEnabled(False)  # SOLO SE ENCIENDE MIENTRAS ALGO SE MUEVE

    def _item_de(self, evento):
        item = QtWidgets.QListWidgetItem(f"{evento.hora:%H:%M:%S}  {ICONO_NIVEL[evento.nivel]} {evento.nivel.upper()} · "
                                         f"{evento.origen}\n{evento.mensaje}")
        item.setForeground(QtGui.QColor(COLOR_NIVEL[evento.nivel]) if evento.nivel != "info" else QtGui.QColor("#1f2d3a"))
        item.setData(QtCore.Qt.UserRole, evento)
        item.setToolTip(evento.mensaje)
        return item

    def _evento_registrado(self, evento):
        if evento.nivel not in self.FILTRO_REGISTRO.currentData():
            return
        self.LISTA_REGISTRO.insertItem(0, self._item_de(evento))  # LO MÁS NUEVO ARRIBA
        while self.LISTA_REGISTRO.count() > reg.MAXIMO_EN_MEMORIA:
            self.LISTA_REGISTRO.takeItem(self.LISTA_REGISTRO.count() - 1)

    def _repintar_registro(self):
        self.LISTA_REGISTRO.clear()
        for evento in reversed(self.registro.filtrar(self.FILTRO_REGISTRO.currentData())):
            self.LISTA_REGISTRO.addItem(self._item_de(evento))

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
        return [self.LISTA_REGISTRO.item(i).data(QtCore.Qt.UserRole) for i in range(self.LISTA_REGISTRO.count())][::-1]

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
            self.INFO_PROGRAMA.setText(f"No hay programas en {self.carpeta_programas}.")
            return
        self.ajustes.setValue("programa/ultimo", nombre)
        try:
            p = programas.cargar(os.path.join(self.carpeta_programas, nombre), self.robot)
        except programas.ErrorPrograma as e:
            self.INFO_PROGRAMA.setText(f"⚠ {e}")
            return
        if p.tipo == "puntos":
            self.INFO_PROGRAMA.setText(f"Programa por puntos · {len(p.puntos)} puntos")
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
        if self.en_movimiento:
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

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% VISIÓN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def _configurar_vision(self):
        verde = self.robot.vision.verde.minimo
        self.SLIDER_color_1.setRange(0, 255)
        self.SLIDER_color_2.setRange(0, 255)
        self.SLIDER_color_1.setValue(int(verde[0]))
        self.SLIDER_color_2.setValue(int(verde[1]))
        self.LABEL1.setText(str(self.SLIDER_color_1.value()))
        self.LABEL2.setText(str(self.SLIDER_color_2.value()))
        self.SLIDER_color_1.valueChanged.connect(lambda v: self.LABEL1.setText(str(v)))
        self.SLIDER_color_2.valueChanged.connect(lambda v: self.LABEL2.setText(str(v)))
        self.GUARDAR.clicked.connect(self.guardar_detecciones)
        self.CARGA.clicked.connect(self.cargar_imagen)

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
        h_min, s_min = self.SLIDER_color_1.value(), self.SLIDER_color_2.value()
        self.LABEL1.setText(str(h_min))
        self.LABEL2.setText(str(s_min))
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

    def closeEvent(self, event):
        self.temporizador_video.stop()
        self.temporizador_movimiento.stop()
        self.escena.plotter.close()
        if self.video is not None:
            self.video.release()
        self.enlace.close()
        event.accept()
