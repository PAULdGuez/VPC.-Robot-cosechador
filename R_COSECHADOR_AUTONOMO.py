import os
import sys
import traceback

import cv2
import numpy as np
from PyQt5 import QtCore, QtTest, QtWidgets, uic
from PyQt5.QtGui import QImage, QPixmap, QSurfaceFormat
from PyQt5.QtWidgets import QApplication
from pyvistaqt import QtInteractor

from cinematica import HOME, LIMITES_SERVO, cinematica, servo_a_dh
from enlace_robot import abrir_enlace
from vision import detectar_aguacates, pedunculo_en_base

def DE_MASCARA_A_QT(mascara):
    bgr = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
    return DE_CV_A_QT(bgr)

def DE_CV_A_QT(imagen_cv):
    rgb_image = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qt_image)

class VentanaPrincipal(QtWidgets.QMainWindow):
    PASOS_TRAYECTORIA = 20   # PUNTOS INTERMEDIOS DE CADA MOVIMIENTO EN LINEA RECTA
    ESPERA_PASO_MS = 50      # PAUSA ENTRE PUNTOS (VELOCIDAD DE LA ANIMACION)

    def __init__(self):
        super().__init__()
        uic.loadUi(os.path.join(os.path.dirname(os.path.abspath(__file__)), "COSECHADOR_AGUACATE.ui"), self)

        # PARA LA SIMULACION, GUARDAMOS LAS COORDENADAS (mm) DE PEDUNCULOS DETECTADOS
        # DADO QUE PARA LA SIMULACION NECESITAMOS X,Y,Z, CREAMOS TRES LISTAS PARA SU FACILIDAD
        self.COORDENADAS_X = [5, 0, 430, -370, -723, 723, -1656, 1656, 0]
        self.COORDENADAS_Y = [-4, 0, 0, 0, 0, 0, 0, 0, 0]
        self.COORDENADAS_Z = [1057, 1065, 1718, 1368, 1552, 1552, 630, 630, 1065]
        self.CONTADOR = 0

        self.simulacion = cinematica()
        # USAMOS UN INTERACTOR PARA PODER INCRUSTAR LA SIMULACION EN LA VENTANA
        self.simulacion.plotter = QtInteractor(self)
        self.dibujo = QtWidgets.QVBoxLayout(self.PANTALLA)
        self.dibujo.setContentsMargins(0, 0, 0, 0)
        self.dibujo.addWidget(self.simulacion.plotter.interactor)

        # ESP32 REAL SI HAY ROBOT_PUERTO; SI NO, LA VIRTUAL
        self.enlace = abrir_enlace()
        for respuesta in self.enlace.dispositivo.leer_respuestas():
            print(f"[{self.enlace.nombre}] {respuesta}")

        self.posicion = HOME
        self.mover_a(HOME)
        self.simulacion.plotter.reset_camera()
        self.configurar_sliders()

        self.SLIDER_color_1.setRange(0, 255)
        self.SLIDER_color_2.setRange(0, 255)
        self.SLIDER_color_1.setValue(30)
        self.SLIDER_color_2.setValue(170)
        self.COSECHAR.setEnabled(True)
        self.GUARDAR.clicked.connect(self.GUARDAR_COORDENADAS)
        self.CARGA.clicked.connect(self.CARGAR_IMAGEN)
        self.imagen_fija = None
        self.COSECHAR.clicked.connect(self.COSECHAR_AGUACATES)

        self.video = self.abrir_camara()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.ACTUALIZAR_FRAME)
        if self.video is not None:
            self.timer.start(30)

        self.avisar(f"Conectado a: {self.enlace.nombre}")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CAMARA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def abrir_camara(self):
        """Abre la camara de ROBOT_CAMARA (URL de la app del celular o indice de webcam).

        Sin la variable no se abre ninguna: antes la IP fija bloqueaba el arranque si el
        celular no estaba en la red.
        """
        fuente = os.environ.get("ROBOT_CAMARA")
        if not fuente:
            self.ORIGINAL.setText("Sin cámara.\nDefine ROBOT_CAMARA, por ejemplo\nhttp://192.168.100.63:8080/video")
            return None
        try:
            fuente = int(fuente)  # INDICE DE WEBCAM ("0", "1"...); SI NO ES NUMERO, ES UNA URL
        except ValueError:
            pass
        video = cv2.VideoCapture(fuente)
        if not video.isOpened():
            self.ORIGINAL.setText(f"No se pudo abrir la cámara:\n{fuente}")
            return None
        return video

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MOVIMIENTO DEL ROBOT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def avisar(self, texto):
        print(texto)
        self.statusBar().showMessage(texto)

    def poner_servos(self, s1, s2, s3):
        """Dibuja el robot con esos angulos de servo y se los manda a la ESP32."""
        self.simulacion.COSECHADOR(*servo_a_dh(s1, s2, s3))
        self.simulacion.plotter.update()
        self.VS_1.setText(f"{s1:.0f}")
        self.VS_2.setText(f"{s2:.0f}")
        self.VS_3.setText(f"{s3:.0f}")
        comando, respuestas = self.enlace.enviar(s1, s2, s3)
        for respuesta in respuestas:
            self.avisar(f"[{self.enlace.nombre}] {respuesta}  (comando: {comando})")
        self.servos = (s1, s2, s3)
        self.posicion = tuple(self.simulacion.CD(*servo_a_dh(s1, s2, s3)))

    def mover_a(self, punto):
        servos = self.simulacion.angulos_servo(*punto)
        if servos is None:
            self.avisar(f"Punto fuera del volumen de trabajo: {punto}")
            return False
        self.poner_servos(*servos)
        self.sincronizar_sliders(*servos)
        return True

    def trayectoria(self, destino):
        """Mueve el efector en linea recta hasta destino. Antes de moverse comprueba TODO el camino."""
        origen = np.array(self.posicion, dtype=float)
        camino = [origen + (np.array(destino, dtype=float) - origen) * k
                  for k in np.linspace(0, 1, self.PASOS_TRAYECTORIA)]
        servos_camino = [self.simulacion.angulos_servo(*p) for p in camino]
        if any(s is None for s in servos_camino):
            self.avisar(f"El camino hacia {tuple(int(round(c)) for c in destino)} pasa por una zona que el robot no alcanza.")
            return False
        for servos in servos_camino:
            self.poner_servos(*servos)
            QtTest.QTest.qWait(self.ESPERA_PASO_MS)
        self.sincronizar_sliders(*servos_camino[-1])
        return True

    def ocupado(self, si):
        """Bloquea los controles de movimiento mientras el robot se mueve (evita clics encimados)."""
        for control in (self.CALCULAR, self.COSECHAR, self.VER_VOLUMEN,
                        self.SLIDER_1, self.SLIDER_2, self.SLIDER_3):
            control.setEnabled(not si)

    def sincronizar_sliders(self, s1, s2, s3):
        for slider, valor in ((self.SLIDER_1, s1), (self.SLIDER_2, s2), (self.SLIDER_3, s3)):
            slider.blockSignals(True)
            slider.setValue(int(round(valor)))
            slider.blockSignals(False)

    def COSECHAR_AGUACATES(self):
        """Visita cada pedunculo guardado y regresa a HOME."""
        puntos = list(zip(self.COORDENADAS_X, self.COORDENADAS_Y, self.COORDENADAS_Z))
        self.ocupado(True)
        try:
            cosechados = 0
            for i, punto in enumerate(puntos, start=1):
                self.avisar(f"Cosechando {i}/{len(puntos)}: {punto}")
                if not self.simulacion.punto_en_volumen(*punto):
                    self.avisar(f"Punto fuera del volumen de trabajo: {punto}")
                    continue
                # SI LA LINEA RECTA CRUZA UNA ZONA INALCANZABLE (P. EJ. EL EJE DE LA BASE), PASA POR HOME
                if self.trayectoria(punto) or (self.trayectoria(HOME) and self.trayectoria(punto)):
                    cosechados += 1
            self.trayectoria(HOME)
            self.avisar(f"Cosecha terminada: {cosechados} de {len(puntos)} puntos alcanzados.")
        finally:
            self.ocupado(False)

    def configurar_sliders(self):
        def actualizar_robot():
            # LOS SLIDERS SON LOS ANGULOS DE CADA SERVO (LO MISMO QUE RECIBE LA ESP32)
            self.poner_servos(self.SLIDER_1.value(), self.SLIDER_2.value(), self.SLIDER_3.value())

        def aplicar_ci():
            try:
                destino = (float(self.TE_X.toPlainText()), float(self.TE_Y.toPlainText()),
                           float(self.TE_Z.toPlainText()))
            except ValueError:
                self.avisar("Escribe números en X, Y y Z (en mm).")
                return
            if not self.simulacion.punto_en_volumen(*destino):
                self.avisar(f"Punto fuera del volumen de trabajo: {destino}")
                return
            self.ocupado(True)
            try:
                # VA AL OBJETIVO Y REGRESA A HOME
                if self.trayectoria(destino):
                    self.trayectoria(HOME)
            finally:
                self.ocupado(False)

        def ver_volumen():
            self.simulacion.volumen_trabajo(progresbar=self.PROCESO)
            self.simulacion.plotter.reset_camera()

        self.CALCULAR.clicked.connect(aplicar_ci)
        self.VER_VOLUMEN.clicked.connect(ver_volumen)

        for slider, (lo, hi) in zip((self.SLIDER_1, self.SLIDER_2, self.SLIDER_3), LIMITES_SERVO):
            slider.setRange(lo, hi)
        self.sincronizar_sliders(*self.simulacion.angulos_servo(*HOME))
        for slider in (self.SLIDER_1, self.SLIDER_2, self.SLIDER_3):
            slider.valueChanged.connect(actualizar_robot)

    def GUARDAR_COORDENADAS(self):
        # EN EL SIGUIENTE FRAME, LOS PEDUNCULOS DETECTADOS REEMPLAZAN LA LISTA DE COSECHA
        self.CONTADOR = 1
        self.COSECHAR.setEnabled(True)

    def CARGAR_IMAGEN(self):
        """Procesa una foto del disco: sirve para probar la vision sin la camara del celular."""
        ruta, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Imagen de aguacates", "",
                                                        "Imágenes (*.png *.jpg *.jpeg *.bmp)")
        if not ruta:
            return
        imagen = cv2.imread(ruta)
        if imagen is None:
            self.avisar(f"No se pudo leer la imagen: {ruta}")
            return
        self.imagen_fija = imagen
        self.timer.start(30)  # SE REPROCESA EN CADA TICK PARA QUE LOS SLIDERS DE COLOR RESPONDAN

    def ACTUALIZAR_FRAME(self):
        if self.imagen_fija is not None:
            frame = self.imagen_fija
        else:
            ret, frame = self.video.read()
            if not ret:
                return
        self.procesar_imagen(frame)

    def procesar_imagen(self, frame):
        c1 = self.SLIDER_color_1.value()
        c2 = self.SLIDER_color_2.value()
        self.LABEL1.setText(str(c1))
        self.LABEL2.setText(str(c2))

        resultado, detecciones, _, _ = detectar_aguacates(frame, c1, c2)

        # LA CAMARA VA EN EL EFECTOR: SU POSE ES LA DEL BRAZO EN ESTE MOMENTO
        alto, ancho = frame.shape[:2]
        T_base_efector = self.simulacion.matrices(*servo_a_dh(*self.servos))[-1]
        pedunculos = []
        for det in detecciones:
            p = pedunculo_en_base(det, ancho, alto, T_base_efector)
            pedunculos.append(p)
            alcanzable = self.simulacion.punto_en_volumen(*p)
            u, v = (int(c) for c in det["pedunculo_px"])
            cv2.putText(resultado, f"({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}) mm", (u, v),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255) if alcanzable else (0, 0, 255), 2)

        if self.CONTADOR == 1:
            self.COORDENADAS_X = [float(p[0]) for p in pedunculos]
            self.COORDENADAS_Y = [float(p[1]) for p in pedunculos]
            self.COORDENADAS_Z = [float(p[2]) for p in pedunculos]
            self.CONTADOR = 0
            self.avisar(f"Guardados {len(pedunculos)} pedúnculos para cosechar.")

        self.ORIGINAL.setPixmap(DE_CV_A_QT(frame))
        self.RESULTADOS.setPixmap(DE_CV_A_QT(resultado))
        self.L_CONTADOR.setText(str(len(detecciones)))

    def closeEvent(self, event):
        
        self.timer.stop()
        self.simulacion.plotter.close()
        if self.video is not None:
            self.video.release()
        self.enlace.close()
        cv2.destroyAllWindows()
        event.accept()


def preparar_opengl():
    """En macOS VTK necesita un contexto OpenGL 3.2+ ("core profile"); sin esto los shaders
    fallan (version '150' is not supported) y no se dibujan las articulaciones ni el volumen.
    Debe llamarse ANTES de crear QApplication."""
    formato = QSurfaceFormat()
    formato.setRenderableType(QSurfaceFormat.OpenGL)
    formato.setVersion(3, 2)
    formato.setProfile(QSurfaceFormat.CoreProfile)
    formato.setDepthBufferSize(24)
    formato.setStencilBufferSize(8)
    formato.setSamples(4)
    QSurfaceFormat.setDefaultFormat(formato)


def _mostrar_error(tipo, valor, tb):
    # SIN ESTO, UNA EXCEPCION DENTRO DE UN BOTON CIERRA TODA LA APP (PyQt5 >= 5.5)
    traceback.print_exception(tipo, valor, tb)
    for ventana in QApplication.topLevelWidgets():
        if isinstance(ventana, VentanaPrincipal):
            ventana.statusBar().showMessage(f"Error: {valor}")


if __name__ == "__main__":
    sys.excepthook = _mostrar_error
    preparar_opengl()
    app = QApplication(sys.argv)
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(app.exec_())
