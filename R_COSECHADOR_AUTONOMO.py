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

# Matriz de rotación (3x3) y vector de traslación (3x1)
R_cam_to_robot = np.array([
    [1, 0, 0],     # Suponiendo que no hay rotación
    [0, 1, 0],
    [0, 0, 1]
])

t_cam_to_robot = np.array([[0.2],  # 20 cm adelante del efector
                           [0.0],  # alineado en Y
                           [0.1]]) # 10 cm arriba del efector

def transformar_a_robot(X, Y, Z):
    punto_cam = np.array([[X], [Y], [Z]])
    punto_robot = R_cam_to_robot @ punto_cam + t_cam_to_robot
    return punto_robot.flatten()

# === PARÁMETROS DE CALIBRACIÓN (REAL) ===
camera_matrix = np.array([
    [1200.0,    0.0, 960.0],  # fx, 0, cx
    [0.0,    1200.0, 540.0],  # 0, fy, cy
    [0.0,       0.0,   1.0]
], dtype=np.float32)

# Coeficientes de distorsión (asumiendo lente muy bien corregido)
# [k1, k2, p1, p2, k3]
dist_coeffs = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

rvec = np.zeros((3, 1), dtype=np.float32)
tvec = np.zeros((3, 1), dtype=np.float32)

def pixeles_a_mundo(punto_pixel, Z=1.0):
    puntos = np.array([[punto_pixel]], dtype=np.float32)
    puntos_normalizados = cv2.undistortPoints(puntos, camera_matrix, dist_coeffs)
    x, y = puntos_normalizados[0, 0]
    return x * Z, y * Z, Z

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
        # EN EL SIGUIENTE FRAME, DETECTAR_FRUTA AGREGA LOS PEDUNCULOS DETECTADOS A LA LISTA
        self.CONTADOR = 1
        self.COSECHAR.setEnabled(True)

    def ACTUALIZAR_FRAME(self):
        ret, frame = self.video.read()
        if not ret:
            return

        c1 = self.SLIDER_color_1.value()
        c2 = self.SLIDER_color_2.value()
        self.LABEL1.setText(str(c1))
        self.LABEL2.setText(str(c2))

        contador, resultado, mascara_verde, mascara_cafe = self.DETECTAR_FRUTA(frame, c1, c2)

        self.ORIGINAL.setPixmap(DE_CV_A_QT(frame))
        self.RESULTADOS.setPixmap(DE_CV_A_QT(resultado))
        self.L_CONTADOR.setText(str(contador))

    def closeEvent(self, event):
        
        self.timer.stop()
        self.simulacion.plotter.close()
        if self.video is not None:
            self.video.release()
        self.enlace.close()
        cv2.destroyAllWindows()
        event.accept()

    #NOTE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCION PRINCIPAL DE PROCESAMIENTO DE IMAGENES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def DETECTAR_FRUTA(self,img, H=30, S=90):
        # COMVERTMIOS AL EPACIO HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #APLICAMOS UN FILTRO GAUSSIANO PARA ELIMIAR RUIDOS
        img_suavizada = cv2.GaussianBlur(img, (5, 5), 0)
        hsv = cv2.cvtColor(img_suavizada, cv2.COLOR_BGR2HSV)

        # CREACION DE LA UNA MASCARA VERDE EN BASE A UN RANGO DE COLORES 
        verde_bajo = np.array([H, S, 70])
        verde_alto = np.array([85, 255, 255])
        mascara_verde = cv2.inRange(hsv, verde_bajo, verde_alto)

        # MEDIANTE MORPHOLOGYEX, HACEMOS MAS SOLIDOS LOS OBJETOS DETECTADOS AL RELLENAR CONTORNOS BLANCOS DETECTADOS
        # TAMBIEN ELIMINAMOS PUNTOS PLANCOS QUE ESTEN RODEADOS POR ENTORNO NEGRO
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mascara_verde = cv2.morphologyEx(mascara_verde, cv2.MORPH_OPEN, kernel)
        mascara_verde = cv2.morphologyEx(mascara_verde, cv2.MORPH_CLOSE, kernel)

        # DETECTAMOS LOS AGUACATES
        contornos_verde, _ = cv2.findContours(mascara_verde, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # MASCARA PARA DEFECTOS
        cafe_bajo = np.array([5, 30, 20])
        cafe_alto = np.array([20, 255, 255])
        mascara_cafe = cv2.inRange(hsv, cafe_bajo, cafe_alto) 
        # DETECTAMOS LOS CONTORNOS RESPECTO A DEFECTOS
        contornos_cafe, _ = cv2.findContours(mascara_cafe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # PARA MEJORES RESULTADOS, CREAMOS UNA IMAGEN DONDE SE RESALTEN A COLOR LOS AGUACATES DETECTADOS. 
        img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gris = cv2.cvtColor(img_gris, cv2.COLOR_GRAY2BGR)
        resultado = np.where(mascara_verde[:, :, np.newaxis], img, img_gris)

        # DIBUJAMOS RECTANGULOS PARA SEÑAALR LOS AGUACATES DETECTADOS ADEMAS DE DIBUAR UN RECTANGULO 
        # DONDE APROX.TE SE ENCUENTRA EL PEDUNCULO, A ESTE ULTIMO SE LE CALCULA EL CENTRO, EL CUAL ES EL PUNTO 
        # DE CORTE DEL PEDUNCULO
        i = 1
        for cnt in contornos_verde:
            area = cv2.contourArea(cnt)
            if area > 10:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(resultado, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(resultado, str(i), (x, y), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

                # RECTANGULO DEL PEDUNCULO 
                h_azul = h // 3
                y_azul = y - h_azul
                if y_azul < 0:
                    y_azul = 0
                cv2.rectangle(resultado, (x, y_azul), (x + w, y), (255, 0, 0), 2)

                cx = x + (w//2)
                cy = y_azul + (h_azul // 2)

                X, Y, Z = pixeles_a_mundo((cx, cy), Z=13)
                # FIXME por ahora tenemos una calibracion eMpirica, pero falta calibrar mejor la camara
                X = X/3 + 0.17
                if X > 0:
                    X = X 
                else:
                    X =  X +14
                Y = Y/10

                texto = f"({X:.2f}, {Y:.2f}, {Z:.2f})"
                cv2.putText(resultado, texto, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                #x_r, y_r, z_r = transformar_a_robot(X, Y, Z)

                # COORDENAS PARA EL ROBOT, OR AHORA COLOCALEMOS LA CAMARA COMO SI FUERA EL EFECTOR POR LO QUE LAS COORDENDAS DEL EFECTOR 
                # SON LAS DE LA CAMARA
                # ahora, la camaara me entrega valroes de x,y,z siendo x = x del robot, y = z del robot y z = y del robot. 
                # por lo tanto, el punto en donde se encuntra el objeto es:
                X_HOME = 56 
                Y_HOME = 84
                Z_HOME = 13
                x_final = Z + X_HOME # x_final REPRSENTA LA POSICON DEL X DEL OBJETO A CORTAR, ESTO ES LA SUMA DE LA DISTANCIA CALCULADA POR LA CAMARA Y LA POS X DEL EFECTOR
                y_final = X + Y_HOME 
                z_final = Y + Z_HOME

                if self.CONTADOR == 1:
                    self.COORDENADAS_X.append(x_final)
                    self.COORDENADAS_Y.append(-1*y_final)
                    self.COORDENADAS_Z.append(z_final)
                    
                
                # de esta manera conseguimos el punto a donde el robot se debe dirigir. 
                print(f"Enviar al robot: ({x_final:.2f}, {y_final:.2f}, {z_final:.2f})") 
                i += 1
            
        self.CONTADOR = 0
        # FOR PARA DIBUJAR LOS CONTORNOS DE DEFECTOS SOLO SI SE ENCUENTRAN DENTRO DE UN CONTORNO DE AGUACATE, PUESTO QUE
        # SOLO DETECTAMEROS DEFECTOS DE UN AGUACATE DETECTADO Y EVITAR DIBUAJAR DEFECTOS QUE NO VAN (POR EJEMPLO UNA RAMA)
        for cnt in contornos_cafe:
            area = cv2.contourArea(cnt)
            if area > 10:
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                dentro = any(cv2.pointPolygonTest(cv_cnt, (cx, cy), False) >= 0 for cv_cnt in contornos_verde)
                if dentro:
                    cv2.drawContours(resultado, [cnt], -1, (255, 0, 0), 2)
                    cv2.putText(resultado, "D", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return i - 1, resultado, mascara_verde, mascara_cafe

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
