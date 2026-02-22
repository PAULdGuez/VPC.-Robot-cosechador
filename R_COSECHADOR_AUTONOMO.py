import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QProgressBar
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import uic, QtWidgets, QtCore
from R_C_ESP32_CLASE import cinematica
from pyvistaqt import QtInteractor
from PyQt5 import uic, QtWidgets, QtTest
import time

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
    # PARA LA SIMULACION, GUARDAMOS LAS COORDENADAAS DE PEDUNUCULOS DETECTADOS
    # DADO QUE PARA LA SIMULACION NECESITAMOS X,Y,Z, CREAMOS TRES ARRAYS PARA SU FACILIDAD
    COORDENADAS_X = [5 ,0,430,-370,-723,723,-1656,1656,0]
    COORDENADAS_Y = [-4,0,0,0,0,0,0,0,0]
    COORDENADAS_Z = [1057,1065,1718,1368,1552,1552,630,630,1065]
    CONTADOR = 0

    def __init__(self):
        super().__init__()
        uic.loadUi("COSECHADOR_AGUACATE.ui", self)
        self.simulacion = cinematica()
        # USAMOS UN INTERACTOR PARA PODER SUSTITUIRLO 
        self.simulacion.plotter = QtInteractor(self)
        #self.simulacion.plotter.camera_position = [(500, 500, 500), (0, 0, 0), (0, 0, 1)]
        self.dibujo = QtWidgets.QVBoxLayout(self.PANTALLA)
        self.dibujo.setContentsMargins(0,0,0,0)
        self.dibujo.addWidget(self.simulacion.plotter.interactor)
        t1, t2, t3 = self.simulacion.CI(0, 0, 1065)
        #self.simulacion.COSECHADOR(t1,100,0,90,   t2,0,100,0,   t3,0,100,0)
        self.simulacion.COSECHADOR(t1,350,0,90,     t2,0,1000,0,      t3,0,700,0)
        #self.simulacion.plotter.show(auto_close=False, interactive_update=True)
        self.configurar_sliders()

        self.PROCESO.setMinimum(0)
        self.PROCESO.setMaximum(46656)
        self.PROCESO.setValue(0)
        self.SLIDER_color_1.setRange(0, 255)
        self.SLIDER_color_2.setRange(0, 255)
        self.SLIDER_color_1.setValue(30)
        self.SLIDER_color_2.setValue(170)
        self.COSECHAR.setEnabled(True)
        self.GUARDAR.clicked.connect(self.GUARDAR_COORDENADAS)
        self.COSECHAR.clicked.connect(self.COSECHAR_AGUACATES)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.ACTUALIZAR_FRAME)
        self.timer.start(30)

        #self.video = cv2.VideoCapture("http://10.11.200.245:8080/video")
        self.video = cv2.VideoCapture("http://192.168.100.63:8080/video")

    def COSECHAR_AGUACATES(self):
        print(self.COORDENADAS_X)
        print(self.COORDENADAS_Y)
        print(self.COORDENADAS_Z)
        x_h = 0
        y_h = 0
        z_h = 1065

        t1, t2, t3 = self.simulacion.CI(0, 0, 1065)
        self.simulacion.COSECHADOR(t1,350,0,90,     t2,0,1000,0,      t3,0,700,0)

        for x_f, y_f, z_f in zip(self.COORDENADAS_X, self.COORDENADAS_Y, self.COORDENADAS_Z):

            if self.simulacion.punto_en_volumen(x_f, y_f, z_f):

                x_val = np.linspace(x_h, x_f, 20)
                y_val = np.linspace(y_h, y_f, 20)
                z_val = np.linspace(z_h, z_f, 20)

                for x, y, z in zip(x_val, y_val, z_val):
                    t1, t2, t3 = self.simulacion.CI(x, y, z)
                    self.simulacion.COSECHADOR(t1,350,0,90,     t2,0,1000,0,      t3,0,700,0)

                    self.simulacion.plotter.renderer.SetInteractive(True)
                    self.simulacion.plotter.update()
                    self.simulacion.plotter.reset_camera()
                    QtTest.QTest.qWait(50)
                
            else:
                print("Punto fuera del volumen de trabajo...")
            
            x_h = x_f
            y_h = y_f
            z_h = z_f
    

    def configurar_sliders(self):
        def actualizar_robot():
            # t_1 = self.SLIDER_1.value()
            # t_2 = self.SLIDER_2.value()
            # t_3 = self.SLIDER_3.value()
            t1 = self.SLIDER_1.value()
            t2 = 180-self.SLIDER_2.value()
            t3 = self.SLIDER_3.value()+180

            
            #self.simulacion.plotter.clear() 
            self.simulacion.plotter.renderer.SetInteractive(False)
            #                  |    1    |            |    2    |        |   3    |      |     4     |      |    5    |      |    6     |
            #t1, t2, t3 = self.simulacion.CI(t_1, t_2, t_3)
            #print(str(int(t1))+","+str(int(t2)))
            #self.enviar_angulos_serial(str(int(t1)) + "," + str(int(180-t2)) +","+ str(int(t3-180)))
            self.VS_1.setText(str(t1))
            self.VS_2.setText(str(-(t2-180)))
            self.VS_3.setText(str((t3)))

            self.simulacion.COSECHADOR(t1,350,0,90,   t2,0,1000,0,   t3,0,700,0)
            # self.simulacion.R7(t1,0,140,0,     t2,0,140,180,      0,d3,1,0,      90,20,0,0)
            #self.simulacion.R7(t1,352,0,0,     t2-90,0,70,-90,      t3,0,360,0,      t4,380,0,-90,     t5,0,1,-90,     t6-180,65,0,-90)
            self.simulacion.plotter.renderer.SetInteractive(True)
            self.simulacion.plotter.update()
            #self.simulacion.plotter.enable_render()
            #self.simulacion.plotter.write_frame() #---------------- PARA PODER HACER EL GIF
            #self.simulacion.plotter.camera.zoom(0.85)
            self.simulacion.plotter.reset_camera()
            #self.simulacion.plotter.render() 
            #time.sleep(0.02) #-------------------------------------- PARA EL GIF
        
        def aplicar_ci():
            # Abrir el GIF al inicio de la función
            self.simulacion.plotter.open_gif("SIMULACION.gif")
            
            try:
                x_f = int(self.TE_X.toPlainText())
                y_f = int(self.TE_Y.toPlainText())
                z_f = int(self.TE_Z.toPlainText())
                
                if self.simulacion.punto_en_volumen(x_f, y_f, z_f):
                    x_h, y_h, z_h = 0, 0, 1065  # Posición home
                    
                    # Configurar posición inicial
                    t1, t2, t3 = self.simulacion.CI(x_h, y_h, z_h)
                    self.simulacion.COSECHADOR(t1,350,0,90,     t2,0,1000,0,      t3,0,700,0)
                    #self.simulacion.plotter.write_frame()  # Capturar frame inicial
                    
                    # Movimiento hacia el objetivo
                    for x, y, z in zip(np.linspace(x_h, x_f, 20),
                                    np.linspace(y_h, y_f, 20),
                                    np.linspace(z_h, z_f, 20)):
                        t1, t2, t3 = self.simulacion.CI(x, y, z)
                        self.simulacion.COSECHADOR(t1,350,0,90,     t2,0,1000,0,      t3,0,700,0)
                        
                        # Actualizar visualización y capturar frame
                        self.simulacion.plotter.update()
                        #self.simulacion.plotter.write_frame()
                        QtTest.QTest.qWait(50)
                    
                    # Movimiento de regreso a home
                    for x, y, z in zip(np.linspace(x_f, x_h, 20),
                                    np.linspace(y_f, y_h, 20),
                                    np.linspace(z_f, z_h, 20)):
                        t1, t2, t3 = self.simulacion.CI(x, y, z)
                        self.simulacion.COSECHADOR(t1,350,0,90,     t2,0,1000,0,      t3,0,700,0)
                        
                        # Actualizar visualización y capturar frame
                        self.simulacion.plotter.update()
                        #self.simulacion.plotter.write_frame()
                        QtTest.QTest.qWait(50)
                else:
                    print("Punto fuera del volumen de trabajo...")
                    
            finally:
                print("finalizado")
                # Cerrar el GIF al final, asegurándose que siempre se ejecute
                #self.simulacion.plotter.close_gif()
                #print("GIF guardado como 'SIMULACION.gif'")

        def ver_volumen():
            self.simulacion.volumen_trabajo(progresbar = self.PROCESO)
            

        self.CALCULAR.clicked.connect(aplicar_ci)
        self.VER_VOLUMEN.clicked.connect(ver_volumen)

        self.SLIDER_1.setRange(0, 180)
        self.SLIDER_1.setValue(0)
        self.SLIDER_1.valueChanged.connect(actualizar_robot)

        self.SLIDER_2.setRange(20, 180)
        self.SLIDER_2.setValue(45)
        self.SLIDER_2.valueChanged.connect(actualizar_robot)

        self.SLIDER_3.setRange(0, 180)
        self.SLIDER_3.setValue(45)
        self.SLIDER_3.valueChanged.connect(actualizar_robot)

        
        #self.SLIDER_4.valueChanged.connect(actualizar_robot)
    
    def GUARDAR_COORDENADAS(self):
        self.CONTADOR = 1
        QtTest.QTest.qWait(50)
        self.COSECHAR.setEnabled(True)

    def COSECHAR_FRUTAS(self):
        print(self.COORDENADAS_X)
        print(self.COORDENADAS_Y)
        print(self.COORDENADAS_Z)
        for x_f, y_f, z_f in zip(self.COORDENADAS_X, self.COORDENADAS_Y, self.COORDENADAS_Z):
            if self.simulacion.punto_en_volumen(x_f, y_f, z_f):
                x_h = 0
                y_h = 0
                z_h = 1065

                t1, t2, t3 = self.simulacion.CI(0, 0, 1065)
                self.simulacion.COSECHADOR(t1,350,0,90,     t2,0,1000,0,      t3,0,700,0)
                x_val = np.linspace(x_h, x_f, 20)
                y_val = np.linspace(y_h, y_f, 20)
                z_val = np.linspace(z_h, z_f, 20)

                for x, y, z in zip(x_val, y_val, z_val):
                    t1, t2, t3 = self.simulacion.CI(x, y, z)
                    self.simulacion.COSECHADOR(t1,350,0,90,     t2,0,1000,0,      t3,0,700,0)

                    self.simulacion.plotter.renderer.SetInteractive(True)
                    self.simulacion.plotter.update()
                    self.simulacion.plotter.reset_camera()
                    QtTest.QTest.qWait(50)

                x_val = np.linspace(x_f, x_h, 20)
                y_val = np.linspace(y_f, y_h, 20)
                z_val = np.linspace(z_f, z_h, 20)

                for x, y, z in zip(x_val, y_val, z_val):
                    t1, t2, t3 = self.simulacion.CI(x, y, z)
                    self.simulacion.COSECHADOR(t1,350,0,90,     t2,0,1000,0,      t3,0,700,0)

                    self.simulacion.plotter.renderer.SetInteractive(True)
                    self.simulacion.plotter.update()
                    self.simulacion.plotter.reset_camera()
                    QtTest.QTest.qWait(50)
                
            else:
                print("Punto fuera del volumen de trabajo...")
        self.COORDENADAS_X.clear()
        self.COORDENADAS_Y.clear()
        self.COORDENADAS_Z.clear()
    
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
        
        self.simulacion.plotter.close()
        self.video.release()
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

        return i, resultado, mascara_verde, mascara_cafe

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(app.exec_())
