import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QProgressBar
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import uic, QtWidgets, QtCore

from PyQt5 import uic, QtWidgets, QtTest

def DE_MASCARA_FLORAL_A_QT(mascara_vainilla):
    """Convierte una máscara binaria de detección floral a formato QPixmap"""
    bgr = cv2.cvtColor(mascara_vainilla, cv2.COLOR_GRAY2BGR)
    return DE_IMAGEN_VAINILLA_A_QT(bgr)

def DE_IMAGEN_VAINILLA_A_QT(imagen_vainilla):
    """Convierte una imagen OpenCV del cultivo de vainilla a formato QPixmap"""
    rgb_imagen_vainilla = cv2.cvtColor(imagen_vainilla, cv2.COLOR_BGR2RGB)
    alto_vainilla, ancho_vainilla, canales_vainilla = rgb_imagen_vainilla.shape
    bytes_por_linea = canales_vainilla * ancho_vainilla
    imagen_qt_vainilla = QImage(rgb_imagen_vainilla.data, ancho_vainilla, alto_vainilla, bytes_por_linea, QImage.Format_RGB888)
    return QPixmap.fromImage(imagen_qt_vainilla)

class VentanaDetectorVainilla(QtWidgets.QMainWindow):
    CONTADOR_FLORES = 0

    def __init__(self):
        super().__init__()

        uic.loadUi("DETECTOR_DE_FLOR_VAINILLA.ui", self)

        #self.PROCESO.setMinimum(0)
        self.SLIDER_color_1.setRange(0, 255)
        self.SLIDER_color_2.setRange(0, 255)
        self.SLIDER_color_1.setValue(20)
        self.SLIDER_color_2.setValue(100)

        self.temporizador_vainilla = QtCore.QTimer()
        self.temporizador_vainilla.timeout.connect(self.ACTUALIZAR_CAPTURA_VAINILLA)
        self.temporizador_vainilla.start(30)

        #self.camara_vainilla = cv2.VideoCapture("http://10.11.200.245:8080/video")
        #self.camara_vainilla = cv2.VideoCapture("http://192.168.100.11:8080/video") 
        self.camara_vainilla = cv2.VideoCapture(2)

    def ACTUALIZAR_CAPTURA_VAINILLA(self):
        """Captura un nuevo frame de la cámara y procesa la detección de flores de vainilla"""
        ret_vainilla, frame_vainilla = self.camara_vainilla.read()
        if not ret_vainilla:
            return

        tono_flor = self.SLIDER_color_1.value()
        saturacion_flor = self.SLIDER_color_2.value()
        self.LABEL1.setText(str(tono_flor))
        self.LABEL2.setText(str(saturacion_flor))

        total_flores, resultado_vainilla, mascara_floral = self.DETECTAR_FLOR_VAINILLA(frame_vainilla, tono_flor, saturacion_flor)

        self.ORIGINAL.setPixmap(DE_IMAGEN_VAINILLA_A_QT(frame_vainilla))
        self.RESULTADOS.setPixmap(DE_IMAGEN_VAINILLA_A_QT(resultado_vainilla))
        self.L_CONTADOR.setText(str(total_flores))
        self.MASCARA.setPixmap(DE_IMAGEN_VAINILLA_A_QT(mascara_floral))

    def closeEvent(self, event):
        self.camara_vainilla.release()
        cv2.destroyAllWindows()
        event.accept()

    #NOTE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCION PRINCIPAL DE DETECCION DE FLORES DE VAINILLA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def DETECTAR_FLOR_VAINILLA(self, imagen_cultivo, H=30, S=90):
        """
        Detecta flores abiertas de vainilla en la imagen del cultivo.
        Divide la imagen en una cuadrícula 3x3 para mapear la ubicación de cada flor.
        """
        alto_cultivo, ancho_cultivo, _ = imagen_cultivo.shape
        linea_vertical_1_x = int(ancho_cultivo / 3)
        linea_vertical_2_x = int((ancho_cultivo / 3) * 2)
        linea_horizontal_1_y = int(alto_cultivo / 3)
        linea_horizontal_2_y = int((alto_cultivo / 3) * 2)
        color_guia = (255, 0, 0)  # Color de las líneas guía
        grosor_guia = 2
    
        # CONVERTIMOS AL ESPACIO DE COLOR HSV PARA SEGMENTAR LAS FLORES DE VAINILLA
        hsv_vainilla = cv2.cvtColor(imagen_cultivo, cv2.COLOR_BGR2HSV)
        # APLICAMOS UN FILTRO GAUSSIANO PARA ELIMINAR RUIDO EN LA IMAGEN DEL CULTIVO
        imagen_suavizada_vainilla = cv2.GaussianBlur(imagen_cultivo, (5, 5), 0)
        hsv_vainilla = cv2.cvtColor(imagen_suavizada_vainilla, cv2.COLOR_BGR2HSV)

        # CREACION DE LA MASCARA DE COLOR PARA DETECTAR LOS PETALOS DE LAS FLORES DE VAINILLA
        # Las flores de vainilla tienen tonos que van del blanco-verdoso al amarillo pálido
        #tono_bajo_flor = np.array([H, S, 70])
        #tono_alto_flor = np.array([35, 255, 255])
        #mascara_floral = cv2.inRange(hsv_vainilla, tono_bajo_flor, tono_alto_flor)

        rojo_bajo_petalo = np.array([0, 100, 20])
        rojo_alto_petalo = np.array([10, 255, 255])
        mascara_floral = cv2.inRange(hsv_vainilla, rojo_bajo_petalo, rojo_alto_petalo)

        # MEDIANTE MORPHOLOGYEX, HACEMOS MAS SOLIDAS LAS FLORES DETECTADAS AL RELLENAR CONTORNOS
        # TAMBIEN ELIMINAMOS PUNTOS AISLADOS QUE NO CORRESPONDEN A FLORES REALES
        kernel_vainilla = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mascara_floral = cv2.morphologyEx(mascara_floral, cv2.MORPH_OPEN, kernel_vainilla)
        mascara_floral = cv2.morphologyEx(mascara_floral, cv2.MORPH_CLOSE, kernel_vainilla)

        # DETECTAMOS LOS CONTORNOS DE LAS FLORES DE VAINILLA
        contornos_florales, _ = cv2.findContours(mascara_floral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # CREAMOS UNA IMAGEN DONDE SE RESALTAN A COLOR LAS FLORES DE VAINILLA DETECTADAS
        imagen_gris_cultivo = cv2.cvtColor(imagen_cultivo, cv2.COLOR_BGR2GRAY)
        imagen_gris_cultivo = cv2.cvtColor(imagen_gris_cultivo, cv2.COLOR_GRAY2BGR)
        resultado_vainilla = np.where(mascara_floral[:, :, np.newaxis], imagen_cultivo, imagen_gris_cultivo)

        # Dibujar las líneas verticales de la cuadrícula de monitoreo
        cv2.line(resultado_vainilla, (linea_vertical_1_x, 0), (linea_vertical_1_x, alto_cultivo), color_guia, grosor_guia)
        cv2.line(resultado_vainilla, (linea_vertical_2_x, 0), (linea_vertical_2_x, alto_cultivo), color_guia, grosor_guia)
        
        # Dibujar las líneas horizontales de la cuadrícula de monitoreo
        cv2.line(resultado_vainilla, (0, linea_horizontal_1_y), (ancho_cultivo, linea_horizontal_1_y), color_guia, grosor_guia)
        cv2.line(resultado_vainilla, (0, linea_horizontal_2_y), (ancho_cultivo, linea_horizontal_2_y), color_guia, grosor_guia)

        # CLASIFICAMOS LOS CONTORNOS EN FLORES COMPLETAS Y BROTES PEQUEÑOS
        # LAS FLORES COMPLETAS SE MARCAN CON RECTANGULO VERDE
        # LOS BROTES AISLADOS (FUERA DE UNA FLOR) SE MARCAN CON RECTANGULO ROJO
        flores_detectadas = []
        brotes_pequenos = []

        indice_flor = 0
        indice_brote = 0

        for contorno_floral in contornos_florales:
            area_floral = cv2.contourArea(contorno_floral)
            if area_floral > 220:
                flores_detectadas.append(contorno_floral)
            else:
                brotes_pequenos.append(contorno_floral)

        FLORACION = np.zeros(9, dtype=int)
        for flor in flores_detectadas:
            indice_flor += 1
            x_flor, y_flor, ancho_flor, alto_flor = cv2.boundingRect(flor)
            cv2.rectangle(resultado_vainilla, (x_flor, y_flor), (x_flor + ancho_flor, y_flor + alto_flor), (0, 255, 0), 2)
            
            # ETIQUETA DE IDENTIFICACION DE LA FLOR
            etiqueta_flor = str(indice_flor)
            fuente_vainilla = cv2.FONT_ITALIC
            escala_fuente = 0.2
            grosor_fuente = 2
            color_texto_flor = (255, 255, 255)  # Color del texto (blanco)
            color_fondo_etiqueta = (0, 0, 0)  # Color de fondo (negro)

            (ancho_texto, alto_texto), linea_base = cv2.getTextSize(etiqueta_flor, fuente_vainilla, escala_fuente, grosor_fuente)
            cv2.rectangle(resultado_vainilla, (x_flor, y_flor - alto_texto - linea_base - 5), (x_flor + ancho_texto, y_flor - 5), color_fondo_etiqueta, -1)
            cv2.putText(resultado_vainilla, etiqueta_flor, (x_flor, y_flor - linea_base - 5), fuente_vainilla, escala_fuente, color_texto_flor, grosor_fuente, cv2.LINE_AA)

            momentos_flor = cv2.moments(flor)
            if momentos_flor['m00'] != 0:
                centro_x_flor = int(momentos_flor['m10'] / momentos_flor['m00'])
                centro_y_flor = int(momentos_flor['m01'] / momentos_flor['m00'])
                if centro_x_flor < linea_vertical_1_x and centro_y_flor < linea_horizontal_1_y:
                    print("VAINILLA ZONA 1,1 - FLOR ABIERTA")
                    FLORACION[0] = 1

                elif centro_x_flor < linea_vertical_1_x and centro_y_flor > linea_horizontal_1_y and centro_y_flor < linea_horizontal_2_y:
                    print("VAINILLA ZONA 2,1 - FLOR ABIERTA")
                    FLORACION[1] = 1
                elif centro_x_flor < linea_vertical_1_x and centro_y_flor > linea_horizontal_2_y:
                    print("VAINILLA ZONA 3,1 - FLOR ABIERTA")
                    FLORACION[2] = 1
                
                if centro_x_flor > linea_vertical_1_x and centro_x_flor < linea_vertical_2_x and centro_y_flor < linea_horizontal_1_y:
                    print("VAINILLA ZONA 1,2 - FLOR ABIERTA")
                    FLORACION[3] = 1
                elif centro_x_flor > linea_vertical_1_x and centro_x_flor < linea_vertical_2_x and centro_y_flor > linea_horizontal_1_y and centro_y_flor < linea_horizontal_2_y:
                    print("VAINILLA ZONA 2,2 - FLOR ABIERTA")
                    FLORACION[4] = 1
                elif centro_x_flor > linea_vertical_1_x and centro_x_flor < linea_vertical_2_x and centro_y_flor > linea_horizontal_2_y:
                    print("VAINILLA ZONA 3,2 - FLOR ABIERTA")
                    FLORACION[5] = 1
                if centro_x_flor > linea_vertical_2_x and centro_y_flor < linea_horizontal_1_y:
                    print("VAINILLA ZONA 1,3 - FLOR ABIERTA")
                    FLORACION[6] = 1
                elif centro_x_flor > linea_vertical_2_x and centro_y_flor > linea_horizontal_1_y and centro_y_flor < linea_horizontal_2_y:
                    print("VAINILLA ZONA 2,3 - FLOR ABIERTA")
                    FLORACION[7] = 1
                elif centro_x_flor > linea_vertical_2_x and centro_y_flor > linea_horizontal_2_y:
                    print("VAINILLA ZONA 3,3 - FLOR ABIERTA")
                    FLORACION[8] = 1
        indicadores_zona = [
            self.L_11, self.L_21, self.L_31,
            self.L_12, self.L_22, self.L_32,
            self.L_13, self.L_23, self.L_33
        ]
        estilo_base_vainilla = """
            border: 2px solid black;
            font-size: 16px; 
            border-radius: 10px;
            padding: 4px;
        """
        for zona, hay_flor in enumerate(FLORACION):
            if hay_flor == 1:
                indicadores_zona[zona].setStyleSheet(estilo_base_vainilla + "background-color: #c4ff0d; color: white;") 
            else:
                indicadores_zona[zona].setStyleSheet(estilo_base_vainilla + "background-color: red; color: white;")


        for brote in brotes_pequenos:
            brote_dentro_de_flor = False
            
            for flor in flores_detectadas:
                # Verificamos si el brote pequeño está dentro de una flor detectada
                # Usamos el centroide del brote para la prueba de inclusión
                momentos_brote = cv2.moments(brote)
                if momentos_brote['m00'] != 0:
                    centro_x_brote = int(momentos_brote['m10'] / momentos_brote['m00'])
                    centro_y_brote = int(momentos_brote['m01'] / momentos_brote['m00'])
            
                    if cv2.pointPolygonTest(flor, (centro_x_brote, centro_y_brote), False) >= 0:
                        brote_dentro_de_flor = True
                        break  # Salimos del bucle, el brote pertenece a esta flor
            
            if not brote_dentro_de_flor:
                indice_brote += 1
                x_brote, y_brote, ancho_brote, alto_brote = cv2.boundingRect(brote)
                cv2.rectangle(resultado_vainilla, (x_brote, y_brote), (x_brote + ancho_brote, y_brote + alto_brote), (0, 0, 255), 2)
            
        return indice_flor, resultado_vainilla, mascara_floral

if __name__ == "__main__":
    app_vainilla = QApplication(sys.argv)
    ventana_vainilla = VentanaDetectorVainilla()
    ventana_vainilla.show()
    sys.exit(app_vainilla.exec_())
