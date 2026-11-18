"""Vision del cosechador: detecta aguacates y convierte cada pedunculo a coordenadas del robot (mm).

Modelo de camara "pinhole" montada en el EFECTOR (eye-in-hand):

    punto_base = T_base_efector @ T_EFECTOR_CAMARA @ punto_camara

- T_base_efector sale de la cinematica directa (la pose del brazo al tomar la foto).
- T_EFECTOR_CAMARA dice como esta atornillada la camara a la punta. Para una camara FIJA
  (en la base o en un tripie) basta con pasar su pose fija en lugar de T_base_efector @ T_EFECTOR_CAMARA.
- La profundidad se estima por el tamaño aparente del aguacate: Z = fx * DIAMETRO / ancho_en_pixeles.

Todo en milimetros. Ejes de la camara (convencion OpenCV): x a la derecha, y hacia abajo, z al frente.
"""
import cv2
import numpy as np

DIAMETRO_AGUACATE_MM = 80.0   # ANCHO TIPICO DE UN HASS; DE AQUI SALE LA PROFUNDIDAD

# PARAMETROS INTRINSECOS DE REFERENCIA (CAMARA DEL CELULAR A 1920x1080).
# SE ESCALAN A LA RESOLUCION REAL DEL FRAME. PARA PRECISION REAL: CALIBRAR CON UN TABLERO DE AJEDREZ.
RESOLUCION_REFERENCIA = (1920, 1080)
FX_REFERENCIA = 1200.0
DISTORSION = np.zeros((1, 5), dtype=np.float64)  # [k1, k2, p1, p2, k3]

# CAMARA EN LA PUNTA DEL EFECTOR, MIRANDO A LO LARGO DEL ANTEBRAZO.
# COLUMNAS = EJES DE LA CAMARA EXPRESADOS EN EL MARCO DEL EFECTOR (DH, ESLABON 3):
#   z_camara (al frente)  = x3 (a lo largo del antebrazo)
#   x_camara (derecha)    = z3
#   y_camara (abajo)      = -y3
T_EFECTOR_CAMARA = np.array([
    [0,  0, 1, 0],
    [0, -1, 0, 0],
    [1,  0, 0, 0],
    [0,  0, 0, 1],
], dtype=np.float64)


def matriz_camara(ancho, alto):
    """Matriz intrinseca K escalada a la resolucion del frame."""
    escala = ancho / RESOLUCION_REFERENCIA[0]
    f = FX_REFERENCIA * escala
    return np.array([[f, 0, ancho / 2], [0, f, alto / 2], [0, 0, 1]], dtype=np.float64)


def profundidad_por_tamano(ancho_px, fx, diametro_mm=DIAMETRO_AGUACATE_MM):
    """Distancia (mm) a un objeto de tamaño conocido por su ancho aparente en pixeles."""
    return fx * diametro_mm / ancho_px


def pixel_a_camara(u, v, profundidad, K, distorsion=DISTORSION):
    """Pixel (u, v) a esa profundidad -> punto 3D en el marco de la camara (mm)."""
    x, y = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float64), K, distorsion)[0, 0]
    return np.array([x * profundidad, y * profundidad, profundidad])


def camara_a_pixel(p_camara, K):
    """Proyeccion inversa (sin distorsion): punto en el marco de la camara -> pixel. Util para pruebas."""
    x, y, z = p_camara
    return K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2]


def transformar(T, p):
    return (T @ np.append(p, 1.0))[:3]


def detectar_aguacates(img, H=30, S=90):
    """Detecta aguacates por color.

    Devuelve (resultado, detecciones, mascara_verde, mascara_cafe). Cada deteccion trae:
    caja (x, y, w, h) del fruto y pedunculo_px (u, v): el punto de corte, arriba del fruto.
    """
    # APLICAMOS UN FILTRO GAUSSIANO PARA ELIMINAR RUIDO Y PASAMOS AL ESPACIO HSV
    img_suavizada = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(img_suavizada, cv2.COLOR_BGR2HSV)

    # MASCARA VERDE EN BASE A UN RANGO DE COLORES
    mascara_verde = cv2.inRange(hsv, np.array([H, S, 70]), np.array([85, 255, 255]))

    # MORPHOLOGYEX: RELLENA HUECOS DE LOS OBJETOS Y QUITA PUNTOS BLANCOS AISLADOS
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mascara_verde = cv2.morphologyEx(mascara_verde, cv2.MORPH_OPEN, kernel)
    mascara_verde = cv2.morphologyEx(mascara_verde, cv2.MORPH_CLOSE, kernel)
    contornos_verde, _ = cv2.findContours(mascara_verde, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # MASCARA PARA DEFECTOS (CAFE)
    mascara_cafe = cv2.inRange(hsv, np.array([5, 30, 20]), np.array([20, 255, 255]))
    contornos_cafe, _ = cv2.findContours(mascara_cafe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # IMAGEN EN GRISES CON LOS AGUACATES DETECTADOS A COLOR
    img_gris = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    resultado = np.where(mascara_verde[:, :, np.newaxis], img, img_gris)

    detecciones = []
    for cnt in contornos_verde:
        if cv2.contourArea(cnt) <= 10:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        numero = len(detecciones) + 1
        cv2.rectangle(resultado, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(resultado, str(numero), (x, y), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

        # RECTANGULO DEL PEDUNCULO: UN TERCIO DE LA ALTURA DEL FRUTO, JUSTO ENCIMA. SU CENTRO ES EL PUNTO DE CORTE
        h_azul = h // 3
        y_azul = max(y - h_azul, 0)
        cv2.rectangle(resultado, (x, y_azul), (x + w, y), (255, 0, 0), 2)
        detecciones.append({"caja": (x, y, w, h),
                            "pedunculo_px": (x + w / 2, y_azul + h_azul / 2)})

    # DEFECTOS: SOLO LOS QUE CAEN DENTRO DE UN AGUACATE (EVITA MARCAR RAMAS)
    for cnt in contornos_cafe:
        if cv2.contourArea(cnt) <= 10:
            continue
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
        if any(cv2.pointPolygonTest(cv_cnt, (cx, cy), False) >= 0 for cv_cnt in contornos_verde):
            cv2.drawContours(resultado, [cnt], -1, (255, 0, 0), 2)
            cv2.putText(resultado, "D", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return resultado, detecciones, mascara_verde, mascara_cafe


def pedunculo_en_base(deteccion, ancho_img, alto_img, T_base_efector):
    """Punto de corte de una deteccion -> coordenadas del robot (mm), con el brazo en T_base_efector."""
    K = matriz_camara(ancho_img, alto_img)
    _, _, w, _ = deteccion["caja"]
    # boundingRect MIDE 2r+1 PIXELES PARA UN OBJETO DE RADIO r: EL ANCHO REAL ES w - 1
    profundidad = profundidad_por_tamano(max(w - 1, 1), K[0, 0])
    p_camara = pixel_a_camara(*deteccion["pedunculo_px"], profundidad, K)
    return transformar(T_base_efector @ T_EFECTOR_CAMARA, p_camara)
