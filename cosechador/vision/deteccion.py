"""Deteccion de aguacates por color (OpenCV). Pura: imagen -> detecciones."""
import cv2
import numpy as np


def detectar_aguacates(img, vision, h_min=None, s_min=None):
    """Detecta aguacates por color con los rangos HSV de la configuracion (`robot.vision`).

    h_min y s_min (los sliders de la interfaz) reemplazan el tono y la saturacion minimos del verde.

    Devuelve (resultado, detecciones, mascara_verde, mascara_cafe). Cada deteccion trae:
    caja (x, y, w, h) del fruto y pedunculo_px (u, v): el punto de corte, arriba del fruto.
    """
    # APLICAMOS UN FILTRO GAUSSIANO PARA ELIMINAR RUIDO Y PASAMOS AL ESPACIO HSV
    img_suavizada = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(img_suavizada, cv2.COLOR_BGR2HSV)

    # MASCARA VERDE EN BASE A UN RANGO DE COLORES
    verde_min = list(vision.verde.minimo)
    if h_min is not None:
        verde_min[0] = h_min
    if s_min is not None:
        verde_min[1] = s_min
    mascara_verde = cv2.inRange(hsv, np.array(verde_min), np.array(vision.verde.maximo))

    # MORPHOLOGYEX: RELLENA HUECOS DE LOS OBJETOS Y QUITA PUNTOS BLANCOS AISLADOS
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mascara_verde = cv2.morphologyEx(mascara_verde, cv2.MORPH_OPEN, kernel)
    mascara_verde = cv2.morphologyEx(mascara_verde, cv2.MORPH_CLOSE, kernel)
    contornos_verde, _ = cv2.findContours(mascara_verde, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # MASCARA PARA DEFECTOS (CAFE)
    mascara_cafe = cv2.inRange(hsv, np.array(vision.cafe.minimo), np.array(vision.cafe.maximo))
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
