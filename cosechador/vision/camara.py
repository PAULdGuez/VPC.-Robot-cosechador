"""Modelo de cámara "pinhole": de un píxel a coordenadas del robot (mm).

    punto_base = pose_camara @ punto_camara

- Montaje "efector" (eye-in-hand): pose_camara = pose del extremo del último eslabón (pose_efector) @ matriz_montaje.
- Montaje "fija": pose_camara = matriz_montaje (la pose de la cámara respecto a la base).
- La profundidad se estima por el tamaño aparente del objeto: Z = fx * diametro / ancho_en_pixeles.

Ejes de la cámara (convención OpenCV): x a la derecha, y hacia abajo, z al frente.
"""
import cv2
import numpy as np

from cosechador.robot.cinematica import pose_efector

DISTORSION = np.zeros((1, 5), dtype=np.float64)  # [k1, k2, p1, p2, k3]; calibrar para precisión real


def matriz_camara(camara, ancho, alto):
    """Matriz intrínseca K escalada a la resolución del frame."""
    f = camara.fx_referencia * ancho / camara.resolucion_referencia[0]
    return np.array([[f, 0, ancho / 2], [0, f, alto / 2], [0, 0, 1]], dtype=np.float64)


def profundidad_por_tamano(ancho_px, fx, diametro_mm):
    """Distancia (mm) a un objeto de tamaño conocido por su ancho aparente en píxeles."""
    return fx * diametro_mm / ancho_px


def pixel_a_camara(u, v, profundidad, K, distorsion=DISTORSION):
    """Píxel (u, v) a esa profundidad -> punto 3D en el marco de la cámara (mm)."""
    x, y = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float64), K, distorsion)[0, 0]
    return np.array([x * profundidad, y * profundidad, profundidad])


def camara_a_pixel(p_camara, K):
    """Proyección (sin distorsión): punto en el marco de la cámara -> píxel. Útil para pruebas."""
    x, y, z = p_camara
    return K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2]


def transformar(T, p):
    return (T @ np.append(p, 1.0))[:3]


def pose_camara(robot, q):
    """Pose 4x4 de la cámara en el marco de la base, con el brazo en los ángulos DH q."""
    if robot.camara.montaje == "efector":
        # LA CÁMARA VA ATORNILLADA AL EXTREMO DEL ÚLTIMO ESLABÓN, NO A LA PUNTA DE LA HERRAMIENTA
        # (cinematica_directa): CAMBIAR EL LARGO DE LA HERRAMIENTA NO LA MUEVE.
        return pose_efector(robot, q) @ robot.camara.matriz_montaje
    return robot.camara.matriz_montaje


def pedunculo_en_base(deteccion, ancho_img, alto_img, robot, q):
    """Punto de corte de una detección -> coordenadas del robot (mm), con el brazo en q."""
    K = matriz_camara(robot.camara, ancho_img, alto_img)
    _, _, w, _ = deteccion["caja"]
    # boundingRect MIDE 2r+1 PIXELES PARA UN OBJETO DE RADIO r: EL ANCHO REAL ES w - 1
    profundidad = profundidad_por_tamano(max(w - 1, 1), K[0, 0], robot.camara.diametro_objeto_mm)
    p_camara = pixel_a_camara(*deteccion["pedunculo_px"], profundidad, K)
    return transformar(pose_camara(robot, q), p_camara)
