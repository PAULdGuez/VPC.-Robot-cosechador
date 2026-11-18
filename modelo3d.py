"""Piezas 3D del robot para la simulacion.

Cada pieza se carga de modelos/<nombre>.stl si existe (exportada de Onshape/Fusion EN MILIMETROS);
si no, se usa una pieza provisional hecha con primitivas de PyVista. En cada cuadro la pieza
NO se regenera: solo se le cambia la matriz de posicion (actor.user_matrix).

CONVENCION PARA MODELAR CADA PIEZA (su propio origen y ejes):
- base:      fija. Origen en el centro del piso de la base, Z hacia arriba.
- columna:   gira con el servo 1. Origen en el piso, sobre el eje de giro; Z hacia arriba.
             Su parte superior llega a D1 (350 mm), donde esta el hombro.
- brazo:     gira con el servo 2. Origen en el eje del HOMBRO; el brazo se extiende sobre +X
             hasta A2 (1000 mm), donde esta el codo. El eje de giro del hombro es Z.
- antebrazo: gira con el servo 3. Origen en el eje del CODO; se extiende sobre +X hasta A3 (700 mm).
             El eje de giro del codo es Z.
- efector:   la herramienta de corte. Origen en la punta del antebrazo, apuntando a +X.
"""
import os

import numpy as np
import pyvista as pv

from cinematica import A2, A3, D1

CARPETA_MODELOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelos")

PIEZAS = ("base", "columna", "brazo", "antebrazo", "efector")

COLORES = {"base": "#4a5560", "columna": "#e07b22", "brazo": "#f09a3e",
           "antebrazo": "#f09a3e", "efector": "#c9d1d9"}


def _rot_z(grados):
    c, s = np.cos(np.radians(grados)), np.sin(np.radians(grados))
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def poses_piezas(robot, t1, t2, t3):
    """Matriz 4x4 de cada pieza (en el marco de la base) para unos angulos DH."""
    _, A0_1, A0_2, A0_3 = robot.matrices(t1, t2, t3)
    return {
        "base": np.eye(4),
        "columna": _rot_z(t1),
        "brazo": A0_1 @ _rot_z(t2),
        "antebrazo": A0_2 @ _rot_z(t3),
        "efector": A0_3,
    }


def _eslabon(largo, ancho, radio_eje):
    """Barra a lo largo de +X con un cubo de articulacion (eje Z) en cada extremo."""
    barra = pv.Box(bounds=(0, largo, -ancho / 2, ancho / 2, -ancho / 2, ancho / 2))
    inicio = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=radio_eje, height=ancho * 1.4)
    fin = pv.Cylinder(center=(largo, 0, 0), direction=(0, 0, 1), radius=radio_eje * 0.85, height=ancho * 1.3)
    return barra.triangulate() + inicio.triangulate() + fin.triangulate()


def pieza_provisional(nombre):
    """Geometria provisional de cada pieza, respetando la convencion de arriba."""
    if nombre == "base":
        return pv.Cylinder(center=(0, 0, 30), direction=(0, 0, 1), radius=180, height=60, resolution=48)
    if nombre == "columna":
        poste = pv.Cylinder(center=(0, 0, (60 + D1) / 2), direction=(0, 0, 1), radius=70,
                            height=D1 - 60, resolution=40)
        # CARCASA DEL HOMBRO: SU EJE ES EL EJE DE GIRO DEL SERVO 2 (HORIZONTAL)
        hombro = pv.Cylinder(center=(0, 0, D1), direction=(0, 1, 0), radius=85, height=150, resolution=40)
        return poste.triangulate() + hombro.triangulate()
    if nombre == "brazo":
        return _eslabon(A2, 80, 70)
    if nombre == "antebrazo":
        return _eslabon(A3, 60, 55)
    if nombre == "efector":
        # TIJERA PROVISIONAL: UN CONO QUE APUNTA AL FRENTE (+X)
        return pv.Cone(center=(60, 0, 0), direction=(1, 0, 0), height=120, radius=35, resolution=32)
    raise ValueError(f"Pieza desconocida: {nombre}")


def cargar_pieza(nombre, carpeta=CARPETA_MODELOS):
    """La pieza del STL si existe; si no, la provisional. Devuelve (malla, origen)."""
    ruta = os.path.join(carpeta, f"{nombre}.stl")
    if os.path.exists(ruta):
        return pv.read(ruta), ruta
    return pieza_provisional(nombre), "provisional"
