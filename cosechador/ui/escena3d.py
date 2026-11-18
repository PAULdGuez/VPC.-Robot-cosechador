"""Escena 3D del robot (PyVista). Funciona con cualquier robot cargado del YAML.

Cada pieza sale de modelos/<archivo> si existe (STL exportado EN MILÍMETROS); si no, se arma una
pieza provisional a partir de la tabla DH. En cada cuadro la geometría NO se regenera: solo se
cambia la matriz de posición de cada pieza (actor.user_matrix).

CONVENCIÓN DE CADA PIEZA (su propio origen y ejes; ver modelos/README.md):
- base:            fija; origen en el centro del piso, Z hacia arriba.
- articulación i:  gira con su servo. Origen en SU eje de giro (que es Z). La pieza sube d_i sobre Z
                   y luego avanza a_i sobre +X, hasta el eje de la siguiente articulación.
- efector:         origen en la punta del último eslabón, apuntando a +X.
"""
import os

import numpy as np
import pyvista as pv

from cosechador.robot.cinematica import matrices, rot_z

ALCANCE_REFERENCIA = 2050.0  # LAS PIEZAS PROVISIONALES SE DISEÑARON PARA EL COSECHADOR; SE ESCALAN


def piezas_del_robot(robot):
    """[(nombre, Pieza), ...] en orden: base, una por articulación, efector."""
    return ([("base", robot.base)]
            + [(j.nombre, j.pieza) for j in robot.articulaciones]
            + [("efector", robot.efector)])


def poses_piezas(robot, q):
    """Matriz 4x4 de cada pieza (en el marco de la base) para unos ángulos DH, en el mismo orden."""
    acumuladas = matrices(robot, q)
    return ([np.eye(4)]
            + [acumuladas[i] @ rot_z(theta) for i, theta in enumerate(q)]
            + [acumuladas[-1]])


def _cilindro(centro, direccion, radio, alto):
    return pv.Cylinder(center=centro, direction=direccion, radius=radio, height=alto, resolution=40).triangulate()


def pieza_provisional(robot, indice):
    """Geometría provisional de la pieza `indice` (0 = base, 1..n = articulaciones, n+1 = efector)."""
    escala = robot.alcance / ALCANCE_REFERENCIA
    if indice == 0:
        return _cilindro((0, 0, 30 * escala), (0, 0, 1), 180 * escala, 60 * escala)
    if indice == robot.gdl + 1:
        # TIJERA PROVISIONAL: UN CONO QUE APUNTA AL FRENTE (+X)
        return pv.Cone(center=(60 * escala, 0, 0), direction=(1, 0, 0), height=120 * escala,
                       radius=35 * escala, resolution=32).triangulate()

    j = robot.articulaciones[indice - 1]
    grosor = 80 * escala * 0.9 ** max(indice - 2, 0)
    partes = [_cilindro((0, 0, 0), (0, 0, 1), grosor * 0.875, grosor * 1.4)]  # CUBO DE SU PROPIO EJE
    if abs(j.d) > 1e-9:  # TRAMO VERTICAL (SOBRE Z)
        partes.append(_cilindro((0, 0, j.d / 2), (0, 0, 1), grosor * 0.875, abs(j.d)))
    if abs(j.a) > 1e-9:  # TRAMO HORIZONTAL (SOBRE +X)
        partes.append(pv.Box(bounds=(min(0, j.a), max(0, j.a), -grosor / 2, grosor / 2,
                                     j.d - grosor / 2, j.d + grosor / 2)).triangulate())
    # CUBO DEL EJE DE LA SIGUIENTE ARTICULACION: Rx(alpha) APLICADO A Z
    eje_siguiente = (0, -np.sin(np.radians(j.alpha)), np.cos(np.radians(j.alpha)))
    partes.append(_cilindro((j.a, 0, j.d), eje_siguiente, grosor * 1.05, grosor * 1.8))
    malla = partes[0]
    for parte in partes[1:]:
        malla = malla + parte
    return malla


def cargar_pieza(robot, indice, pieza):
    """La pieza del STL si existe; si no, la provisional. Devuelve (malla, origen)."""
    if pieza.archivo:
        ruta = os.path.join(robot.carpeta_modelos, pieza.archivo)
        if os.path.exists(ruta):
            return pv.read(ruta), ruta
    return pieza_provisional(robot, indice), "provisional"


class Escena:
    """El robot dibujado en un plotter de PyVista (o en el QtInteractor de la ventana)."""

    def __init__(self, plotter, robot):
        self.plotter = plotter
        self.robot = robot
        self.actores = []
        self.origen_piezas = {}
        self.actor_volumen = None

        lado = 2 * robot.alcance
        piso = pv.Plane(i_size=lado, j_size=lado, i_resolution=20, j_resolution=20)
        self.plotter.add_mesh(piso, style="wireframe", color="gray", opacity=0.3)
        for indice, (nombre, pieza) in enumerate(piezas_del_robot(robot)):
            malla, origen = cargar_pieza(robot, indice, pieza)
            self.origen_piezas[nombre] = origen
            self.actores.append(self.plotter.add_mesh(malla, color=pieza.color, smooth_shading=True, specular=0.4))

    def dibujar(self, q):
        """Pone cada pieza en su lugar para los ángulos DH q."""
        for actor, T in zip(self.actores, poses_piezas(self.robot, q)):
            actor.user_matrix = T
        self.plotter.render()

    def mostrar_volumen(self, puntos):
        """Dibuja la nube del volumen de trabajo (reemplaza la anterior en vez de apilarla)."""
        if self.actor_volumen is not None:
            self.plotter.remove_actor(self.actor_volumen)
        self.actor_volumen = self.plotter.add_mesh(pv.PolyData(np.asarray(puntos)), color="cyan",
                                                   point_size=2, render_points_as_spheres=True)
        self.plotter.show_grid()
        self.plotter.render()

    def encuadrar(self, zoom=1.8):
        """Encuadra la escena; el zoom acerca al robot porque el encuadre incluye todo el piso."""
        self.plotter.reset_camera()
        self.plotter.camera.zoom(zoom)
