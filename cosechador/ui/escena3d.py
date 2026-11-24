"""Escena 3D del robot (PyVista). Funciona con cualquier robot cargado del YAML.

Cada pieza sale de modelos/<archivo> si existe (STL exportado EN MILÍMETROS); si no, se arma una
pieza provisional a partir de la tabla DH. En cada cuadro la geometría NO se regenera: solo se
cambia la matriz de posición de cada pieza (actor.user_matrix).

CONVENCIÓN DE CADA PIEZA (su propio origen y ejes; ver modelos/README.md):
- base:            fija; origen en el centro del piso, Z hacia arriba.
- articulación i:  gira con su servo. Origen en SU eje de giro (que es Z). La pieza sube d_i sobre Z
                   y luego avanza a_i sobre +X, hasta el eje de la siguiente articulación.
- efector:         origen en el extremo del último eslabón, apuntando a +X; su punta queda a `efector.largo` mm.
"""
import os

import numpy as np
import pyvista as pv

from cosechador.robot.cinematica import matrices, rot_z

ALCANCE_REFERENCIA = 2050.0  # LAS PIEZAS PROVISIONALES SE DISEÑARON PARA EL COSECHADOR; SE ESCALAN

# EJES CON LA CONVENCIÓN DE CAD Y ROS: X ROJO, Y VERDE, Z AZUL
COLORES_EJES = {"X": "#d62728", "Y": "#2ca02c", "Z": "#1f77b4"}
COLOR_PLAN, COLOR_RECORRIDO = "#2c6fbb", "#e07b22"
VISTAS = {"iso": lambda p: p.view_isometric(), "arriba": lambda p: p.view_xy(),
          "frente": lambda p: p.view_xz(), "lado": lambda p: p.view_yz()}
MAXIMO_RECORRIDO = 5000          # PUNTOS DE LA ESTELA (LOS MÁS VIEJOS SE DESCARTAN)
DISTANCIA_MINIMA_ESTELA = 0.5    # mm: NO SE AGREGAN PUNTOS REPETIDOS
# VOLUMEN DE TRABAJO: SUPERFICIE TRANSLÚCIDA (EL ROBOT Y LA TRAYECTORIA SE VEN A TRAVÉS) Y SU CONTORNO
COLOR_VOLUMEN, COLOR_BORDE_VOLUMEN, OPACIDAD_VOLUMEN = "#1aa3a3", "#0b4f6c", 0.3
GRADOS_POR_GAJO = 2              # LA SUPERFICIE DE GIRO SE ARMA CON GAJOS PLANOS DE ESTE ÁNGULO, COMO MUCHO


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
        # HERRAMIENTA PROVISIONAL: UN CONO QUE NACE EN EL EXTREMO DEL ÚLTIMO ESLABÓN, APUNTA AL FRENTE (+X) Y MIDE LO
        # QUE DICE EL DATO DEL ROBOT (efector.largo, EN mm REALES): SU PUNTA ES LA QUE DA cinematica_directa
        largo = robot.efector.largo
        return pv.Cone(center=(largo / 2, 0, 0), direction=(1, 0, 0), height=largo,
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


def superficie_de_giro(lazos, giro):
    """El contorno de un corte, girado sobre el eje Z de la base: (superficie, bordes).

    lazos: contornos del corte (ver espacio_trabajo.contorno_del_corte), cada uno (n, 2) con (alcance, altura).
    giro: (desde, hasta), en grados: hacia dónde mira la base en los dos extremos de su giro.
    La superficie queda abierta en esos dos extremos; `bordes` es el contorno del corte en cada uno.
    """
    desde, hasta = giro
    corte = None
    for lazo in lazos:
        linea = pv.lines_from_points(np.column_stack([lazo[:, 0], np.zeros(len(lazo)), lazo[:, 1]]), close=True)
        corte = linea if corte is None else corte + linea
    inicio = corte.rotate_z(desde)
    gajos = max(int(np.ceil(abs(hasta - desde) / GRADOS_POR_GAJO)), 1)
    superficie = inicio.extrude_rotate(resolution=gajos, angle=hasta - desde, capping=False)
    return superficie, inicio + corte.rotate_z(hasta)


class Escena:
    """El robot dibujado en un plotter de PyVista (o en el QtInteractor de la ventana)."""

    def __init__(self, plotter, robot):
        self.plotter = plotter
        self.robot = robot
        self.actores = []
        self.origen_piezas = {}
        self.actores_volumen = []

        lado = 2 * robot.alcance
        piso = pv.Plane(i_size=lado, j_size=lado, i_resolution=20, j_resolution=20)
        self.plotter.add_mesh(piso, style="wireframe", color="gray", opacity=0.3)
        for indice, (nombre, pieza) in enumerate(piezas_del_robot(robot)):
            malla, origen = cargar_pieza(robot, indice, pieza)
            self.origen_piezas[nombre] = origen
            self.actores.append(self.plotter.add_mesh(malla, color=pieza.color, smooth_shading=True, specular=0.4))
        self._dibujar_ejes()
        self._plan = self._recorrido = None
        self._actor_plan = self._actor_recorrido = None
        self.puntos_recorrido = []
        self.trayectoria_visible = True

    def _dibujar_ejes(self):
        """Flechas X/Y/Z en el origen de la base, con su letra, y el cubo de orientación en la esquina."""
        largo = 0.3 * self.robot.alcance
        for eje, direccion in (("X", (1, 0, 0)), ("Y", (0, 1, 0)), ("Z", (0, 0, 1))):
            self.plotter.add_mesh(pv.Arrow(start=(0, 0, 0), direction=direccion, scale=largo, tip_length=0.15,
                                           shaft_radius=0.012, tip_radius=0.04), color=COLORES_EJES[eje])
            self.plotter.add_point_labels(np.array([direccion], dtype=float) * largo * 1.12, [eje], font_size=18,
                                          text_color=COLORES_EJES[eje], shape=None, show_points=False, always_visible=True)
        self.plotter.add_axes(interactive=False, line_width=3)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAYECTORIA DE LA PUNTA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def _linea(self, malla, actor, puntos, color, ancho, opacidad):
        puntos = np.asarray(puntos, dtype=float)
        if len(puntos) < 2:
            if actor is not None:
                actor.SetVisibility(False)
            return malla, actor
        nueva = pv.lines_from_points(puntos)
        if malla is None:
            malla = nueva
            actor = self.plotter.add_mesh(malla, color=color, line_width=ancho, opacity=opacidad)
        else:
            malla.copy_from(nueva)
        actor.SetVisibility(self.trayectoria_visible)
        return malla, actor

    def mostrar_plan(self, puntos):
        """Línea tenue con lo que va a recorrer la punta."""
        self._plan, self._actor_plan = self._linea(self._plan, self._actor_plan, puntos, COLOR_PLAN, 3, 0.55)
        self.plotter.render()

    def reiniciar_recorrido(self, punto_inicial):
        self.puntos_recorrido = [np.asarray(punto_inicial, dtype=float)]
        self._recorrido, self._actor_recorrido = self._linea(self._recorrido, self._actor_recorrido,
                                                             self.puntos_recorrido, COLOR_RECORRIDO, 5, 1.0)

    def agregar_al_recorrido(self, punto):
        """Estela sólida de lo que ya recorrió la punta (se llama en cada paso)."""
        punto = np.asarray(punto, dtype=float)
        if self.puntos_recorrido and np.linalg.norm(punto - self.puntos_recorrido[-1]) < DISTANCIA_MINIMA_ESTELA:
            return
        self.puntos_recorrido.append(punto)
        del self.puntos_recorrido[:-MAXIMO_RECORRIDO]
        self._recorrido, self._actor_recorrido = self._linea(self._recorrido, self._actor_recorrido,
                                                             self.puntos_recorrido, COLOR_RECORRIDO, 5, 1.0)

    def limpiar_trayectoria(self, punto_actual):
        self.mostrar_plan([])
        self.reiniciar_recorrido(punto_actual)
        self.plotter.render()

    def mostrar_trayectoria(self, visible):
        self.trayectoria_visible = bool(visible)
        for actor, malla in ((self._actor_plan, self._plan), (self._actor_recorrido, self._recorrido)):
            if actor is not None:
                actor.SetVisibility(self.trayectoria_visible and malla is not None and malla.n_points >= 2)
        self.plotter.render()

    def dibujar(self, q):
        """Pone cada pieza en su lugar para los ángulos DH q."""
        for actor, T in zip(self.actores, poses_piezas(self.robot, q)):
            actor.user_matrix = T
        self.plotter.render()

    def _poner_volumen(self, *actores):
        """El volumen de trabajo recién dibujado reemplaza al anterior (nube o superficie) en vez de apilarse.
        Sin actores, el volumen se quita, con todo y su rejilla de medidas."""
        for actor in self.actores_volumen:
            self.plotter.remove_actor(actor)
        self.actores_volumen = list(actores)
        if actores:
            self.plotter.show_grid()
        else:
            self.plotter.remove_bounds_axes()
        self.plotter.render()

    def ocultar_volumen(self):
        """Quita el volumen de trabajo de la escena: queda como antes de dibujarlo."""
        self._poner_volumen()

    def mostrar_volumen(self, puntos):
        """Dibuja el volumen de trabajo como nube de puntos: para el robot cuyo corte no se puede girar."""
        self._poner_volumen(self.plotter.add_mesh(pv.PolyData(np.asarray(puntos)), color="cyan",
                                                  point_size=2, render_points_as_spheres=True))

    def mostrar_superficie(self, lazos, giro):
        """Dibuja el volumen de trabajo como superficie translúcida, con su contorno (ver superficie_de_giro)."""
        superficie, bordes = superficie_de_giro(lazos, giro)
        self._poner_volumen(self.plotter.add_mesh(superficie, color=COLOR_VOLUMEN, opacity=OPACIDAD_VOLUMEN),
                            self.plotter.add_mesh(bordes, color=COLOR_BORDE_VOLUMEN, line_width=2))

    def encuadrar(self, zoom=1.8):
        """Encuadra la escena sin cambiar el ángulo; el zoom acerca al robot porque el encuadre incluye todo el piso."""
        self.plotter.reset_camera()
        self.plotter.camera.zoom(zoom)

    def vista(self, nombre):
        """Pone la cámara en una vista fija: iso, arriba (plano X-Y), frente (X-Z) o lado (Y-Z)."""
        VISTAS[nombre](self.plotter)
        self.encuadrar(1.8 if nombre == "iso" else 1.4)
        self.plotter.render()
