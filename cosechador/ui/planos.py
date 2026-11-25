"""Las vistas planas del robot: LATERAL y SUPERIOR (matplotlib), pensadas para vivir dentro de la ventana.

- Todo se mide en PÍXELES LÓGICOS: la figura va a 100 dpi, así que 1 px = 0.72 pt. Ninguna letra baja de LETRA_MINIMA (la
  «letra_chica» del tema). Si el recuadro es chico NO se encoge la letra: los letreros se acortan o se quitan.
- Lo que NO cambia con la postura (contorno, relleno, cotas, rejilla) se dibuja una vez y se guarda como fondo; lo que SÍ
  cambia (el robot) son cuatro artistas «animados»: en cada paso solo se repintan esos (blitting).
- Cada dibujo sabe decir dónde quedó cada rótulo (`rotulos()`): texto, rectángulo en px del lienzo y tamaño de letra. De ahí
  sale la revisión del auditor (tests/auditoria_interfaz.py): nada se enciman, nada se sale, nada baja de la letra mínima.
- Con el volumen a la vista se dibuja su contorno (lazos del corte) y sus medidas; sin él (o si el robot no tiene un corte
  plano que dibujar) se ve el robot, el nivel de la base y los ejes.
"""
from collections import namedtuple

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Arc, PathPatch, Rectangle, Wedge
from matplotlib.path import Path

from cosechador.robot.cinematica import matrices, servo_a_dh
from cosechador.robot.espacio_trabajo import giro_de_la_base

from . import tema
from .escena3d import COLOR_BORDE_VOLUMEN, COLOR_VOLUMEN, COLORES_EJES, OPACIDAD_VOLUMEN

T = tema.TOKENS
PT = 0.72                       # 1 px LÓGICO EN PUNTOS (FIGURA A 100 dpi)
LETRA_MINIMA = 12               # px: LA letra_chica DEL TEMA; NINGÚN RÓTULO DEL DIBUJO ES MÁS CHICO (LA DECLARA EL AUDITOR)
FAMILIA = ["Arial", "DejaVu Sans"]   # matplotlib NO PUEDE USAR LA LETRA DEL SISTEMA EN NEGRITAS (SFNS.ttf SOLO TRAE UN PESO)
TINTA, SUAVE, ROJO, BORDE_SUAVE = T["tinta"], T["tinta_suave"], T["rojo"], T["borde_suave"]
GRIS_BAJO_LA_BASE = "#dfe3e7"   # LO QUE EL ROBOT ALCANZA POR DEBAJO DEL NIVEL DE SU BASE, EN EL PLANO
GRANDE = (480, 300)             # DESDE ESTE TAMAÑO (px) EL DIBUJO LLEVA NÚMEROS EN LOS EJES Y NOMBRES EN LAS COTAS
EPS = 1e-6

Rotulo = namedtuple("Rotulo", "texto x0 y0 x1 y1 letra")   # px DEL LIENZO (EL ORIGEN, ARRIBA A LA IZQUIERDA) Y px DE LETRA


def mezclar(color, opacidad, fondo="#ffffff"):
    """El color que se ve al poner `color` translúcido sobre `fondo`."""
    c, f = (np.array([int(x[i:i + 2], 16) for i in (1, 3, 5)], float) for x in (color, fondo))
    return "#%02x%02x%02x" % tuple(int(round(v)) for v in f + (c - f) * opacidad)


RELLENO = mezclar(COLOR_VOLUMEN, OPACIDAD_VOLUMEN)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GEOMETRÍA (SIN Qt) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class Geometria:
    """Lo que los dibujos necesitan saber del robot: dónde está cada articulación en cada plano y cuánto mide el volumen.

    `lazos`: el contorno del corte (contorno_del_corte) o None si no hay uno que dibujar (todavía no se calculó, o el robot
    no tiene un corte plano). Todo lo del volumen sale de ahí; lo del robot, de su cinemática: cualquier robot, no el de ahora.
    """

    def __init__(self, robot, lazos=None):
        self.robot = robot
        self.lazos = [np.asarray(lazo, float) for lazo in lazos] if lazos else None
        self.desde, self.hasta = giro_de_la_base(robot)
        self.alcance = float(robot.alcance_de_la_punta)
        if self.lazos:
            p = np.vstack(self.lazos)
            self.xmin, self.xmax, self.zmin, self.zmax = (float(x) for x in (p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()))
            self.z_en_xmax, self.z_en_xmin = float(p[np.argmax(p[:, 0]), 1]), float(p[np.argmin(p[:, 0]), 1])
            self.x_en_zmax, self.x_en_zmin = float(p[np.argmax(p[:, 1]), 0]), float(p[np.argmin(p[:, 1]), 0])
            self.frente, self.atras = self.xmax, abs(min(self.xmin, 0.0))
        else:
            self.frente = self.atras = self.alcance
        self._esquinas = None

    @property
    def con_volumen(self):
        return self.lazos is not None

    # %%%%% EL ROBOT EN CADA PLANO %%%%%
    def puntos(self, servos):
        """Base, marco de cada articulación y punta de la herramienta, en el mundo: (n + 2, 3)."""
        m = matrices(self.robot, servo_a_dh(self.robot, servos))
        punta = m[-1] @ (self.robot.efector.largo, 0.0, 0.0, 1.0)
        return np.array([x[:3, 3] for x in m] + [punta[:3]], dtype=float)

    def lateral(self, servos):
        """La postura en el plano vertical del brazo: (n + 2, 2) con (alcance con signo, altura). Mira hacia donde apunta la
        base: por eso el brazo se ve siempre de perfil, sin importar cuánto haya girado."""
        p = self.puntos(servos)
        theta = np.radians(servo_a_dh(self.robot, servos)[0])
        return np.column_stack([p[:, 0] * np.cos(theta) + p[:, 1] * np.sin(theta), p[:, 2]])

    def superior(self, servos):
        """La postura vista desde arriba: (n + 2, 2) con (x, y)."""
        return self.puntos(servos)[:, :2]

    # %%%%% MARCOS %%%%%
    def marco_lateral(self, con_volumen):
        """(x0, x1, z0, z1) en mm de lo que se enmarca: el volumen, o el robot en las posturas extremas."""
        if con_volumen and self.con_volumen:
            x0, x1, z0, z1 = self.xmin, self.xmax, self.zmin, self.zmax
        else:
            if self._esquinas is None:
                import itertools
                rangos = [(j.servo.minimo, (j.servo.minimo + j.servo.maximo) / 2, j.servo.maximo) for j in self.robot.articulaciones]
                self._esquinas = np.vstack([self.lateral(s) for s in itertools.product(*rangos)])
            x0, z0 = self._esquinas.min(axis=0)
            x1, z1 = self._esquinas.max(axis=0)
        return min(x0, 0.0), max(x1, 0.0), min(z0, 0.0), max(z1, 0.0)

    def radio(self, grados):
        """Hasta dónde llega el volumen visto desde arriba en la dirección `grados` (0 si no llega)."""
        a, b = sorted((self.desde, self.hasta))
        dentro = lambda fi, lo, hi: (b - a) >= 360 - EPS or ((fi - lo) % 360) <= (hi - lo) + EPS
        return max(self.frente if dentro(grados, a, b) else 0.0, self.atras if dentro(grados, a + 180, b + 180) else 0.0)

    def tramos(self):
        """El contorno visto desde arriba en tramos de radio constante: [(desde°, hasta°, radio)], de menor a mayor ángulo.

        El radio solo cambia donde empieza o acaba un sector (el del frente: de `desde` a `hasta`; el de atrás: lo mismo
        girado 180°). Vale para CUALQUIER giro de la base, no solo para 180°."""
        a, b = sorted((self.desde, self.hasta))
        if b - a >= 360 - EPS:
            return [(0.0, 360.0, self.radio(0.0))]
        cortes = sorted({round(x % 360, 9) for x in (a, b, a + 180, b + 180)})
        tramos = []
        for p, q in zip(cortes, cortes[1:] + [cortes[0] + 360]):
            tramos.append((p, q, self.radio((p + q) / 2)))
        return tramos

    def marco_superior(self, con_volumen):
        if con_volumen and self.con_volumen:
            puntos = [(0.0, 0.0)]
            for p, q, r in self.tramos():
                if r > 0:
                    fi = np.radians(np.linspace(p, q, max(int(abs(q - p)), 2) + 1))
                    puntos += list(zip(r * np.cos(fi), r * np.sin(fi)))
            puntos = np.array(puntos)
            return float(puntos[:, 0].min()), float(puntos[:, 0].max()), float(puntos[:, 1].min()), float(puntos[:, 1].max())
        return -self.alcance, self.alcance, -self.alcance, self.alcance


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOS DIBUJOS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class Dibujo:
    TITULO = ""

    def __init__(self, geometria, clase_lienzo=FigureCanvasQTAgg, servos=None):
        self.geo, self.robot = geometria, geometria.robot
        self.fig = Figure(figsize=(3, 2), dpi=100, facecolor=T["fondo"])
        self.lienzo = clase_lienzo(self.fig)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.volumen_visible, self.con_titulo = False, True
        self.servos = tuple(servos if servos is not None else self.robot.postura_home or [j.servo.centro for j in self.robot.articulaciones])
        self._fondo, self.moviles, self.textos, self.grande, self.s, self._ocupado = None, [], [], False, 1.0, []
        self.lienzo.mpl_connect("resize_event", lambda evento: self.componer())
        self.lienzo.mpl_connect("draw_event", self._al_dibujar)
        self.componer()

    # %%%%% TAMAÑO Y MARCO %%%%%
    def px(self):
        """Ancho y alto del dibujo en píxeles lógicos (los de Qt), sin importar la densidad de la pantalla."""
        ancho, alto = self.fig.get_size_inches() * 100
        return float(ancho), float(alto)

    @property
    def con_volumen(self):
        """¿Se dibuja el contorno? Solo si se pidió y hay uno (un robot sin corte plano no lo tiene)."""
        return self.volumen_visible and self.geo.con_volumen

    def componer(self):
        """Arma TODO el dibujo para el tamaño que tiene ahora. Se llama al cambiar de tamaño o al cambiar el volumen."""
        ancho, alto = self.px()
        self.grande = ancho >= GRANDE[0] and alto >= GRANDE[1]
        self.ax.clear()
        self._fondo, self.textos, self.moviles, self._ocupado = None, [], [], []
        self._componer(ancho, alto)
        self._colocar_robot()

    def _marco(self, ancho, alto, x0, x1, y0, y1, izq, der, inf, sup):
        """Coloca el eje y sus límites: el contenido (x0..x1, y0..y1 en mm) queda centrado, a la MISMA escala en los dos
        ejes, dejando libres `izq`, `der`, `inf` y `sup` píxeles para cotas y letreros. Devuelve la escala (px por mm)."""
        m_izq, m_inf, m_sup, m_der = (64, 40, 8, 10) if self.grande else (1, 1, 1, 1)
        ancho_eje, alto_eje = max(ancho - m_izq - m_der, 10), max(alto - m_inf - m_sup, 10)
        self.ax.set_position([m_izq / ancho, m_inf / alto, ancho_eje / ancho, alto_eje / alto])
        s = max(min((ancho_eje - izq - der) / max(x1 - x0, 1.0), (alto_eje - inf - sup) / max(y1 - y0, 1.0)), 1e-3)
        xa = x0 - (izq + (ancho_eje - izq - der - s * (x1 - x0)) / 2) / s
        ya = y0 - (inf + (alto_eje - inf - sup - s * (y1 - y0)) / 2) / s
        self.ax.set_xlim(xa, xa + ancho_eje / s)
        self.ax.set_ylim(ya, ya + alto_eje / s)
        paso = 500 if s * 500 >= (48 if self.grande else 14) else 1000
        # SOLO LOS NÚMEROS QUE CABEN ENTEROS EN EL LIENZO: UNO CENTRADO EN UNA MARCA PEGADA A LA ORILLA SE QUEDARÍA A LA MITAD
        # (SE ESTIMA EL ANCHO DE UN NÚMERO EN 7.5 px POR CIFRA O SIGNO Y SU ALTO EN 1.3 VECES LA LETRA)
        xt, yt = (np.arange(np.ceil(a / paso) * paso, a + largo / s, paso) for a, largo in ((xa, ancho_eje), (ya, alto_eje)))
        if self.grande:
            xt = [t for t in xt if 3.75 * len(f"{t:.0f}") + 1 <= m_izq + (t - xa) * s <= ancho - 3.75 * len(f"{t:.0f}") - 1]
            yt = [t for t in yt if 0.65 * LETRA_MINIMA + 1 <= m_inf + (t - ya) * s <= alto - 0.65 * LETRA_MINIMA - 1]
        self.ax.set_xticks(xt)
        self.ax.set_yticks(yt)
        self.ax.grid(True, color=BORDE_SUAVE, alpha=0.35, linewidth=0.8 * PT)
        self.ax.set_axisbelow(True)
        for lado in self.ax.spines.values():
            lado.set_color(BORDE_SUAVE)
            lado.set_linewidth(1 * PT)
        if self.grande:
            # ax.clear() NO REPONE LO QUE SE APAGÓ CON tick_params: HAY QUE VOLVER A PRENDER LOS NÚMEROS
            self.ax.tick_params(labelsize=LETRA_MINIMA * PT, labelcolor=SUAVE, color=BORDE_SUAVE, length=3,
                                labelfontfamily=FAMILIA[0], labelbottom=True, labelleft=True)
        else:
            self.ax.tick_params(labelbottom=False, labelleft=False, length=0)
        self.s = s
        return s

    # %%%%% RÓTULOS: SE COLOCAN DONDE NO ESTORBAN %%%%%
    def _caja(self, t):
        """El rectángulo de un texto en px lógicos del lienzo, con el origen ABAJO a la izquierda (el de matplotlib)."""
        c, k = t.get_window_extent(self.lienzo.get_renderer()), self.fig.dpi / 100
        return (c.x0 / k, c.y0 / k, c.x1 / k, c.y1 / k)

    def _cabe(self, caja):
        ancho, alto = self.px()
        ax = self.ax.get_position()
        x0, y0, x1, y1 = ax.x0 * ancho, ax.y0 * alto, ax.x1 * ancho, ax.y1 * alto   # DENTRO DEL EJE: LOS NÚMEROS VAN AFUERA
        if caja[0] < x0 - 0.5 or caja[1] < y0 - 0.5 or caja[2] > x1 + 0.5 or caja[3] > y1 + 0.5:
            return False
        return not any(min(caja[2], o[2]) - max(caja[0], o[0]) > 0.5 and min(caja[3], o[3]) - max(caja[1], o[1]) > 0.5
                       for o in self._ocupado)

    def _letrero(self, variantes, posiciones, opcional=False, tramo=None, letra=LETRA_MINIMA, **estilo):
        """Pone un letrero donde cabe: prueba cada texto (del más completo al más corto) en cada posición (de la mejor a la
        peor) y se queda con el primero que está dentro del eje, no se enciman con otro y, si lleva `tramo` (px), cabe en
        él. Si no cabe en ninguna: se quita (`opcional`) o queda con el texto más corto en la primera posición.
        Cada posición es (xy, desfase_px, coordenadas, ha, va)."""
        estilo.setdefault("color", TINTA)
        variantes = [variantes] if isinstance(variantes, str) else list(variantes)
        t = None
        for texto in variantes:
            for xy, desfase, coordenadas, ha, va in posiciones:
                if t is None:
                    t = self.ax.annotate(texto, xy, xycoords=coordenadas, xytext=(desfase[0] * PT, desfase[1] * PT),
                                         textcoords="offset points", fontsize=letra * PT, fontfamily=FAMILIA, ha=ha, va=va, **estilo)
                    self.textos.append(t)
                else:
                    t.set_text(texto)
                    t.xy, t.xycoords, t.ha, t.va = xy, coordenadas, ha, va
                    t.set_position((desfase[0] * PT, desfase[1] * PT))
                caja = self._caja(t)
                largo = (caja[3] - caja[1]) if estilo.get("rotation") == 90 else (caja[2] - caja[0])
                if self._cabe(caja) and (tramo is None or largo <= tramo):
                    self._ocupado.append(caja)
                    return t
        if opcional:
            t.set_visible(False)
            return None
        xy, desfase, coordenadas, ha, va = posiciones[0]
        t.set_text(variantes[-1])
        t.xy, t.xycoords, t.ha, t.va = xy, coordenadas, ha, va
        t.set_position((desfase[0] * PT, desfase[1] * PT))
        self._ocupado.append(self._caja(t))
        return t

    def _titulo(self):
        if self.con_titulo:
            self._letrero(self.TITULO, [((0, 1), (6, -5), "axes fraction", "left", "top")], color=SUAVE, fontweight="bold")

    def _roseta(self, horizontal, vertical):
        """Sin números en los ejes: dos flechitas en la esquina inferior izquierda que dicen hacia dónde va cada eje.
        `horizontal` y `vertical` son (letrero, color)."""
        for (letrero, color), (dx, dy) in ((horizontal, (1, 0)), (vertical, (0, 1))):
            self.ax.annotate("", ((10 + 20 * dx) * PT, (10 + 20 * dy) * PT), (10 * PT, 10 * PT), xycoords="axes points",
                             textcoords="axes points", arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.6 * PT,
                                                                       shrinkA=0, shrinkB=0, mutation_scale=8 * PT))
            self._letrero(letrero, [((10 + 27 * dx, 10 + 27 * dy), (0, 0), "axes points", "center" if dy else "left",
                                     "bottom" if dy else "center")], fontweight="bold")

    def _ejes(self, horizontal, vertical):
        if self.grande:
            self.ax.set_xlabel(horizontal, fontsize=LETRA_MINIMA * PT, color=SUAVE, fontfamily=FAMILIA, labelpad=2)
            self.ax.set_ylabel(vertical, fontsize=LETRA_MINIMA * PT, color=SUAVE, fontfamily=FAMILIA, labelpad=2)

    # %%%%% EL ROBOT %%%%%
    def _robot(self, grosor):
        """Los artistas que SÍ cambian con la postura (animados: no entran al fondo)."""
        ax, eslabon = self.ax, self.robot.articulaciones[0].pieza.color   # EL COLOR DE LA COLUMNA EN EL YAML
        self.l_brazo, = ax.plot([], [], color=eslabon, linewidth=grosor * PT, solid_capstyle="round",
                                solid_joinstyle="round", animated=True, zorder=5)
        self.l_herramienta, = ax.plot([], [], color=SUAVE, linewidth=max(grosor * 0.55, 2) * PT, solid_capstyle="round",
                                      animated=True, zorder=5)
        self.p_ejes, = ax.plot([], [], linestyle="none", marker="o", markersize=(grosor + 2) * PT, markerfacecolor="white",
                               markeredgecolor=TINTA, markeredgewidth=1.3 * PT, animated=True, zorder=6)
        self.p_punta, = ax.plot([], [], linestyle="none", marker="o", markersize=(grosor + 1) * PT, markerfacecolor=ROJO,
                                markeredgecolor=ROJO, animated=True, zorder=7)
        self.moviles = [self.l_brazo, self.l_herramienta, self.p_ejes, self.p_punta]

    def _poner_puntos(self, p):
        """p: base, marco de cada articulación y punta de la herramienta, en este plano."""
        self.l_brazo.set_data(p[:-1, 0], p[:-1, 1])
        self.l_herramienta.set_data(p[-2:, 0], p[-2:, 1])
        self.p_ejes.set_data(p[1:-1, 0], p[1:-1, 1])
        self.p_punta.set_data(p[-1:, 0], p[-1:, 1])

    # %%%%% DIBUJAR %%%%%
    def _al_dibujar(self, evento):
        """Después de cada dibujo completo: se guarda el fondo (sin robot) y se pinta el robot encima."""
        self._fondo = self.lienzo.copy_from_bbox(self.fig.bbox)
        for movil in self.moviles:
            self.ax.draw_artist(movil)

    def anotar_pose(self, servos):
        """La postura nueva SIN pintar: para el dibujo que no está a la vista (al verse sale al día)."""
        self.servos = tuple(servos)
        self._colocar_robot()

    def poner_pose(self, servos, parcial=True):
        """El robot en su postura nueva. parcial=True: se repone el fondo guardado y solo se pintan los artistas del robot
        (blitting). parcial=False: se vuelve a dibujar todo (draw())."""
        self.anotar_pose(servos)
        if parcial and self._fondo is not None:
            self.lienzo.restore_region(self._fondo)
            for movil in self.moviles:
                self.ax.draw_artist(movil)
            self.lienzo.blit(self.fig.bbox)
        else:
            self.lienzo.draw()

    def poner_geometria(self, geometria, volumen_visible):
        """Cambió el contorno, el robot o si el volumen está a la vista: el dibujo se arma de nuevo."""
        self.geo, self.robot, self.volumen_visible = geometria, geometria.robot, bool(volumen_visible)
        self.componer()
        self.lienzo.draw()

    # %%%%% MEDIR (EL AUDITOR DE Qt NO VE DENTRO DE UN DIBUJO) %%%%%
    def rotulos(self):
        """Cada rótulo del dibujo tal como quedó (textos, números de los ejes y sus títulos): lista de Rotulo."""
        self.lienzo.draw()
        alto = self.px()[1]
        quienes = [t for t in self.textos if t.get_visible() and t.get_text()]
        if self.grande:
            quienes += [t for t in self.ax.get_xticklabels() + self.ax.get_yticklabels() + [self.ax.xaxis.label, self.ax.yaxis.label]
                        if t.get_visible() and t.get_text()]
        rotulos = []
        for t in quienes:
            x0, y0, x1, y1 = self._caja(t)
            rotulos.append(Rotulo(t.get_text().replace("\n", " "), x0, alto - y1, x1, alto - y0, round(t.get_fontsize() / PT, 1)))
        return rotulos

    def letras(self):
        """Tamaños de letra (px) que hay a la vista en el dibujo."""
        return sorted({r.letra for r in self.rotulos()})


class Lateral(Dibujo):
    """El corte del volumen en el plano del brazo, con el robot en su postura."""
    TITULO = "Lateral"

    def _componer(self, ancho, alto):
        g, ax = self.geo, self.ax
        cota = 13 if self.grande else 12
        con = self.con_volumen
        x0, x1, z0, z1 = g.marco_lateral(con)
        # EL ESPACIO DE LAS COTAS SE APARTA SIEMPRE QUE HAYA VOLUMEN: SIN ÉL SE ENMARCA EL ROBOT
        extra = (10 + 3 + cota + 5, 10 + 3 + cota + 4) if con else (8, 8)
        s = self._marco(ancho, alto, x0, x1, z0, z1, izq=8, der=extra[0], inf=extra[1], sup=8)
        self._titulo()
        if con:
            vertices = np.vstack([np.vstack([lazo, lazo[:1]]) for lazo in g.lazos])
            codigos = np.concatenate([[Path.MOVETO] + [Path.LINETO] * (len(lazo) - 1) + [Path.CLOSEPOLY] for lazo in g.lazos])
            for color, y0 in ((RELLENO, 0.0), (GRIS_BAJO_LA_BASE, -1e6)):   # ARRIBA DE LA BASE EN COLOR, ABAJO EN GRIS
                parche = PathPatch(Path(vertices, codigos), facecolor=color, edgecolor="none")
                ax.add_patch(parche)
                parche.set_clip_path(Rectangle((-1e6, y0), 2e6, 1e6, transform=ax.transData))
            for lazo in g.lazos:
                ax.plot(*np.vstack([lazo, lazo[:1]]).T, color=COLOR_BORDE_VOLUMEN, linewidth=1.6 * PT)
        # EL NIVEL DE LA BASE LLEGA HASTA LA COTA VERTICAL, NO HASTA LA ORILLA: ASÍ NO CRUZA SU LETRERO («abajo 470»)
        hasta = g.xmax + 10 / s if con else ax.get_xlim()[1]
        ax.plot([ax.get_xlim()[0], hasta], [0, 0], color=SUAVE, linewidth=1.2 * PT, linestyle=(0, (6, 4)))
        if not con:
            # SIN VOLUMEN NO HAY COTAS QUE DIGAN PARA DÓNDE ES EL FRENTE: LO DICE LA ROSETA (O LOS EJES, EN GRANDE). VA ANTES QUE
            # «nivel de la base» PARA QUE ESE LETRERO ESQUIVE SUS LETRAS (EL NIVEL DE LA BASE CAE CERCA DE LA ESQUINA)
            self._roseta(("frente", SUAVE), ("Z", COLORES_EJES["Z"]))
        self._letrero(["nivel de la base", "base"], [((ax.get_xlim()[0], 0), (6, 3), "data", "left", "bottom"),
                                                     ((ax.get_xlim()[1], 0), (-6, 3), "data", "right", "bottom")],
                      opcional=True, color=SUAVE)
        if con:
            self._cotas(s, cota)
        self._ejes("alcance horizontal (mm)", "altura (mm)")
        self._robot(min(max(70 * s, 3.0), 9.0))

    def _cotas(self, s, cota):
        """Las cuatro medidas del contorno: frente, atrás, arriba y abajo (la altura mínima, bajo el nivel de la base)."""
        g, ax = self.geo, self.ax
        flecha = dict(arrowstyle="<->", color=ROJO, linewidth=1.2 * PT, shrinkA=0, shrinkB=0, mutation_scale=9 * PT)
        fina = dict(color=ROJO, linewidth=0.8 * PT, alpha=0.6)
        yc, xc = g.zmin - 10 / s if g.zmin < 0 else -10 / s, g.xmax + 10 / s
        for x, z in ((g.xmax, g.z_en_xmax), (g.xmin, g.z_en_xmin), (0, 0)):
            ax.plot([x, x], [z, yc], **fina)        # HASTA LA COTA Y NI UN PÍXEL MÁS: ASÍ NO CRUZAN SU LETRERO
        for z, x in ((g.zmax, g.x_en_zmax), (g.zmin, g.x_en_zmin), (0, g.xmax)):
            ax.plot([x, xc], [z, z], **fina)
        estilo = dict(letra=cota, color=ROJO, fontweight="bold")
        for a, b, completo, corto in ((0, g.xmax, "al frente", "frente"), (g.xmin, 0, "hacia atrás", "atrás")):
            if b - a <= 0:
                continue
            ax.annotate("", (b, yc), (a, yc), arrowprops=flecha)
            valor = f"{b - a:.0f}"
            # SI NO CABE CON SU NOMBRE, QUEDA EL NÚMERO: «frente» Y «atrás» SE ADIVINAN POR LA POSICIÓN DE LA FLECHA
            variantes = ([f"{completo} {valor}", f"{corto} {valor}", valor] if self.grande else [f"{corto} {valor}", valor])
            self._letrero(variantes, [(((a + b) / 2, yc), (0, -3), "data", "center", "top")], tramo=(b - a) * s - 6, **estilo)
        for a, b, completo in ((0, g.zmax, "arriba"), (g.zmin, 0, "abajo")):
            if b - a <= 0:
                continue
            ax.annotate("", (xc, b), (xc, a), arrowprops=flecha)
            valor = f"{b - a:.0f}"
            # EL LETRERO VA AL LADO DE LA FLECHA, GIRADO: PUEDE SER MÁS LARGO QUE SU TRAMO SIN CRUZARLA
            self._letrero([f"{completo} {valor}", valor] if self.grande else [valor], [((xc, (a + b) / 2), (3, 0), "data", "left", "center")],
                          rotation=90, **estilo)

    def _colocar_robot(self):
        self._poner_puntos(self.geo.lateral(self.servos))


class Superior(Dibujo):
    """El volumen visto desde arriba (el corte girado lo que gira la base), con el robot en su postura."""
    TITULO = "Superior"

    def _componer(self, ancho, alto):
        g, ax = self.geo, self.ax
        cota = 13 if self.grande else 12
        con = self.con_volumen
        x0, x1, y0, y1 = g.marco_superior(con)
        s = self._marco(ancho, alto, x0, x1, y0, y1, 6, 6, 6, 6)
        self._titulo()
        if con:
            self._volumen(s, cota)
        else:
            ax.add_patch(Arc((0, 0), 2 * g.alcance, 2 * g.alcance, theta1=0, theta2=360, color=BORDE_SUAVE, linewidth=1.2 * PT,
                             linestyle=(0, (6, 4))))   # EL ALCANCE DEL ROBOT: UN CÍRCULO TENUE
            self._roseta(("X", COLORES_EJES["X"]), ("Y", COLORES_EJES["Y"]))
        self._ejes("X (mm)", "Y (mm)")
        self._robot(min(max(70 * s, 3.0), 9.0))

    def _volumen(self, s, cota):
        """El contorno por tramos de radio constante (vale para cualquier giro), sus cortes radiales y las medidas."""
        g, ax = self.geo, self.ax
        tramos = g.tramos()
        for p, q, r in tramos:
            if r > 0:
                ax.add_patch(Wedge((0, 0), r, p, q, facecolor=RELLENO, edgecolor="none"))
                ax.add_patch(Arc((0, 0), 2 * r, 2 * r, theta1=p, theta2=q, color=COLOR_BORDE_VOLUMEN, linewidth=1.6 * PT))
        # DONDE EL RADIO CAMBIA (O EMPIEZA EL VACÍO) EL CONTORNO TIENE UN ESCALÓN RADIAL; EL ÚLTIMO TRAMO CIERRA CON EL PRIMERO
        for k, (p, q, r) in enumerate(tramos):
            antes = tramos[k - 1][2]
            if abs(antes - r) > EPS and len(tramos) > 1:
                c, sn = np.cos(np.radians(p)), np.sin(np.radians(p))
                ax.plot([min(antes, r) * c, max(antes, r) * c], [min(antes, r) * sn, max(antes, r) * sn],
                        color=COLOR_BORDE_VOLUMEN, linewidth=1.6 * PT)
        flecha = dict(arrowstyle="<->", color=ROJO, linewidth=1.2 * PT, shrinkA=0, shrinkB=0, mutation_scale=9 * PT)
        # LAS COTAS: UNA POR RADIO DISTINTO, EN LA MITAD DE SU TRAMO MÁS ANCHO (EL DEL FRENTE Y EL DE ATRÁS)
        radios = sorted({round(r, 6) for _, _, r in tramos if r > 0}, reverse=True)
        for radio in radios:
            p, q, _ = max((t for t in tramos if abs(t[2] - radio) < 1e-5), key=lambda t: t[1] - t[0])
            fi = np.radians(p + 0.5 * (q - p) if q - p < 90 else p + 0.7 * (q - p))
            c, sn = np.cos(fi), np.sin(fi)
            ax.annotate("", (radio * c, radio * sn), (0, 0), arrowprops=flecha)
            posiciones = [((f * radio * c, f * radio * sn), (0, 0), "data", "center", "center") for f in (0.62, 0.5, 0.75, 0.38)]
            self._letrero(f"R {radio:.0f}", posiciones, letra=cota, color=ROJO, fontweight="bold",
                          bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.85))
        a, b = sorted((g.desde, g.hasta))
        medio = np.radians((a + b) / 2)
        if g.frente > 0 and b - a < 360 - EPS:      # CON UN GIRO COMPLETO NO HAY «FRENTE» NI «ATRÁS» QUE SEÑALAR
            self._letrero("al frente", [((f * g.frente * np.cos(medio), f * g.frente * np.sin(medio)), (0, 0), "data", "center", "center")
                                        for f in (0.82, 0.3, 0.62)], opcional=True, color=COLOR_BORDE_VOLUMEN, fontweight="bold")
        if g.atras > 0 and b - a < 360 - EPS:
            ang = np.radians((a + b) / 2 + 180)
            self._letrero(["hacia atrás\n(por encima de la base)", "hacia atrás"] if self.grande else ["hacia atrás"],
                          [((f * g.atras * np.cos(ang), f * g.atras * np.sin(ang)), (0, 0), "data", "center", "center")
                           for f in (0.62, 0.35, 0.8)], opcional=True, color=COLOR_BORDE_VOLUMEN)
        radio_giro = 0.25 * g.frente
        if b - a < 360 - EPS and self.grande and radio_giro * s >= 40:    # MÁS CHICO, EL LETRERO DEL GIRO CHOCA CON LAS COTAS
            ax.add_patch(Arc((0, 0), 2 * radio_giro, 2 * radio_giro, theta1=a, theta2=b, color=SUAVE, linewidth=1.2 * PT))
            self._letrero(f"giro de la base {b - a:.0f}°", [((radio_giro * np.cos(medio), radio_giro * np.sin(medio)), (0, 4), "data",
                                                           "center", "bottom")], opcional=True, color=SUAVE)

    def _colocar_robot(self):
        self._poner_puntos(self.geo.superior(self.servos))
