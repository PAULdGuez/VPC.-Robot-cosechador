"""Descripción de un robot leída de un archivo YAML (ver robots/cosechador_3gdl.yaml).

Este módulo no sabe nada de un robot en particular: solo lee, valida y entrega datos inmutables.
Si el archivo trae un error, ErrorConfiguracion dice en qué campo está y por qué.
"""
import difflib
import os
from dataclasses import dataclass, field, replace

import numpy as np
import yaml

from .cinematica import cinematica_directa, servo_a_dh
from .cinematica_inversa import angulos_servo, motivo_de_no_alcanzar

CARPETA_ROBOTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "robots")
ROBOT_POR_OMISION = os.path.join(CARPETA_ROBOTS, "cosechador_3gdl.yaml")

SOLVERS_CONOCIDOS = ("antropomorfico_3r",)
MONTAJES_CAMARA = ("efector", "fija")
CAMBIOS_DE_POSTURA = ("despacio", "negar")

# LAS LLAVES QUE SE LEEN DE CADA SECCIÓN DEL ARCHIVO, EN EL ORDEN EN QUE SE ESCRIBEN. UNA QUE NO ESTÉ AQUÍ NO SE IGNORA:
# CASI SIEMPRE ES UNA DE ESTAS MAL ESCRITA, Y EL ROBOT ABRIRÍA CON EL VALOR POR OMISIÓN SIN QUE NADIE SE ENTERE
LLAVES = {
    "robot": ("nombre", "cinematica_inversa", "home_mm", "base", "articulaciones", "efector", "camara", "vision",
              "movimiento", "puntos_demo_mm"),
    "base": ("pieza", "color"),
    "articulacion": ("nombre", "dh", "servo", "pieza", "color"),
    "dh": ("d", "a", "alpha"),
    "servo": ("pin", "min", "max", "offset", "signo"),
    "efector": ("pieza", "color", "largo"),
    "camara": ("montaje", "matriz_montaje", "fx_referencia", "resolucion_referencia", "diametro_objeto_mm"),
    "vision": ("verde", "cafe"),
    "rango_hsv": ("min", "max"),
    "movimiento": ("ms_por_paso", "salto_max_por_paso", "al_cambiar_postura"),
}


class ErrorConfiguracion(ValueError):
    """El archivo del robot tiene un campo faltante o inválido."""


@dataclass(frozen=True)
class Servo:
    pin: int
    minimo: float
    maximo: float
    offset: float = 0.0
    signo: int = 1

    @property
    def centro(self):
        return (self.minimo + self.maximo) / 2


@dataclass(frozen=True)
class Pieza:
    archivo: str | None
    color: str = "#f09a3e"


@dataclass(frozen=True)
class Efector(Pieza):
    """La herramienta: prolonga el último eslabón `largo` mm, alineada con él. 0 = sin herramienta."""
    largo: float = 0.0


@dataclass(frozen=True)
class Articulacion:
    nombre: str
    d: float
    a: float
    alpha: float
    servo: Servo
    pieza: Pieza


@dataclass(frozen=True)
class Camara:
    montaje: str
    matriz_montaje: np.ndarray = field(repr=False)
    fx_referencia: float
    resolucion_referencia: tuple
    diametro_objeto_mm: float


@dataclass(frozen=True)
class RangoHSV:
    minimo: tuple
    maximo: tuple


@dataclass(frozen=True)
class Vision:
    verde: RangoHSV
    cafe: RangoHSV


@dataclass(frozen=True)
class Movimiento:
    """Cómo se mueve el robot. Valores por omisión pensados para servos SG90."""
    ms_por_paso: float = 50.0
    salto_max_por_paso: float = 15.0      # grados que un servo puede girar en un paso
    al_cambiar_postura: str = "despacio"   # despacio | negar


@dataclass(frozen=True)
class Robot:
    nombre: str
    articulaciones: tuple
    home: tuple                    # DÓNDE QUEDA LA PUNTA EN LA POSTURA DE HOME (con_herramienta lo calcula)
    punto_de_giro: tuple           # EL home_mm DEL ARCHIVO: SOBRE EL EJE DE LA BASE, DONDE GIRARLA NO MUEVE LA PUNTA
    base: Pieza
    efector: Efector
    camara: Camara
    vision: Vision
    cinematica_inversa: str | None = None
    puntos_demo: tuple = ()
    carpeta_modelos: str = ""
    movimiento: Movimiento = Movimiento()
    postura_home: tuple | None = None  # SERVOS DE HOME: LOS QUE LLEVAN EL BRAZO SIN HERRAMIENTA AL PUNTO DE GIRO

    @property
    def gdl(self):
        return len(self.articulaciones)

    @property
    def alcance(self):
        """Distancia máxima aproximada de la base al extremo del último eslabón (para escalar el dibujo)."""
        return sum(abs(j.a) + abs(j.d) for j in self.articulaciones)

    @property
    def alcance_de_la_punta(self):
        """Lo más lejos de la base que puede quedar la punta de la herramienta (hasta dónde se pide X, Y, Z)."""
        return self.alcance + self.efector.largo


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LECTURA Y VALIDACIÓN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def _solo_llaves_conocidas(datos, seccion, donde):
    """Rechaza la primera llave de `datos` que la app no lee en esa sección, diciendo cuál es y cuáles sí se aceptan.

    `Largo: 120` en lugar de `largo: 120` se ignoraba y el robot abría con herramienta 0, sin avisar. Si la llave se
    parece a una conocida (otra mayúscula, un error de dedo), se sugiere. Que `datos` no sea una sección lo dice quien la lee.
    """
    if not isinstance(datos, dict):
        return
    conocidas = LLAVES[seccion]
    for llave in datos:
        if llave in conocidas:
            continue
        escrita = str(llave)
        parecidas = difflib.get_close_matches(escrita.lower(), conocidas, n=1)
        pista = f" ¿Será '{parecidas[0]}'?" if parecidas else ""
        raise ErrorConfiguracion(f"No se reconoce '{escrita}' en {donde}.{pista} Ahí se aceptan: {', '.join(conocidas)}.")


def _campo(datos, clave, donde, tipo=None):
    if not isinstance(datos, dict) or clave not in datos:
        raise ErrorConfiguracion(f"Falta el campo '{clave}' en {donde}.")
    valor = datos[clave]
    if tipo is not None and not isinstance(valor, tipo):
        raise ErrorConfiguracion(f"El campo '{clave}' en {donde} debe ser {_nombre_tipo(tipo)} (llegó {valor!r}).")
    return valor


def _nombre_tipo(tipo):
    nombres = {(int, float): "un número", int: "un entero", str: "un texto", list: "una lista", dict: "una sección"}
    return nombres.get(tipo, str(tipo))


def _numero(datos, clave, donde, por_omision=None):
    if por_omision is not None and clave not in datos:
        return float(por_omision)
    valor = _campo(datos, clave, donde)
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        raise ErrorConfiguracion(f"El campo '{clave}' en {donde} debe ser un número (llegó {valor!r}).")
    return float(valor)


def _vector(datos, clave, donde, largo):
    valor = _campo(datos, clave, donde, list)
    if len(valor) != largo or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in valor):
        raise ErrorConfiguracion(f"El campo '{clave}' en {donde} debe ser una lista de {largo} números (llegó {valor!r}).")
    return tuple(float(v) for v in valor)


def _pieza(datos, donde, clase=Pieza):
    archivo = datos.get("pieza")
    if archivo is not None and not isinstance(archivo, str):
        raise ErrorConfiguracion(f"El campo 'pieza' en {donde} debe ser un nombre de archivo (llegó {archivo!r}).")
    color = datos.get("color", "#f09a3e")
    return clase(archivo=archivo, color=str(color))


def con_herramienta(robot, largo):
    """El mismo robot con otra herramienta: su punta queda `largo` mm más allá del extremo del último eslabón.

    Es el único lugar que decide qué largo vale y qué es HOME con él. El largo: un número de milímetros entre 0 y el
    alcance del brazo (una herramienta más larga que el brazo entero es un error de dedo). HOME es una POSTURA: la que
    lleva al brazo SIN herramienta al punto de giro (`home_mm` del archivo), de modo que el brazo y la cámara quedan en
    HOME donde estaban antes de tener herramienta; `home` es dónde queda la punta en esa postura. Si con la herramienta
    la punta ya no alcanza el punto de giro (dónde se cruza de un lado al otro), el archivo no es válido.
    """
    if isinstance(largo, bool) or not isinstance(largo, (int, float)) or not 0 <= largo <= robot.alcance:
        raise ErrorConfiguracion(f"'efector.largo' debe ser un número de milímetros entre 0 y {robot.alcance:g}, "
                                 f"el alcance del brazo (llegó {largo!r}).")
    sin = replace(robot, efector=replace(robot.efector, largo=0.0))
    con = replace(robot, efector=replace(robot.efector, largo=float(largo)))
    postura = angulos_servo(sin, robot.punto_de_giro) if robot.cinematica_inversa is not None else None
    if postura is None:  # SIN SOLVER, O NI SIN HERRAMIENTA SE ALCANZA EL PUNTO (YA ERA ASÍ): NO HAY POSTURA DE HOME
        return replace(con, home=robot.punto_de_giro, postura_home=None)
    motivo = motivo_de_no_alcanzar(con, robot.punto_de_giro)
    if motivo is not None:
        punto = ", ".join(f"{c:g}" for c in robot.punto_de_giro)
        raise ErrorConfiguracion(f"Con una herramienta de {con.efector.largo:g} mm la punta no puede llegar al punto de giro "
                                 f"({punto}) que el archivo pone sobre el eje de la base: {motivo}. "
                                 "Acorta 'efector.largo' o cambia 'home_mm'.")
    home = tuple(float(c) for c in cinematica_directa(con, servo_a_dh(con, postura)))
    return replace(con, home=home, postura_home=tuple(postura))


def _servo(datos, donde):
    _solo_llaves_conocidas(datos, "servo", donde)
    pin = _campo(datos, "pin", donde)
    if isinstance(pin, bool) or not isinstance(pin, int):
        raise ErrorConfiguracion(f"El campo 'pin' en {donde} debe ser un entero (llegó {pin!r}).")
    minimo, maximo = _numero(datos, "min", donde), _numero(datos, "max", donde)
    if minimo >= maximo:
        raise ErrorConfiguracion(f"En {donde}, 'min' ({minimo:g}) debe ser menor que 'max' ({maximo:g}).")
    if maximo - minimo > 360:
        raise ErrorConfiguracion(f"En {donde}, el rango del servo no puede pasar de 360° (min {minimo:g}, max {maximo:g}).")
    signo = datos.get("signo", 1)
    if signo not in (1, -1):
        raise ErrorConfiguracion(f"El campo 'signo' en {donde} debe ser 1 o -1 (llegó {signo!r}).")
    return Servo(pin=pin, minimo=minimo, maximo=maximo, offset=_numero(datos, "offset", donde, 0), signo=signo)


def _articulacion(datos, i):
    donde = f"articulaciones[{i}]"
    if not isinstance(datos, dict):
        raise ErrorConfiguracion(f"{donde} debe ser una sección con dh, servo y pieza.")
    _solo_llaves_conocidas(datos, "articulacion", donde)
    dh = _campo(datos, "dh", donde, dict)
    _solo_llaves_conocidas(dh, "dh", f"{donde}.dh")
    return Articulacion(
        nombre=str(datos.get("nombre", f"Articulación {i + 1}")),
        d=_numero(dh, "d", f"{donde}.dh"),
        a=_numero(dh, "a", f"{donde}.dh"),
        alpha=_numero(dh, "alpha", f"{donde}.dh"),
        servo=_servo(_campo(datos, "servo", donde, dict), f"{donde}.servo"),
        pieza=_pieza(datos, donde),
    )


def _camara(datos):
    donde = "camara"
    _solo_llaves_conocidas(datos, "camara", donde)
    montaje = _campo(datos, "montaje", donde, str)
    if montaje not in MONTAJES_CAMARA:
        raise ErrorConfiguracion(f"'camara.montaje' debe ser uno de {MONTAJES_CAMARA} (llegó {montaje!r}).")
    matriz = np.array(_campo(datos, "matriz_montaje", donde, list), dtype=float)
    if matriz.shape != (4, 4):
        raise ErrorConfiguracion(f"'camara.matriz_montaje' debe ser una matriz de 4x4 (llegó de {matriz.shape}).")
    rotacion = matriz[:3, :3]
    if not np.allclose(rotacion @ rotacion.T, np.eye(3), atol=1e-6) or np.linalg.det(rotacion) < 0:
        raise ErrorConfiguracion("'camara.matriz_montaje' no es una rotación válida (sus columnas deben ser ejes unitarios perpendiculares).")
    diametro = _numero(datos, "diametro_objeto_mm", donde)
    fx = _numero(datos, "fx_referencia", donde)
    if diametro <= 0 or fx <= 0:
        raise ErrorConfiguracion("'camara.diametro_objeto_mm' y 'camara.fx_referencia' deben ser mayores que 0.")
    return Camara(montaje=montaje, matriz_montaje=matriz, fx_referencia=fx,
                  resolucion_referencia=_vector(datos, "resolucion_referencia", donde, 2),
                  diametro_objeto_mm=diametro)


def _movimiento(datos):
    if datos is None:
        return Movimiento()
    if not isinstance(datos, dict):
        raise ErrorConfiguracion("'movimiento' debe ser una sección (ms_por_paso, salto_max_por_paso, al_cambiar_postura).")
    _solo_llaves_conocidas(datos, "movimiento", "movimiento")
    por_omision = Movimiento()
    ms = _numero(datos, "ms_por_paso", "movimiento", por_omision.ms_por_paso)
    salto = _numero(datos, "salto_max_por_paso", "movimiento", por_omision.salto_max_por_paso)
    if ms <= 0 or salto <= 0:
        raise ErrorConfiguracion("'movimiento.ms_por_paso' y 'movimiento.salto_max_por_paso' deben ser mayores que 0.")
    postura = datos.get("al_cambiar_postura", por_omision.al_cambiar_postura)
    if postura not in CAMBIOS_DE_POSTURA:
        raise ErrorConfiguracion(f"'movimiento.al_cambiar_postura' debe ser uno de {CAMBIOS_DE_POSTURA} (llegó {postura!r}).")
    return Movimiento(ms_por_paso=ms, salto_max_por_paso=salto, al_cambiar_postura=postura)


def _rango_hsv(datos, donde):
    _solo_llaves_conocidas(datos, "rango_hsv", donde)
    return RangoHSV(minimo=_vector(datos, "min", donde, 3), maximo=_vector(datos, "max", donde, 3))


def robot_desde_dict(datos, carpeta_modelos=""):
    """Construye y valida un Robot a partir de un diccionario (lo que devuelve el YAML)."""
    if not isinstance(datos, dict):
        raise ErrorConfiguracion("El archivo del robot debe ser una sección con nombre, articulaciones, etc.")
    _solo_llaves_conocidas(datos, "robot", "el robot")
    articulaciones = _campo(datos, "articulaciones", "el robot", list)
    if not articulaciones:
        raise ErrorConfiguracion("El robot debe tener al menos una articulación.")
    solver = datos.get("cinematica_inversa")
    if solver is not None and solver not in SOLVERS_CONOCIDOS:
        raise ErrorConfiguracion(f"'cinematica_inversa' debe ser uno de {SOLVERS_CONOCIDOS} o no ponerse (llegó {solver!r}).")
    vision = _campo(datos, "vision", "el robot", dict)
    efector = datos.get("efector", {})
    puntos = datos.get("puntos_demo_mm", [])
    for seccion, contenido in (("base", datos.get("base", {})), ("efector", efector), ("vision", vision)):
        _solo_llaves_conocidas(contenido, seccion, seccion)
    robot = Robot(
        nombre=str(_campo(datos, "nombre", "el robot")),
        articulaciones=tuple(_articulacion(j, i) for i, j in enumerate(articulaciones)),
        home=_vector(datos, "home_mm", "el robot", 3),  # PROVISIONAL: con_herramienta CALCULA EL DEFINITIVO
        punto_de_giro=_vector(datos, "home_mm", "el robot", 3),
        base=_pieza(datos.get("base", {}), "base"),
        efector=_pieza(efector, "efector", Efector),
        camara=_camara(_campo(datos, "camara", "el robot", dict)),
        vision=Vision(verde=_rango_hsv(_campo(vision, "verde", "vision", dict), "vision.verde"),
                      cafe=_rango_hsv(_campo(vision, "cafe", "vision", dict), "vision.cafe")),
        cinematica_inversa=solver,
        puntos_demo=tuple(_vector({"p": p}, "p", f"puntos_demo_mm[{i}]", 3) for i, p in enumerate(puntos)),
        carpeta_modelos=carpeta_modelos,
        movimiento=_movimiento(datos.get("movimiento")),
    )
    if solver == "antropomorfico_3r":
        _validar_topologia_3r(robot)
    return con_herramienta(robot, efector.get("largo", 0))


def _validar_topologia_3r(robot):
    """El solver analítico 3R solo vale para: base vertical + hombro + codo paralelos."""
    if robot.gdl != 3:
        raise ErrorConfiguracion(f"El solver 'antropomorfico_3r' necesita 3 articulaciones (hay {robot.gdl}).")
    j1, j2, j3 = robot.articulaciones
    if not (abs(j1.a) < 1e-9 and abs(j1.alpha - 90) < 1e-9 and abs(j2.d) < 1e-9 and abs(j2.alpha) < 1e-9
            and abs(j3.d) < 1e-9 and abs(j3.alpha) < 1e-9 and j2.a > 0 and j3.a > 0):
        raise ErrorConfiguracion(
            "El solver 'antropomorfico_3r' necesita: articulación 1 con a=0 y alpha=90; articulaciones 2 y 3 "
            "con d=0, alpha=0 y a>0. Revisa la tabla DH o quita 'cinematica_inversa'.")


def cargar_robot(ruta=None):
    """Lee y valida el YAML de un robot. Por omisión, ROBOT_CONFIG o robots/cosechador_3gdl.yaml."""
    ruta = ruta or os.environ.get("ROBOT_CONFIG") or ROBOT_POR_OMISION
    try:
        with open(ruta, encoding="utf-8") as f:
            datos = yaml.safe_load(f)
    except FileNotFoundError:
        raise ErrorConfiguracion(f"No existe el archivo del robot: {ruta}") from None
    except yaml.YAMLError as e:
        raise ErrorConfiguracion(f"El archivo {ruta} no es YAML válido: {e}") from None
    carpeta = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(ruta))), "modelos")
    try:
        return robot_desde_dict(datos, carpeta_modelos=carpeta)
    except ErrorConfiguracion as e:
        raise ErrorConfiguracion(f"{os.path.basename(ruta)}: {e}") from None
