"""Programas del robot: lo que se guarda para repetir después. Puro (sin Qt).

Dos tipos:
- "puntos":    lista de puntos; cada uno con sus ángulos de servo, cómo llegar (articular / lineal)
               y su velocidad (% de la máxima segura).
- "grabacion": muestras (tiempo en s, ángulos de servo) tal como Paul movió el robot.

Se guardan ángulos de SERVO, no XYZ: dos posturas distintas pueden dar la misma XYZ, y así la
postura queda sin ambigüedad. Archivos YAML en programas/ (ver PROGRAMAS_POR_OMISION).
"""
import os
import re
import unicodedata
from dataclasses import dataclass, field

import yaml

from cosechador.robot.cinematica import servos_en_rango

FORMATO = 1
TIPOS = ("puntos", "grabacion")
MOVIMIENTOS = ("articular", "lineal")
VELOCIDAD_MIN, VELOCIDAD_MAX = 10, 100
CARPETA_PROGRAMAS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "programas")


class ErrorPrograma(ValueError):
    """El archivo del programa tiene un campo faltante o inválido."""


@dataclass(frozen=True)
class Punto:
    servos: tuple
    movimiento: str = "articular"
    velocidad: int = 100


@dataclass(frozen=True)
class Muestra:
    t: float          # segundos desde que empezó la grabación
    servos: tuple


@dataclass(frozen=True)
class Programa:
    nombre: str
    tipo: str
    robot: str
    gdl: int
    puntos: tuple = field(default=())
    muestras: tuple = field(default=())

    @property
    def duracion(self):
        """Segundos que dura una grabación tal como se grabó (0 para un programa por puntos)."""
        return self.muestras[-1].t - self.muestras[0].t if self.muestras else 0.0


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% VALIDACIÓN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def _servos(valor, donde, robot):
    if (not isinstance(valor, (list, tuple)) or len(valor) != robot.gdl
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in valor)):
        raise ErrorPrograma(f"{donde}: 'servos' debe ser una lista de {robot.gdl} números (llegó {valor!r}).")
    servos = tuple(float(v) for v in valor)
    if not servos_en_rango(robot, servos):
        rangos = ", ".join(f"{j.servo.minimo:g}–{j.servo.maximo:g}" for j in robot.articulaciones)
        raise ErrorPrograma(f"{donde}: los servos {servos} salen del rango del robot ({rangos}).")
    return servos


def programa_desde_dict(datos, robot):
    """Construye y valida un Programa para este robot."""
    if not isinstance(datos, dict):
        raise ErrorPrograma("El archivo debe ser una sección con nombre, tipo, robot, gdl y puntos o muestras.")
    if datos.get("formato") != FORMATO:
        raise ErrorPrograma(f"Formato de programa desconocido: {datos.get('formato')!r} (se espera {FORMATO}).")
    tipo = datos.get("tipo")
    if tipo not in TIPOS:
        raise ErrorPrograma(f"'tipo' debe ser uno de {TIPOS} (llegó {tipo!r}).")
    gdl = datos.get("gdl")
    if gdl != robot.gdl:
        raise ErrorPrograma(f"El programa es para un robot de {gdl} articulaciones y este tiene {robot.gdl}.")
    nombre = str(datos.get("nombre") or "sin nombre")

    if tipo == "puntos":
        crudos = datos.get("puntos")
        if not isinstance(crudos, list) or not crudos:
            raise ErrorPrograma("Un programa por puntos necesita al menos un punto en 'puntos'.")
        puntos = []
        for i, p in enumerate(crudos, start=1):
            donde = f"punto {i}"
            if not isinstance(p, dict):
                raise ErrorPrograma(f"{donde}: debe ser una sección con servos, movimiento y velocidad.")
            movimiento = p.get("movimiento", "articular")
            if movimiento not in MOVIMIENTOS:
                raise ErrorPrograma(f"{donde}: 'movimiento' debe ser uno de {MOVIMIENTOS} (llegó {movimiento!r}).")
            velocidad = p.get("velocidad", 100)
            if isinstance(velocidad, bool) or not isinstance(velocidad, int) or not VELOCIDAD_MIN <= velocidad <= VELOCIDAD_MAX:
                raise ErrorPrograma(f"{donde}: 'velocidad' debe ser un entero de {VELOCIDAD_MIN} a {VELOCIDAD_MAX} (llegó {velocidad!r}).")
            puntos.append(Punto(_servos(p.get("servos"), donde, robot), movimiento, velocidad))
        return Programa(nombre, tipo, str(datos.get("robot", "")), gdl, puntos=tuple(puntos))

    crudas = datos.get("muestras")
    if not isinstance(crudas, list) or len(crudas) < 2:
        raise ErrorPrograma("Una grabación necesita al menos dos muestras en 'muestras'.")
    muestras, anterior = [], None
    for i, m in enumerate(crudas, start=1):
        donde = f"muestra {i}"
        t = m.get("t") if isinstance(m, dict) else None
        if isinstance(t, bool) or not isinstance(t, (int, float)) or t < 0:
            raise ErrorPrograma(f"{donde}: 't' debe ser un número de segundos ≥ 0 (llegó {t!r}).")
        if anterior is not None and t < anterior:
            raise ErrorPrograma(f"{donde}: el tiempo va hacia atrás ({t} s después de {anterior} s).")
        muestras.append(Muestra(float(t), _servos(m.get("servos"), donde, robot)))
        anterior = t
    return Programa(nombre, tipo, str(datos.get("robot", "")), gdl, muestras=tuple(muestras))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ARCHIVOS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class _Lista(list):
    """Lista que se escribe en una sola línea en el YAML (los servos de cada punto)."""


class _Volcador(yaml.SafeDumper):
    pass


_Volcador.add_representer(_Lista, lambda d, v: d.represent_sequence("tag:yaml.org,2002:seq", v, flow_style=True))


def programa_a_dict(programa):
    datos = {"formato": FORMATO, "nombre": programa.nombre, "tipo": programa.tipo,
             "robot": programa.robot, "gdl": programa.gdl}
    if programa.tipo == "puntos":
        datos["puntos"] = [{"servos": _Lista(round(s, 6) for s in p.servos), "movimiento": p.movimiento,
                            "velocidad": p.velocidad} for p in programa.puntos]
    else:
        datos["muestras"] = [{"t": round(m.t, 4), "servos": _Lista(round(s, 6) for s in m.servos)}
                             for m in programa.muestras]
    return datos


def guardar(programa, ruta):
    with open(ruta, "w", encoding="utf-8") as f:
        yaml.dump(programa_a_dict(programa), f, Dumper=_Volcador, allow_unicode=True, sort_keys=False)


def cargar(ruta, robot):
    """Lee y valida un programa. Lanza ErrorPrograma con el nombre del archivo y el motivo."""
    try:
        with open(ruta, encoding="utf-8") as f:
            datos = yaml.safe_load(f)
    except FileNotFoundError:
        raise ErrorPrograma(f"No existe el programa: {ruta}") from None
    except yaml.YAMLError as e:
        raise ErrorPrograma(f"{os.path.basename(ruta)} no es YAML válido: {e}") from None
    try:
        return programa_desde_dict(datos, robot)
    except ErrorPrograma as e:
        raise ErrorPrograma(f"{os.path.basename(ruta)}: {e}") from None


def nombre_de_archivo(nombre):
    """El archivo para el nombre que escribe la persona: «Mi grabación 2» → «mi-grabacion-2.yaml».

    Solo letras sin acento, números y guiones: el archivo nunca sale de programas/ ni lleva caracteres que
    algún sistema no acepte. El nombre tal cual se escribió va DENTRO del archivo (campo «nombre»).
    """
    base = re.sub(r"\.ya?ml\s*$", "", nombre.strip(), flags=re.IGNORECASE)
    base = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode("ascii")
    base = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")[:60].strip("-")
    if not base:
        raise ErrorPrograma("El nombre necesita al menos una letra o un número.")
    return base + ".yaml"


def listar(carpeta=CARPETA_PROGRAMAS):
    """Nombres de archivo de los programas de la carpeta, en orden alfabético."""
    if not os.path.isdir(carpeta):
        return []
    return sorted(n for n in os.listdir(carpeta) if n.endswith((".yaml", ".yml")) and not n.startswith("."))
