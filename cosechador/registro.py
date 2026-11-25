"""Registro estructurado de lo que pasa en el robot (puro: sin Qt).

Cada evento tiene hora, nivel, origen, mensaje y datos aparte (no texto corrido), y se escribe en un
archivo JSON Lines: un evento por línea, que se abre con pandas, Excel o grep. Ejemplo:

    {"hora": "2026-11-18T17:21:03.120", "nivel": "aviso", "origen": "Jog",
     "mensaje": "No se mueve: zona inalcanzable", "datos": {"destino": [1355, 117, 1398]}}
"""
import datetime as dt
import json
import os
from dataclasses import dataclass, field

NIVELES = ("info", "ok", "aviso", "error")
CARPETA_LOGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
MAXIMO_EN_MEMORIA = 2000  # LA LISTA DE LA VENTANA NO CRECE SIN FIN; EL ARCHIVO GUARDA TODO


def _a_json(valor):
    """Convierte tuplas, numpy y flotantes largos a algo que JSON entiende y se lee bien."""
    if isinstance(valor, dict):
        return {str(k): _a_json(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_a_json(v) for v in valor]
    if hasattr(valor, "tolist"):          # numpy
        return _a_json(valor.tolist())
    if isinstance(valor, float):
        return round(valor, 3) + 0.0  # (+ 0.0 CONVIERTE -0.0 EN 0.0)
    if isinstance(valor, (int, str, bool)) or valor is None:
        return valor
    return str(valor)


@dataclass(frozen=True)
class Evento:
    hora: dt.datetime
    nivel: str
    origen: str
    mensaje: str
    datos: dict = field(default_factory=dict)

    def a_dict(self):
        return {"hora": self.hora.isoformat(timespec="milliseconds"), "nivel": self.nivel,
                "origen": self.origen, "mensaje": self.mensaje, "datos": _a_json(self.datos)}

    def a_jsonl(self):
        return json.dumps(self.a_dict(), ensure_ascii=False)

    @classmethod
    def desde_dict(cls, d):
        return cls(dt.datetime.fromisoformat(d["hora"]), d["nivel"], d["origen"], d["mensaje"], d.get("datos") or {})


class Registro:
    """Guarda los eventos en memoria, avisa a quien se suscriba y los escribe en el archivo del día."""

    def __init__(self, carpeta=CARPETA_LOGS, reloj=dt.datetime.now):
        self.carpeta = carpeta       # None = no escribir archivo (pruebas)
        self.reloj = reloj
        self.eventos = []
        self._suscriptores = []

    @property
    def archivo(self):
        if self.carpeta is None:
            return None
        return os.path.join(self.carpeta, f"registro-{self.reloj():%Y-%m-%d}.jsonl")

    def suscribir(self, funcion):
        self._suscriptores.append(funcion)

    def registrar(self, nivel, origen, mensaje, **datos):
        if nivel not in NIVELES:
            raise ValueError(f"Nivel desconocido: {nivel!r} (se espera uno de {NIVELES}).")
        evento = Evento(self.reloj(), nivel, origen, mensaje, datos)
        self.eventos.append(evento)
        del self.eventos[:-MAXIMO_EN_MEMORIA]
        if self.carpeta is not None:
            os.makedirs(self.carpeta, exist_ok=True)
            with open(self.archivo, "a", encoding="utf-8") as f:
                f.write(evento.a_jsonl() + "\n")
        for funcion in self._suscriptores:
            funcion(evento)
        return evento

    def filtrar(self, niveles=NIVELES):
        return [e for e in self.eventos if e.nivel in niveles]


def leer_jsonl(ruta):
    """Lee un archivo de registro (para análisis o pruebas)."""
    with open(ruta, encoding="utf-8") as f:
        return [Evento.desde_dict(json.loads(linea)) for linea in f if linea.strip()]


def exportar_jsonl(eventos, ruta):
    with open(ruta, "w", encoding="utf-8") as f:
        for e in eventos:
            f.write(e.a_jsonl() + "\n")
