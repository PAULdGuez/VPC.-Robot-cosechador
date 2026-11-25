"""Detector de rachas de avisos idénticos en el registro (no es una prueba; lo usan el explorador y sus pruebas).

Una RACHA son avisos o errores SEGUIDOS (ningún otro evento en medio) del mismo nivel, del mismo origen y del mismo
mensaje, sin contar los números: «No se mueve. Para pasar por (155, 0, 1654)…» y «…por (156, 0, 1654)…» son el mismo
aviso. Es lo que el 25-nov dejó escrito en el registro: 31 negativas seguidas en 21 s, con el slider rebotando cada vez.
Lee los dos tipos de registro: los eventos de la ventana en memoria y el archivo .jsonl de un día.
"""
import re
from typing import NamedTuple

from cosechador import registro as reg

MINIMO_DE_UNA_RACHA = 3                 # DOS AVISOS SEGUIDOS IGUALES SE PUEDEN DAR (UN CLIC REPETIDO); TRES YA SON UNA RACHA
NIVELES_DE_AVISO = ("aviso", "error")
_NUMERO = re.compile(r"-?\d+(?:[.,]\d+)?")


class Racha(NamedTuple):
    longitud: int
    nivel: str
    origen: str
    mensaje: str            # EL DEL PRIMER AVISO, TAL CUAL
    desde: object           # HORA DEL PRIMERO (datetime)
    hasta: object           # HORA DEL ÚLTIMO
    primero: int            # POSICIÓN DEL PRIMERO EN LA LISTA QUE SE REVISÓ (desde 0)

    @property
    def segundos(self):
        return (self.hasta - self.desde).total_seconds()


def forma_del_aviso(mensaje):
    """El mensaje sin sus números: lo que hace «el mismo aviso» a dos avisos."""
    return _NUMERO.sub("#", mensaje)


def rachas_de_avisos(eventos, minimo=MINIMO_DE_UNA_RACHA):
    """Todas las rachas de `minimo` avisos seguidos o más, en orden. `eventos`: lista de registro.Evento."""
    rachas, i = [], 0
    clave = lambda e: (e.nivel, e.origen, forma_del_aviso(e.mensaje))
    while i < len(eventos):
        j = i
        if eventos[i].nivel in NIVELES_DE_AVISO:
            while j + 1 < len(eventos) and clave(eventos[j + 1]) == clave(eventos[i]):
                j += 1
            if j - i + 1 >= minimo:
                e = eventos[i]
                rachas.append(Racha(j - i + 1, e.nivel, e.origen, e.mensaje, e.hora, eventos[j].hora, i))
        i = j + 1
    return rachas


def rachas_del_archivo(ruta, minimo=MINIMO_DE_UNA_RACHA):
    """Lo mismo, leyendo un archivo de registro (solo lectura)."""
    return rachas_de_avisos(reg.leer_jsonl(ruta), minimo)
