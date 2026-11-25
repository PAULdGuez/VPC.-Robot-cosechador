"""El botón CALCULAR con lo que la persona escribe en X, Y y Z.

D5 (aprobado el 25-nov-2026): si en X, Y o Z se escribe algo que no es un número normal y se pulsa CALCULAR, sale el aviso
de siempre, el robot no se mueve y nada revienta. Lo encontró el explorador al azar: «inf» y «nan» son números para
float(), y pasaban el filtro hasta reventar más adentro.
"""
import pytest

from cosechador.robot import cargar_robot
from helpers_ventana import borrar_desechable, carpeta_desechable, cerrar, correr, sin_dialogos_de_verdad, ventana_aislada

ROBOT = cargar_robot()
AVISO = "Escribe números en X, Y y Z (en mm)."


@pytest.fixture
def ventana():
    carpeta = carpeta_desechable()
    w = ventana_aislada(ROBOT, carpeta)
    yield w
    cerrar(w)
    borrar_desechable(carpeta)


def _calcular(w, x, y, z):
    for campo, texto in zip((w.TE_X, w.TE_Y, w.TE_Z), (x, y, z)):
        campo.setText(texto)
    with sin_dialogos_de_verdad():
        w.CALCULAR.click()
        reventado = list(w.x.excepciones)
        correr(w)
    return reventado


@pytest.mark.parametrize("escrito", [("inf", "0", "0"), ("100", "nan", "500"), ("-inf", "inf", "nan"), ("0", "0", "Infinity"),
                                     ("1e999", "0", "0"), ("NaN", "NaN", "NaN"), ("hola", "0", "0"), ("", "", "")])
def test_d5_lo_que_no_es_un_numero_normal_da_el_aviso_y_el_robot_no_se_mueve(ventana, escrito):
    w = ventana
    reventado = _calcular(w, *escrito)
    assert not reventado, f"CALCULAR reventó con {escrito}: {reventado[0][1]!r}"
    assert [e.mensaje for e in w.x.eventos if e.origen == "Movimiento"][-1:] == [AVISO], [e.mensaje for e in w.x.eventos][-3:]
    assert w.x.enviados == [], f"el robot se movió con {escrito}"


def test_d5_un_punto_normal_se_sigue_calculando(ventana):
    """El control: el filtro no se lleva lo que sí es un punto (uno alcanzable mueve el robot y no da el aviso)."""
    w = ventana
    reventado = _calcular(w, "300", "0", "900")
    assert not reventado and AVISO not in [e.mensaje for e in w.x.eventos]
    assert len(w.x.enviados) > 0, [e.mensaje for e in w.x.eventos][-3:]
