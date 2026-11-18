"""Piezas por código (build123d) y el comparador de STL. Los errores típicos se fabrican a propósito."""
import math
import os
import sys

import numpy as np
import pytest
import pyvista as pv

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "modelos"))

import comparar_piezas  # noqa: E402

build123d = pytest.importorskip("build123d", reason="requirements-cad.txt no instalado")
import generar_piezas  # noqa: E402


@pytest.fixture(scope="module")
def brazo_stl(robot, tmp_path_factory):
    ruta = tmp_path_factory.mktemp("piezas") / "brazo-codigo.stl"
    generar_piezas.export_stl(generar_piezas.brazo(robot), str(ruta))
    return ruta


def guardar(malla, carpeta, nombre):
    ruta = carpeta / nombre
    malla.save(ruta)
    return str(ruta)


def test_el_brazo_sale_de_la_tabla_dh(robot, brazo_stl):
    _, caja, volumen = comparar_piezas.medir(str(brazo_stl))
    a = robot.articulaciones[1].a
    assert caja[0] == pytest.approx((-40, a + 40), abs=0.5)    # DEL HOMBRO AL CODO SOBRE +X
    assert caja[1] == pytest.approx((-40, 40), abs=0.5)
    assert caja[2] == pytest.approx((-40, 40), abs=0.5)        # GROSOR CENTRADO EN Z
    esperado = (a * 80 + math.pi * 40 ** 2 - 2 * math.pi * 10 ** 2) * 80
    assert volumen == pytest.approx(esperado, rel=0.01)


def test_iguales_coinciden(brazo_stl, tmp_path):
    copia = guardar(pv.read(brazo_stl), tmp_path, "copia.stl")
    assert comparar_piezas.main([copia, str(brazo_stl)]) == 0


@pytest.mark.parametrize("error, transformar, diagnostico", [
    ("origen en el centro de la barra", lambda m: m.translate((-500, 0, 0)), "ORIGEN"),
    ("largo sobre Y en vez de X", lambda m: m.rotate_z(90, point=(0, 0, 0)), "GIRADA"),
    ("exportada en pulgadas", lambda m: m.scale(1 / 25.4), "pulgadas"),
])
def test_cada_error_tipico_se_diagnostica(brazo_stl, tmp_path, capsys, error, transformar, diagnostico):
    mala = guardar(transformar(pv.read(brazo_stl)), tmp_path, "mala.stl")
    assert comparar_piezas.main([mala, str(brazo_stl)]) == 1, error
    assert diagnostico in capsys.readouterr().out


def test_misma_caja_distinto_volumen(brazo_stl, tmp_path, capsys):
    # SIN AGUJEROS: MISMA CAJA, MÁS VOLUMEN
    with build123d.BuildPart() as sin_agujeros:
        with build123d.BuildSketch():
            with build123d.Locations((500, 0)):
                build123d.SlotCenterToCenter(1000, 80)
        build123d.extrude(amount=40, both=True)
    ruta = str(tmp_path / "sin_agujeros.stl")
    build123d.export_stl(sin_agujeros.part, ruta)
    assert comparar_piezas.main([ruta, str(brazo_stl)]) == 1
    assert "volumen distinto" in capsys.readouterr().out
    assert np.isfinite(comparar_piezas.medir(ruta)[2])
