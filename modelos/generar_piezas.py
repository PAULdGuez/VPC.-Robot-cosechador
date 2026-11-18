"""Genera piezas del robot por código (CAD con build123d) a partir del YAML del robot.

    .venv/bin/python modelos/generar_piezas.py brazo            # -> modelos/brazo-codigo.stl
    .venv/bin/python modelos/generar_piezas.py brazo --salida modelos/brazo.stl

Las medidas salen del YAML (robots/cosechador_3gdl.yaml): si cambias la tabla DH, la pieza se
regenera con la medida nueva. Se respeta la convención de modelos/README.md: origen en el eje de
giro (Z) y el eslabón a lo largo de +X. Necesita requirements-cad.txt (build123d).
"""
import argparse
import os
import sys

from build123d import (BuildPart, BuildSketch, Circle, Locations, Mode, Plane, SlotCenterToCenter,
                       export_stl, extrude)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cosechador.robot import cargar_robot  # noqa: E402

CARPETA = os.path.dirname(os.path.abspath(__file__))


def eslabon(largo, ancho=80.0, grueso=80.0, diametro_eje=20.0):
    """Barra con extremos redondos ("slot"): ejes de Ø diametro_eje en x = 0 y x = largo.

    Origen en el centro del primer eje; largo sobre +X; grueso centrado en Z (-grueso/2 a +grueso/2).
    """
    with BuildPart() as pieza:
        with BuildSketch(Plane.XY):
            with Locations((largo / 2, 0)):
                SlotCenterToCenter(largo, ancho)
            with Locations((0, 0), (largo, 0)):
                Circle(diametro_eje / 2, mode=Mode.SUBTRACT)
        extrude(amount=grueso / 2, both=True)
    return pieza.part


def brazo(robot):
    """El brazo (articulación 2, del hombro al codo) con el largo `a` de la tabla DH."""
    return eslabon(robot.articulaciones[1].a)


PIEZAS = {"brazo": brazo}


def main(argv=None):
    opciones = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    opciones.add_argument("pieza", choices=sorted(PIEZAS))
    opciones.add_argument("--robot", help="YAML del robot (por omisión, el cosechador)")
    opciones.add_argument("--salida", help="ruta del STL (por omisión modelos/<pieza>-codigo.stl)")
    args = opciones.parse_args(argv)

    robot = cargar_robot(args.robot)
    pieza = PIEZAS[args.pieza](robot)
    salida = args.salida or os.path.join(CARPETA, f"{args.pieza}-codigo.stl")
    export_stl(pieza, salida)
    caja = pieza.bounding_box()
    print(f"{salida}\n  volumen {pieza.volume:,.0f} mm³ · caja X [{caja.min.X:.1f}, {caja.max.X:.1f}] "
          f"Y [{caja.min.Y:.1f}, {caja.max.Y:.1f}] Z [{caja.min.Z:.1f}, {caja.max.Z:.1f}] mm")


if __name__ == "__main__":
    main()
