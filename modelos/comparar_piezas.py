"""Compara dos STL de la misma pieza (p. ej. la hecha a mano en Onshape contra la generada por código).

    .venv/bin/python modelos/comparar_piezas.py modelos/brazo-mano.stl modelos/brazo-codigo.stl

Mide la caja (dónde empieza y acaba la pieza en cada eje), el volumen y si las dos ocupan el mismo
espacio. Un desfase en la caja casi siempre es el ORIGEN mal puesto; una caja girada, los EJES; un
volumen 25.4 o 16387 veces distinto, las UNIDADES (pulgadas en vez de mm).
"""
import argparse
import sys

import numpy as np
import pyvista as pv

TOLERANCIA_MM = 1.0
TOLERANCIA_VOLUMEN = 0.003  # 0.3 %: los dos agujeros de Ø20 cambian el volumen 0.74 %; el triangulado, ~0.002 %


def medir(ruta):
    malla = pv.read(ruta).triangulate()
    caja = np.array(malla.bounds).reshape(3, 2)  # [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
    return malla, caja, malla.volume


def diagnostico(caja_a, caja_b, vol_a, vol_b):
    """Lista de problemas en lenguaje claro (vacía si coinciden)."""
    problemas = []
    razon = vol_a / vol_b if vol_b else float("inf")
    for factor, unidad in ((25.4 ** 3, "pulgadas"), (10 ** 3, "centímetros")):
        if abs(razon / factor - 1) < 0.05 or abs(razon * factor - 1) < 0.05:
            problemas.append(f"El volumen difiere ~{factor:,.0f} veces: una de las dos se exportó en {unidad}, no en mm.")
            return problemas
    tam_a, tam_b = caja_a[:, 1] - caja_a[:, 0], caja_b[:, 1] - caja_b[:, 0]
    if not np.allclose(np.sort(tam_a), np.sort(tam_b), atol=TOLERANCIA_MM):
        problemas.append(f"Tienen distinto tamaño: {np.round(tam_a, 1)} contra {np.round(tam_b, 1)} mm.")
    elif not np.allclose(tam_a, tam_b, atol=TOLERANCIA_MM):
        problemas.append("Miden lo mismo pero en ejes distintos: una está GIRADA (el largo debe ir sobre +X).")
    elif not np.allclose(caja_a, caja_b, atol=TOLERANCIA_MM):
        desfase = caja_a[:, 0] - caja_b[:, 0]
        problemas.append(f"Están desplazadas {np.round(desfase, 1)} mm: el ORIGEN no está en el mismo lugar "
                         "(debe ir en el centro del eje del hombro).")
    if not problemas and abs(vol_a - vol_b) / vol_b > TOLERANCIA_VOLUMEN:
        problemas.append(f"Misma caja pero volumen distinto ({vol_a:,.0f} contra {vol_b:,.0f} mm³): "
                         "cambian los agujeros, los redondeos o el grosor.")
    return problemas


def main(argv=None):
    opciones = argparse.ArgumentParser(description="Compara dos STL de la misma pieza")
    opciones.add_argument("a")
    opciones.add_argument("b")
    args = opciones.parse_args(argv)

    _, caja_a, vol_a = medir(args.a)
    _, caja_b, vol_b = medir(args.b)
    for nombre, caja, vol in ((args.a, caja_a, vol_a), (args.b, caja_b, vol_b)):
        ejes = "  ".join(f"{e} [{lo:8.1f}, {hi:8.1f}]" for e, (lo, hi) in zip("XYZ", caja))
        print(f"{nombre}\n  {ejes}   volumen {vol:,.0f} mm³")
    problemas = diagnostico(caja_a, caja_b, vol_a, vol_b)
    if problemas:
        print("\nNO COINCIDEN:")
        for p in problemas:
            print(f"  - {p}")
        return 1
    print(f"\nCOINCIDEN (caja ±{TOLERANCIA_MM} mm, volumen ±{TOLERANCIA_VOLUMEN:.1%}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
