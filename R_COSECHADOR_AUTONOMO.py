"""Lanzador de siempre. Equivale a:  python -m cosechador [--robot robots/otro.yaml]"""
import sys

from cosechador.app import main

if __name__ == "__main__":
    sys.exit(main())
