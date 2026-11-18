import os
import sys

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from cosechador.robot import cargar_robot

CARPETA_PRUEBAS = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="session")
def robot():
    return cargar_robot()


@pytest.fixture(scope="session")
def robot_4gdl():
    return cargar_robot(os.path.join(CARPETA_PRUEBAS, "robots", "brazo_4gdl.yaml"))
