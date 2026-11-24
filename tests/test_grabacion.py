"""F1, bloque 2: grabación de movimientos, la lógica (sin Qt). La ventana: test_grabacion_ventana.py.

La lógica vive en cosechador/movimiento/grabadora.py y se prueba con un reloj falso: nada duerme.
"""
import pytest

from cosechador.movimiento import programa as prog
from cosechador.movimiento.grabadora import Grabadora, programa_de_grabacion
from cosechador.movimiento.trayectorias import duracion_ms, plan_grabacion, servos_del_plan, tramo_articular
from cosechador.robot.cinematica_inversa import angulos_servo, salto

PASO_S = 0.05  # LO QUE DURA UN PASO A VELOCIDAD 100 % (ms_por_paso DEL ROBOT)


class RelojFalso:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture(scope="module")
def home_s(robot):
    return angulos_servo(robot, robot.home)


def grabar_pasos(grabadora, reloj, pasos):
    """Como el temporizador de la ventana: manda cada pose y espera lo que dura su paso."""
    for paso in pasos:
        grabadora.anotar(paso.servos, PASO_S)
        reloj.t += PASO_S


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% B2.2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@pytest.mark.parametrize("quieto_s", [0.5, 3.0, 10.0])
def test_b2_2_quieto_n_segundos_al_reproducir_espera_n_segundos(robot, home_s, quieto_s):
    reloj = RelojFalso()
    g = Grabadora(reloj=reloj)
    g.empezar(home_s)
    a = (home_s[0] + 30, home_s[1], home_s[2])
    grabar_pasos(g, reloj, tramo_articular(robot, home_s, a))
    reloj.t += quieto_s                                       # QUIETO, SIN MANDAR NADA
    grabar_pasos(g, reloj, tramo_articular(robot, a, home_s))
    p = programa_de_grabacion(g.terminar(), "quieto", robot)

    plan, _ = plan_grabacion(robot, home_s, p)
    esperas = sum(paso.ms for paso in plan if paso.servos is None and paso.ms)
    assert esperas == pytest.approx(quieto_s * 1000, rel=0.10)
    # EN TOTAL: IR AL INICIO (UN PASO, YA ESTÁ AHÍ) + LO QUE SE MOVIÓ + LO QUE SE QUEDÓ QUIETO
    movimiento_ms = 2 * len(tramo_articular(robot, home_s, a)) * PASO_S * 1000
    assert duracion_ms(robot, plan) == pytest.approx(PASO_S * 1000 + movimiento_ms + quieto_s * 1000, rel=0.10)
    # Y CON «QUITAR PAUSAS» YA NO ESPERA
    sin_pausas, _ = plan_grabacion(robot, home_s, p, quitar_pausas=True)
    assert sum(paso.ms for paso in sin_pausas if paso.servos is None and paso.ms) == 0


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% B2.3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_b2_3_nombre_de_archivo_desde_lo_que_escribe_la_persona():
    assert prog.nombre_de_archivo("Mi grabación 2") == "mi-grabacion-2.yaml"
    assert prog.nombre_de_archivo("  Cosecha / árbol #3  ") == "cosecha-arbol-3.yaml"
    assert prog.nombre_de_archivo("demo.yaml") == "demo.yaml"
    assert prog.nombre_de_archivo("../../etc/x") == "etc-x.yaml"      # NUNCA SALE DE programas/
    for vacio in ("", "   ", "¿¡!?"):
        with pytest.raises(prog.ErrorPrograma, match="letra o un número"):
            prog.nombre_de_archivo(vacio)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% B2.4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_b2_4_sin_movimiento_no_hay_grabacion(home_s):
    reloj = RelojFalso()
    g = Grabadora(reloj=reloj)
    g.empezar(home_s)
    reloj.t += 5
    assert g.terminar() is None and not g.grabando
    g.empezar(home_s)
    g.anotar(home_s, PASO_S)                    # SE MANDÓ LA MISMA POSE: TAMPOCO ES MOVIMIENTO
    assert g.terminar() is None


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% B2.6 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_b2_6_la_grabacion_se_reproduce_igual_desde_el_archivo(robot, home_s, tmp_path):
    reloj = RelojFalso()
    g = Grabadora(reloj=reloj)
    g.empezar(home_s)
    poses = [home_s, (120.0, 60.0, 70.0), (60.0, 30.0, 100.0), (95.5, 47.25, 52.125)]
    enviadas = []
    for a, b in zip(poses, poses[1:]):
        pasos = tramo_articular(robot, a, b)
        enviadas += [p.servos for p in pasos]
        grabar_pasos(g, reloj, pasos)
        reloj.t += 0.7
    prog.guardar(programa_de_grabacion(g.terminar(), "archivo", robot), tmp_path / "archivo.yaml")
    leida = prog.cargar(tmp_path / "archivo.yaml", robot)
    plan, _ = plan_grabacion(robot, (10.0, 100.0, 20.0), leida)
    recorridas = servos_del_plan(plan)
    # CADA POSE QUE SE LE MANDÓ AL ROBOT AL GRABAR APARECE, EN ORDEN, AL REPRODUCIR DESDE EL ARCHIVO (±0.5°)
    i = 0
    for pose in enviadas:
        while i < len(recorridas) and salto(recorridas[i], pose) > 0.5:
            i += 1
        assert i < len(recorridas), f"no pasó por {pose}"
