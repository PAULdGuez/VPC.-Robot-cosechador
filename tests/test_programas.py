"""F1, bloque 1: programas (archivos, validación y planes de reproducción)."""
import numpy as np
import pytest

from cosechador.movimiento import programa as prog
from cosechador.movimiento.programa import ErrorPrograma, Muestra, Programa, Punto
from cosechador.movimiento.trayectorias import (duracion_ms, plan_grabacion, plan_puntos, posicion,
                                                servos_del_plan, tramo_articular)
from cosechador.robot.cinematica_inversa import angulos_servo, salto


@pytest.fixture(scope="module")
def home_s(robot):
    return angulos_servo(robot, robot.home)


def programa_puntos(robot, *puntos):
    return Programa("prueba", "puntos", robot.nombre, robot.gdl, puntos=tuple(puntos))


def saltos(inicio, plan):
    s = [inicio] + servos_del_plan(plan)
    return [salto(s[i], s[i + 1]) for i in range(len(s) - 1)]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ARCHIVOS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_b1_1_ida_y_vuelta_del_archivo(robot, home_s, tmp_path):
    puntos = programa_puntos(robot, Punto(home_s, "articular", 100),
                             Punto(angulos_servo(robot, (356, 711, 1014)), "lineal", 20))
    grabacion = Programa("g", "grabacion", robot.nombre, robot.gdl,
                         muestras=(Muestra(0.0, home_s), Muestra(1.25, (100.0, 50.0, 60.0))))
    for original in (puntos, grabacion):
        ruta = tmp_path / f"{original.tipo}.yaml"
        prog.guardar(original, ruta)
        leido = prog.cargar(ruta, robot)
        assert leido.tipo == original.tipo and leido.nombre == original.nombre
        for a, b in zip(leido.puntos or leido.muestras, original.puntos or original.muestras):
            assert np.allclose(a.servos, b.servos, atol=1e-3)
        assert [(p.movimiento, p.velocidad) for p in leido.puntos] == [(p.movimiento, p.velocidad) for p in original.puntos]
        assert [m.t for m in leido.muestras] == [m.t for m in original.muestras]
    assert "servos: [" in (tmp_path / "puntos.yaml").read_text()  # LEGIBLE: LOS SERVOS EN UNA LÍNEA


def _base(home_s):
    return {"formato": 1, "nombre": "x", "tipo": "puntos", "robot": "r", "gdl": 3,
            "puntos": [{"servos": list(home_s), "movimiento": "articular", "velocidad": 100}]}


@pytest.mark.parametrize("romper, mensaje", [
    (lambda d: d.update({"gdl": 4}), "4 articulaciones y este tiene 3"),
    (lambda d: d.update({"formato": 9}), "Formato de programa desconocido"),
    (lambda d: d.update({"tipo": "baile"}), "'tipo'"),
    (lambda d: d["puntos"][0].update({"servos": [90, 5, 46]}), "salen del rango"),
    (lambda d: d["puntos"][0].update({"servos": [90, 46]}), "lista de 3 números"),
    (lambda d: d["puntos"][0].update({"movimiento": "salto"}), "'movimiento'"),
    (lambda d: d["puntos"][0].update({"velocidad": 150}), "'velocidad'"),
    (lambda d: d.update({"puntos": []}), "al menos un punto"),
    (lambda d: d.update({"tipo": "grabacion", "muestras": [{"t": 1, "servos": [90, 46, 46]}, {"t": 0.5, "servos": [90, 46, 46]}]}), "hacia atrás"),
])
def test_b1_2_archivo_invalido_dice_que_esta_mal(robot, home_s, romper, mensaje):
    datos = _base(home_s)
    romper(datos)
    with pytest.raises(ErrorPrograma, match=mensaje.replace("(", r"\(")):
        prog.programa_desde_dict(datos, robot)


def test_b1_2_yaml_roto_y_archivo_que_no_existe(robot, tmp_path):
    roto = tmp_path / "roto.yaml"
    roto.write_text("puntos: [1, 2\n")
    with pytest.raises(ErrorPrograma, match="roto.yaml no es YAML válido"):
        prog.cargar(roto, robot)
    with pytest.raises(ErrorPrograma, match="No existe"):
        prog.cargar(tmp_path / "nada.yaml", robot)


def test_b1_3_listar_solo_programas(tmp_path):
    for nombre in ("b.yaml", "a.yml", ".oculto.yaml", "notas.txt"):
        (tmp_path / nombre).write_text("x")
    assert prog.listar(tmp_path) == ["a.yml", "b.yaml"]
    assert prog.listar(tmp_path / "no-existe") == []


def test_los_programas_del_repo_son_validos(robot):
    for nombre in prog.listar():
        prog.cargar(f"{prog.CARPETA_PROGRAMAS}/{nombre}", robot)  # NO DEBE LANZAR


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% REPRODUCCIÓN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_b1_4_reproducir_puntos(robot, home_s):
    arriba, corte = (356, 711, 1150), (356, 711, 1014)
    p = programa_puntos(robot,
                        Punto(angulos_servo(robot, arriba), "articular", 100),
                        Punto(angulos_servo(robot, corte), "lineal", 20),
                        Punto(home_s, "articular", 100))
    plan, problemas = plan_puntos(robot, home_s, p)
    assert problemas == []
    visitadas = [posicion(robot, s) for s in servos_del_plan(plan)]
    for objetivo in (arriba, corte, robot.home):
        assert min(np.linalg.norm(v - objetivo) for v in visitadas) < 0.5
    assert max(saltos(home_s, plan)) <= robot.movimiento.salto_max_por_paso + 1e-9
    # EL TRAMO LINEAL VA EN RECTA (X, Y FIJAS) Y A 20 %: CADA PASO DURA 5 VECES MÁS
    lineal = [q for q in plan if q.servos is not None and q.ms == pytest.approx(robot.movimiento.ms_por_paso * 5)]
    assert lineal
    for q in lineal:
        x, y, _ = posicion(robot, q.servos)
        assert (x, y) == pytest.approx((356, 711), abs=0.5)


def test_b1_5_programa_imposible_no_arranca(robot):
    # DE (-1400, 150, 1300) A (-1400, -150, 1300) EN LÍNEA RECTA HAY QUE VOLTEAR LA BASE: EN «negar» NO SE PUEDE
    p = programa_puntos(robot,
                        Punto(angulos_servo(robot, (-1400, 150, 1300)), "articular", 100),
                        Punto(angulos_servo(robot, (-1400, -150, 1300)), "lineal", 100))
    inicio = angulos_servo(robot, (-1400, 150, 1300))
    plan, problemas = plan_puntos(robot, inicio, p, postura="negar")
    assert plan is None
    assert [n for n, _ in problemas] == [2] and "«negar»" in problemas[0][1]
    # EN ARTICULAR EL MISMO PUNTO SÍ SE ALCANZA
    p2 = programa_puntos(robot, p.puntos[0], Punto(p.puntos[1].servos, "articular", 100))
    assert plan_puntos(robot, inicio, p2, postura="negar")[1] == []


def test_b1_6_reproducir_grabacion(robot, home_s):
    a, b = home_s, (home_s[0] + 30, home_s[1], home_s[2])
    g = Programa("g", "grabacion", robot.nombre, robot.gdl, muestras=(
        Muestra(0.0, a), Muestra(1.0, b),     # 1 s MOVIÉNDOSE
        Muestra(2.0, b),                       # 1 s QUIETO
        Muestra(3.0, a)))                      # 1 s DE REGRESO
    plan, _ = plan_grabacion(robot, home_s, g)
    assert duracion_ms(robot, plan) == pytest.approx(3000, rel=0.10)
    sin_pausas, _ = plan_grabacion(robot, home_s, g, quitar_pausas=True)
    assert duracion_ms(robot, sin_pausas) == pytest.approx(2000, rel=0.10)
    assert max(saltos(home_s, plan)) <= robot.movimiento.salto_max_por_paso + 1e-9
    assert posicion(robot, servos_del_plan(plan)[-1]) == pytest.approx(posicion(robot, a), abs=1e-6)


def test_b1_6_grabacion_demasiado_rapida_avisa(robot, home_s):
    # 90° EN 0.1 s: EL SG90 NO PUEDE (6 PASOS DE 15° × 50 ms = 0.3 s)
    g = Programa("rapida", "grabacion", robot.nombre, robot.gdl, muestras=(
        Muestra(0.0, (0.0, 46.0, 46.0)), Muestra(0.1, (90.0, 46.0, 46.0))))
    plan, _ = plan_grabacion(robot, (0.0, 46.0, 46.0), g)
    assert duracion_ms(robot, plan) >= 300 - 1e-6
    assert "más que lo grabado" in plan[-1].aviso


def test_b1_8_slider_de_servo_al_extremo_va_poco_a_poco(robot):
    pasos = tramo_articular(robot, (0, 46, 46), (180, 46, 46))
    assert len(pasos) == 12                       # 180° / 15° POR PASO
    assert max(saltos((0, 46, 46), pasos)) <= robot.movimiento.salto_max_por_paso + 1e-9
    assert pasos[-1].servos == pytest.approx((180, 46, 46))
