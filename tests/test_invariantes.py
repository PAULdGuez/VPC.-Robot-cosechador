"""INV1 y las reglas de un paso válido (GS2, GS3, GS7, GS11): la función pura, sin guardia ni placa.

Los criterios (palabras del dueño) viven en el nombre de cada prueba: test_inv01_…, test_gs02_…, etc.
"""
import dataclasses
import decimal
import fractions
import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from cosechador.movimiento import invariantes
from cosechador.movimiento.invariantes import (HOLGURA_RAFAGA_PASOS, HOLGURA_RELATIVA, avance_punta_mm,
                                               intervalo_ms, revisar_paso, revisar_plan)
from cosechador.movimiento.trayectorias import Paso, posicion, tramo_articular
from cosechador.robot.cinematica import primer_servo_fuera, servos_en_rango
from helpers_guardia import NOMBRES_ROBOT, PROPIEDADES, punto_medio, robot_de


@pytest.fixture(params=NOMBRES_ROBOT)
def r(request):
    return robot_de(request.param)


def con_servo(robot, i, valor, base=None):
    s = list(base or punto_medio(robot))
    s[i] = valor
    return tuple(s)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INV1: UNA SOLA FUNCIÓN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_inv01_revisar_plan_decide_paso_a_paso_con_revisar_paso(r, monkeypatch):
    """Un plan se revisa llamando a LA función por cada paso con servos: no tiene reglas aparte."""
    llamadas = []
    original = invariantes.revisar_paso
    monkeypatch.setattr(invariantes, "revisar_paso", lambda *a, **k: llamadas.append(a[1]) or original(*a, **k))
    inicio = punto_medio(r)
    plan = [Paso(aviso="hola")] + tramo_articular(r, inicio, tuple(s + 40 for s in inicio)) + [Paso(ms=100)]
    informe = revisar_plan(r, plan, inicio)
    esperados = [p.servos for p in plan if p.servos is not None]
    assert informe.ok and informe.pasos_revisados == len(esperados) > 1
    assert llamadas == esperados


def test_inv01_un_plan_con_un_paso_malo_se_para_en_ese_paso_con_la_misma_violacion(r):
    inicio = punto_medio(r)
    malo = con_servo(r, 0, inicio[0] + 16, inicio)
    plan = [Paso(servos=inicio), Paso(servos=malo), Paso(servos=inicio)]
    informe = revisar_plan(r, plan, inicio)
    directo = revisar_paso(r, malo, inicio).violacion
    assert not informe.ok
    assert informe.violacion == directo._replace(paso=1)
    assert informe.violacion.clave == "salto_excedido"


def test_inv01_el_avance_de_la_punta_solo_se_mide_y_no_rechaza(r):
    """Un plan con la punta corriendo muchos mm entre pasos, pero dentro de los topes de servo, es válido."""
    a = punto_medio(r)
    b = tuple(s + 15 for s in a)
    plan = [Paso(servos=a), Paso(servos=b)]
    informe = revisar_plan(r, plan, a)
    esperado = avance_punta_mm(r, a, b)
    assert informe.ok
    assert esperado > 0
    assert informe.avance_max_mm == pytest.approx(esperado)
    assert informe.avance_en_paso == 1
    assert avance_punta_mm(r, a, a) == 0
    assert avance_punta_mm(r, a, b) == pytest.approx(float(np.linalg.norm(posicion(r, b) - posicion(r, a))))


def test_inv01_planes_y_tuplas_sueltas_dan_el_mismo_informe(r):
    inicio = punto_medio(r)
    pasos = tramo_articular(r, inicio, tuple(s - 30 for s in inicio))
    assert revisar_plan(r, pasos, inicio) == revisar_plan(r, [p.servos for p in pasos], inicio)


def test_inv01_un_plan_con_inicio_invalido_no_se_revisa_mas(r):
    informe = revisar_plan(r, [Paso(servos=punto_medio(r))], (math.nan,) * r.gdl)
    assert not informe.ok and informe.violacion.clave == "no_es_numero" and informe.violacion.paso == -1


@PROPIEDADES
@given(data=st.data())
def test_inv01_el_rango_lo_decide_un_solo_lugar(data):
    """servos_en_rango (lo que usa el planeador y los programas) y la revisión no pueden discrepar."""
    robot = robot_de(data.draw(st.sampled_from(NOMBRES_ROBOT)))
    servos = tuple(data.draw(st.floats(-20, 200, allow_nan=False)) for _ in range(robot.gdl))
    assert servos_en_rango(robot, servos) == (primer_servo_fuera(robot, servos) is None)
    if not servos_en_rango(robot, servos):
        assert revisar_paso(robot, servos).violacion.clave == "fuera_de_rango"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GS2: RANGO Y NÚMEROS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

NO_NUMEROS = [math.nan, math.inf, -math.inf, None, "90", "", True, False, [], (90,), b"90", 10 ** 400, -10 ** 400,
              1j, decimal.Decimal(90), np.bool_(True), object()]


@pytest.mark.parametrize("malo", NO_NUMEROS, ids=repr)
def test_gs02_un_servo_que_no_es_un_numero_no_sale(r, malo):
    for i in range(r.gdl):
        v = revisar_paso(r, con_servo(r, i, malo))
        assert not v.ok and v.servos is None
        assert v.violacion.clave == "no_es_numero" and v.violacion.servo == i + 1


@pytest.mark.parametrize("malo", [None, 90, 90.0, "90,90,90", b"90", {1, 2, 3}, {"a": 1}, math.nan, object(),
                                  np.zeros((3, 3))], ids=repr)
def test_gs02_una_orden_que_no_es_una_lista_no_sale(r, malo):
    v = revisar_paso(r, malo).violacion
    assert v is not None and v.clave == "no_es_lista"


@pytest.mark.parametrize("bueno", [np.float64(90), np.int64(90), 90, 90.0, fractions.Fraction(90)], ids=repr)
def test_gs02_los_numeros_de_verdad_si_pasan(r, bueno):
    assert revisar_paso(r, con_servo(r, 0, bueno)).ok


def test_gs02_una_lista_de_numpy_se_acepta_y_sale_como_floats_de_python(r):
    v = revisar_paso(r, np.array(punto_medio(r)))
    assert v.ok and all(type(s) is float for s in v.servos)


def test_gs02_los_bordes_del_rango_pasan_y_un_grado_fuera_no(r):
    for i, j in enumerate(r.articulaciones):
        lo, hi = j.servo.minimo, j.servo.maximo
        assert revisar_paso(r, con_servo(r, i, lo)).ok, f"servo {i + 1} en su minimo"
        assert revisar_paso(r, con_servo(r, i, hi)).ok, f"servo {i + 1} en su maximo"
        for fuera, tope in ((lo - 1, lo), (hi + 1, hi)):
            v = revisar_paso(r, con_servo(r, i, fuera)).violacion
            assert v is not None and v.clave == "fuera_de_rango"
            assert (v.servo, v.pedido, v.tope) == (i + 1, fuera, tope)
            assert f"servo {i + 1}" in v.motivo and "rango" in v.motivo


def test_gs02_la_tolerancia_es_la_del_planeador_y_no_mas(r):
    """0.005° de ruido numérico pasa (y llega a la placa como el borde); 0.02° ya es salirse."""
    hi = r.articulaciones[0].servo.maximo
    assert revisar_paso(r, con_servo(r, 0, hi + 0.005)).ok
    assert revisar_paso(r, con_servo(r, 0, hi + 0.02)).violacion.clave == "fuera_de_rango"
    lo = r.articulaciones[1].servo.minimo
    assert revisar_paso(r, con_servo(r, 1, lo - 0.005)).ok
    assert revisar_paso(r, con_servo(r, 1, lo - 0.02)).violacion.clave == "fuera_de_rango"


def test_gs02_se_revisa_tambien_el_entero_que_de_verdad_llega_a_la_placa(robot):
    """Con un mínimo de 20.4°, 20.4 está en rango pero la placa recibe 20, que NO lo está: no sale."""
    fraccionario = dataclasses.replace(robot, articulaciones=tuple(
        dataclasses.replace(j, servo=dataclasses.replace(j.servo, minimo=20.4)) if i == 0 else j
        for i, j in enumerate(robot.articulaciones)))
    assert primer_servo_fuera(fraccionario, con_servo(fraccionario, 0, 20.4)) is None  # EL FLOAT SÍ CABE
    v = revisar_paso(fraccionario, con_servo(fraccionario, 0, 20.4)).violacion
    assert v is not None and v.clave == "fuera_de_rango" and v.servo == 1
    assert revisar_paso(fraccionario, con_servo(fraccionario, 0, 20.6)).ok  # 20.6 -> 21, cabe


@PROPIEDADES
@given(data=st.data())
def test_gs02_lo_que_pasa_es_finito_y_cabe_en_su_rango_y_lo_demas_no_lanza(data):
    robot = robot_de(data.draw(st.sampled_from(NOMBRES_ROBOT)))
    valores = st.one_of(st.floats(allow_nan=True, allow_infinity=True), st.floats(-5, 185), st.none(), st.text(max_size=3),
                        st.booleans(), st.integers(-10 ** 3, 10 ** 3), st.just(10 ** 400), st.lists(st.integers(), max_size=2))
    orden = data.draw(st.one_of(st.lists(valores, max_size=robot.gdl + 2),
                                st.lists(st.floats(-5, 185), min_size=robot.gdl, max_size=robot.gdl)))
    resultado = revisar_paso(robot, orden)  # NO DEBE LANZAR NUNCA POR UNA ORDEN MALA
    if resultado.ok:
        assert len(resultado.servos) == robot.gdl
        for s, j in zip(resultado.servos, robot.articulaciones):
            assert type(s) is float and math.isfinite(s)
            assert j.servo.minimo - 0.01 <= s <= j.servo.maximo + 0.01
            assert j.servo.minimo <= round(s) <= j.servo.maximo
    else:
        assert resultado.violacion.clave and resultado.violacion.motivo


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GS3: SALTO POR PASO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_gs03_justo_en_el_tope_pasa_y_apenas_arriba_no(r):
    tope = r.movimiento.salto_max_por_paso
    base = punto_medio(r)
    for i in range(r.gdl):
        for signo in (1, -1):
            assert revisar_paso(r, con_servo(r, i, base[i] + signo * tope, base), base).ok
            v = revisar_paso(r, con_servo(r, i, base[i] + signo * (tope + 0.001), base), base).violacion
            assert v is not None and v.clave == "salto_excedido"
            assert v.servo == i + 1 and v.pedido == pytest.approx(tope + 0.001) and v.tope == tope
            assert f"servo {i + 1}" in v.motivo


def test_gs03_el_ruido_de_coma_flotante_del_planeador_no_es_un_salto_de_mas(r):
    tope = r.movimiento.salto_max_por_paso
    base = punto_medio(r)
    assert revisar_paso(r, con_servo(r, 0, base[0] + tope + 1e-9, base), base).ok


def test_gs03_el_tope_es_por_servo_no_la_suma(r):
    base = punto_medio(r)
    todos = tuple(s + r.movimiento.salto_max_por_paso for s in base)
    assert revisar_paso(r, todos, base).ok


def test_gs03_el_tope_sale_de_los_datos_del_robot(robot):
    """Con otro salto_max_por_paso en el robot, cambia el tope: no hay un 15 escrito en el código."""
    lento = dataclasses.replace(robot, movimiento=dataclasses.replace(robot.movimiento, salto_max_por_paso=5.0))
    base = punto_medio(lento)
    assert revisar_paso(lento, con_servo(lento, 0, base[0] + 5, base), base).ok
    assert revisar_paso(lento, con_servo(lento, 0, base[0] + 6, base), base).violacion.clave == "salto_excedido"


@PROPIEDADES
@given(data=st.data())
def test_gs03_pasa_si_y_solo_si_ningun_servo_gira_mas_que_el_tope(data):
    robot = robot_de(data.draw(st.sampled_from(NOMBRES_ROBOT)))
    pose = st.tuples(*[st.floats(j.servo.minimo, j.servo.maximo) for j in robot.articulaciones])
    a, b = data.draw(pose), data.draw(pose)
    giro = max(abs(x - y) for x, y in zip(a, b))
    tope = robot.movimiento.salto_max_por_paso
    ok = revisar_paso(robot, b, a).ok  # SIN dt: SOLO EL SALTO
    if giro > tope + 1e-5:
        assert not ok
    if giro <= tope:
        assert ok


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GS11: CANTIDAD DE SERVOS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_gs11_la_trampa_servos_en_rango_da_por_buena_una_orden_corta(robot):
    """Por qué existe GS11: servos_en_rango usa zip y no cuenta."""
    assert servos_en_rango(robot, (90, 90))
    assert servos_en_rango(robot, ())


def test_gs11_mas_o_menos_servos_de_los_que_tiene_el_robot_no_sale(r):
    buena = punto_medio(r)
    for n in (0, 1, r.gdl - 1, r.gdl + 1, r.gdl + 3, 50):
        orden = (buena * 20)[:n]
        v = revisar_paso(r, orden).violacion
        assert v is not None and v.clave == "cantidad_servos", n
        assert (v.pedido, v.tope) == (n, r.gdl)
        assert str(n) in v.motivo and str(r.gdl) in v.motivo
    assert revisar_paso(r, buena).ok


def test_gs11_una_orden_de_otro_robot_no_sale_aunque_sus_servos_sean_validos(robot, robot_4gdl):
    assert not revisar_paso(robot, punto_medio(robot_4gdl)).ok
    assert not revisar_paso(robot_4gdl, punto_medio(robot)).ok


@PROPIEDADES
@given(data=st.data())
def test_gs11_ninguna_orden_de_otro_largo_sale(data):
    robot = robot_de(data.draw(st.sampled_from(NOMBRES_ROBOT)))
    n = data.draw(st.integers(0, 12).filter(lambda k: k != robot.gdl))
    orden = data.draw(st.lists(st.floats(0, 180), min_size=n, max_size=n))
    assert revisar_paso(robot, orden).violacion.clave == "cantidad_servos"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GS7: EL TOPE TAMBIÉN POR TIEMPO (LA FUNCIÓN PURA) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_gs07_el_ritmo_maximo_sale_de_los_datos_y_es_300_grados_por_segundo_con_los_actuales(robot):
    mov = robot.movimiento
    assert mov.salto_max_por_paso / mov.ms_por_paso * 1000 == 300
    assert HOLGURA_RELATIVA == 0.05 and HOLGURA_RAFAGA_PASOS == 0.8


def test_gs07_un_paso_completo_cada_ms_por_paso_pasa_siempre(r):
    base, ms = punto_medio(r), r.movimiento.ms_por_paso
    dt, deuda = ms / 1000, 0.0
    pose = base
    for k in range(200):
        siguiente = con_servo(r, 0, base[0] + (r.movimiento.salto_max_por_paso if k % 2 == 0 else 0), base)
        v = revisar_paso(r, siguiente, pose, dt, deuda)
        assert v.ok, k
        pose, deuda = v.servos, v.deuda_ms
    assert deuda == 0


def test_gs07_dos_ordenes_completas_con_la_misma_hora_no_pasan(r):
    base = punto_medio(r)
    uno = revisar_paso(r, con_servo(r, 0, base[0] + 15, base), base, 1.0, 0.0)
    assert uno.ok
    dos = revisar_paso(r, con_servo(r, 0, base[0] + 30, base), uno.servos, 0.0, uno.deuda_ms)
    assert dos.violacion.clave == "ritmo_excedido" and dos.violacion.servo == 1
    assert dos.violacion.pedido == pytest.approx(15) and dos.violacion.tope < 15
    assert "°/s" in dos.violacion.motivo and "ms" in dos.violacion.motivo


def test_gs07_muchas_ordenes_chicas_muy_seguidas_no_pasan_aunque_cada_una_cumpla_el_tope_por_paso(r):
    base = punto_medio(r)
    pose, deuda, salieron = base, 0.0, 0
    for k in range(1, 200):
        v = revisar_paso(r, con_servo(r, 0, base[0] + k * 1.0, base), pose, 0.001, deuda)
        if not v.ok:
            assert v.violacion.clave == "ritmo_excedido"
            break
        pose, deuda, salieron = v.servos, v.deuda_ms, salieron + 1
    else:
        pytest.fail("pasaron 199 órdenes de 1° con 1 ms entre ellas (1000 °/s)")
    assert salieron < 25


def test_gs07_la_holgura_deja_pasar_un_temporizador_que_dispara_un_poco_antes(r):
    """1 ms antes de tiempo en CADA paso (49 ms en vez de 50), durante 1000 pasos completos: ni una negativa."""
    base = punto_medio(r)
    pose, deuda = base, 0.0
    for k in range(1000):
        sig = con_servo(r, 0, base[0] + (15 if k % 2 == 0 else 0), base)
        v = revisar_paso(r, sig, pose, 0.049, deuda)
        assert v.ok, k
        pose, deuda = v.servos, v.deuda_ms


def test_gs07_pero_un_ritmo_sostenido_apenas_arriba_de_la_holgura_si_se_niega(r):
    """40 ms entre pasos completos (375 °/s, 25 % más rápido): la ráfaga se agota y se niega."""
    base = punto_medio(r)
    pose, deuda, negadas = base, 0.0, 0
    for k in range(400):
        sig = con_servo(r, 0, base[0] + (15 if k % 2 == 0 else 0), base)
        v = revisar_paso(r, sig, pose, 0.040, deuda)
        if not v.ok:
            negadas += 1
            break
        pose, deuda = v.servos, v.deuda_ms
    assert negadas == 1


def test_gs07_una_pausa_larga_no_guarda_permiso_para_una_rafaga_despues(r):
    base = punto_medio(r)
    despues_de_la_pausa = revisar_paso(r, con_servo(r, 0, base[0] + 15, base), base, 3600.0, 0.0)
    assert despues_de_la_pausa.ok and despues_de_la_pausa.deuda_ms == 0
    otra = revisar_paso(r, con_servo(r, 0, base[0] + 30, base), despues_de_la_pausa.servos, 0.0, 0.0)
    assert otra.violacion.clave == "ritmo_excedido"


def test_gs07_una_hora_que_retrocede_no_regala_tiempo(r):
    """dt negativo cuenta como 0: cada orden se anota como deuda y la tercera de 8° ya no cabe."""
    base = punto_medio(r)
    uno = revisar_paso(r, con_servo(r, 0, base[0] + 8, base), base, -5.0, 0.0)
    assert uno.ok and uno.deuda_ms > 0  # NO SE ANOTA COMO TIEMPO A FAVOR
    dos = revisar_paso(r, con_servo(r, 0, base[0] + 16, base), uno.servos, -5.0, uno.deuda_ms)
    assert dos.violacion.clave == "ritmo_excedido"


def test_gs07_un_paso_completo_con_la_misma_hora_que_el_anterior_no_pasa(r):
    base = punto_medio(r)
    v = revisar_paso(r, con_servo(r, 0, base[0] + 15, base), base, 0.0, 0.0)
    assert v.violacion.clave == "ritmo_excedido"


def test_gs07_la_misma_pose_a_la_misma_hora_no_cuesta_nada(r):
    base = punto_medio(r)
    pose, deuda = base, 0.0
    for _ in range(100):
        v = revisar_paso(r, base, pose, 0.0, deuda)
        assert v.ok and v.deuda_ms == 0
        pose, deuda = v.servos, v.deuda_ms


def test_gs07_el_intervalo_de_un_paso_es_el_que_arma_la_ventana(r):
    ms = r.movimiento.ms_por_paso
    assert intervalo_ms(r, Paso(servos=punto_medio(r))) == round(ms)
    assert intervalo_ms(r, Paso(servos=punto_medio(r), ms=100), velocidad=50) == 200
    assert intervalo_ms(r, Paso(aviso="x")) == 1
    assert intervalo_ms(r, Paso(ms=2000)) == 2000


@pytest.mark.parametrize("dt", [math.nan, math.inf, "1"], ids=repr)
def test_gs07_un_dt_que_no_es_numero_se_rechaza_en_vez_de_darse_por_bueno(r, dt):
    base = punto_medio(r)
    with pytest.raises(ValueError):
        revisar_paso(r, base, base, dt, 0.0)


@pytest.mark.parametrize("deuda", [math.nan, math.inf, None, "1"], ids=repr)
def test_gs07_una_deuda_que_no_es_numero_se_rechaza_en_vez_de_darse_por_buena(r, deuda):
    base = punto_medio(r)
    with pytest.raises(ValueError):
        revisar_paso(r, base, base, 0.05, deuda)
