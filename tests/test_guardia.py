"""La guardia de salida: GS5, GS7, GS8, GS9, GS10 (mitad sin pantalla) e INV1 vista desde la guardia.

Sin ventana ni placa: un enlace falso que anota lo que de verdad «salió» y un reloj falso que mueve la prueba.
"""
import logging
import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from cosechador.hardware import guardia as modulo_guardia
from cosechador.hardware.guardia import GuardiaDeSalida, OrdenNegada
from cosechador.movimiento import programa as prog
from cosechador.movimiento.invariantes import (HOLGURA_RAFAGA_PASOS, HOLGURA_RELATIVA, Revision, Violacion,
                                               intervalo_ms, revisar_paso, revisar_plan)
from cosechador.movimiento.trayectorias import (Paso, plan_cosecha, plan_grabacion, plan_ir_y_volver, plan_puntos,
                                                tramo_articular, tramo_continuo)
from helpers_guardia import CARPETA_REPO, NOMBRES_ROBOT, PROPIEDADES, EnlaceFalso, Reloj, punto_medio, robot_de


@pytest.fixture(params=NOMBRES_ROBOT)
def r(request):
    return robot_de(request.param)


def armar(robot, t=1000.0):
    reloj = Reloj(t)
    enlace = EnlaceFalso(reloj)
    return GuardiaDeSalida(enlace, robot, reloj), enlace, reloj


def mueve(robot, base, delta, i=0):
    s = list(base)
    s[i] += delta
    return tuple(s)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INV1: LA GUARDIA NO TIENE REGLAS PROPIAS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_inv01_la_guardia_niega_lo_que_revisar_paso_niega_aunque_la_orden_sea_buena(r, monkeypatch):
    g, enlace, _ = armar(r)
    veto = Violacion("salto_excedido", "veto de prueba", servo=1)
    monkeypatch.setattr(modulo_guardia, "revisar_paso", lambda *a, **k: Revision(veto, None, 0.0))
    with pytest.raises(OrdenNegada) as e:
        g.enviar(*punto_medio(r))
    assert e.value.violacion == veto and enlace.salidas == []


def test_inv01_la_guardia_deja_salir_lo_que_revisar_paso_acepta_aunque_la_orden_sea_mala(r, monkeypatch):
    g, enlace, _ = armar(r)
    aceptada = tuple(float(s) for s in punto_medio(r))
    monkeypatch.setattr(modulo_guardia, "revisar_paso", lambda *a, **k: Revision(None, aceptada, 0.0))
    g.enviar(float("nan"))  # LA ORDEN ES BASURA: LO QUE DECIDE ES LA FUNCIÓN ÚNICA, Y SALE LO QUE ELLA SANEÓ
    assert enlace.servos == [aceptada]


SALTOS = st.sampled_from([0.0, 1.0, 5.0, 14.9, 15.0, 15.1, 40.0])
PAUSAS_MS = st.sampled_from([1, 1, 5, 20, 39, 40, 41, 50, 50, 60, 100, 5000])


@PROPIEDADES
@given(data=st.data())
def test_inv01_la_guardia_y_revisar_plan_deciden_igual_la_primera_negativa(data):
    """La misma secuencia con las mismas horas: la guardia niega en el mismo paso y por la misma clave que el plan."""
    robot = robot_de(data.draw(st.sampled_from(NOMBRES_ROBOT)))
    inicio = punto_medio(robot)
    ordenes, esperas, base = [inicio], [data.draw(PAUSAS_MS)], inicio
    for _ in range(data.draw(st.integers(1, 25))):
        if data.draw(st.integers(0, 9)) == 0:
            ordenes.append(tuple(data.draw(st.lists(st.one_of(st.floats(allow_nan=True), st.none(), st.text(max_size=2)),
                                                    max_size=robot.gdl + 1))))
        else:
            base = tuple(b + data.draw(st.sampled_from([-1, 1])) * data.draw(SALTOS) for b in base)
            ordenes.append(base)
        esperas.append(data.draw(PAUSAS_MS))
    g, _, reloj = armar(robot)
    hora, primera = 0, None
    for i, orden in enumerate(ordenes):
        reloj.t = 1000.0 + hora / 1000.0
        try:
            g.enviar(*orden)
        except OrdenNegada as e:
            primera = (i, e.clave)
            break
        hora += esperas[i]
    plan = [Paso(servos=o, ms=m) for o, m in zip(ordenes, esperas)]
    informe = revisar_plan(robot, plan, inicio)
    if primera is None:
        assert informe.ok
    else:
        assert not informe.ok and (informe.violacion.paso, informe.violacion.clave) == primera


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GS8: LA CUENTA ES SUYA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_gs08_la_guardia_lleva_su_cuenta_de_lo_ultimo_que_salio_sin_preguntarle_a_nadie(r):
    g, enlace, reloj = armar(r)
    assert g.ultima is None
    a = punto_medio(r)
    g.enviar(*a)
    assert g.ultima == a
    b = mueve(r, a, 10)
    reloj.t += 1
    g.enviar(*b)
    assert g.ultima == b and enlace.servos == [a, b]
    # EL ENLACE FALSO SOLO SABE enviar (NI POSICIÓN NI ESTADO): DE AHÍ NO SACÓ NADA
    assert not any(n in dir(enlace) for n in ("posicion", "servos_actuales", "leer_posicion"))


def test_gs08_el_salto_se_mide_contra_lo_que_salio_no_contra_lo_que_se_intento(r):
    g, enlace, reloj = armar(r)
    a = punto_medio(r)
    g.enviar(*a)
    lejos = mueve(r, a, 40)
    reloj.t += 1
    with pytest.raises(OrdenNegada) as e1:
        g.enviar(*lejos)
    assert e1.value.clave == "salto_excedido"
    g.reiniciar()
    with pytest.raises(OrdenNegada) as e2:  # SI SE HUBIERA ANOTADO LO INTENTADO, ESTA YA «SALTARÍA» 0°
        g.enviar(*lejos)
    assert e2.value.clave == "salto_excedido" and e2.value.nueva
    assert g.ultima == a
    g.reiniciar()
    reloj.t += 1
    g.enviar(*mueve(r, a, 10))
    assert enlace.servos == [a, mueve(r, a, 10)]


def test_gs08_la_primera_orden_solo_se_revisa_por_rango_y_cantidad(r):
    """Decisión declarada: al arrancar no hay «última que salió», así que no hay salto que medir."""
    g, enlace, _ = armar(r)
    extremo = tuple(j.servo.maximo for j in r.articulaciones)
    g.enviar(*extremo)
    assert enlace.servos == [extremo]
    g2, enlace2, _ = armar(r)
    with pytest.raises(OrdenNegada):
        g2.enviar(*extremo[:-1])
    with pytest.raises(OrdenNegada):
        armar(r)[0].enviar(*tuple(j.servo.maximo + 5 for j in r.articulaciones))
    assert enlace2.salidas == []


def test_gs08_si_la_placa_falla_la_cuenta_conserva_lo_ultimo_que_si_salio(r):
    g, enlace, reloj = armar(r)
    a = punto_medio(r)
    g.enviar(*a)
    enlace.enviar = lambda *s: (_ for _ in ()).throw(OSError("puerto caído"))
    reloj.t += 1
    with pytest.raises(OrdenNegada) as e:
        g.enviar(*mueve(r, a, 5))
    assert e.value.clave == "enlace_fallo" and isinstance(e.value.__cause__, OSError)
    assert "OSError" in e.value.violacion.detalle
    assert g.ultima == a and g.detenida


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GS5 y GS10: DETENIDA HASTA QUE UNA PERSONA LA REINICIA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def negar_una(g, reloj, r, a):
    """Deja detenida a la guardia con una orden de un salto de 40°. Devuelve la OrdenNegada."""
    reloj.t += 1
    with pytest.raises(OrdenNegada) as e:
        g.enviar(*mueve(r, a, 40))
    return e.value


def test_gs05_tras_negarse_la_guardia_queda_detenida_y_no_sale_nada_mas(r):
    g, enlace, reloj = armar(r)
    a = punto_medio(r)
    g.enviar(*a)
    primera = negar_una(g, reloj, r, a)
    assert g.detenida and g.violacion == primera.violacion and primera.nueva
    for k in range(5):
        reloj.t += 3600  # NI EL TIEMPO LA LEVANTA
        with pytest.raises(OrdenNegada) as e:
            g.enviar(*mueve(r, a, 1))  # UNA ORDEN PERFECTAMENTE VÁLIDA
        assert not e.value.nueva
    assert enlace.servos == [a]


def test_gs10_detenida_niega_toda_orden_con_el_mismo_motivo_y_la_misma_clave(r):
    g, enlace, reloj = armar(r)
    a = punto_medio(r)
    g.enviar(*a)
    primera = negar_una(g, reloj, r, a)
    ordenes = [a, mueve(r, a, 1), (), (float("nan"),) * r.gdl, (None,), (1, 2, 3, 4, 5, 6, 7), mueve(r, a, 500)]
    for orden in ordenes:
        with pytest.raises(OrdenNegada) as e:
            g.enviar(*orden)
        assert (e.value.clave, e.value.motivo) == (primera.clave, primera.motivo)
        assert e.value.violacion == primera.violacion and e.value.nueva is False
    assert enlace.servos == [a]


def test_gs10_reiniciar_no_manda_nada_ni_reenvia_la_orden_negada(r):
    g, enlace, reloj = armar(r)
    a = punto_medio(r)
    g.enviar(*a)
    negar_una(g, reloj, r, a)
    antes = list(enlace.salidas)
    g.reiniciar()
    assert not g.detenida and g.violacion is None
    assert enlace.salidas == antes  # NI EL ROBOT SE MUEVE NI SE REENVÍA LO NEGADO
    assert g.ultima == a


def test_gs10_reiniciar_sin_estar_detenida_no_cambia_nada(r):
    g, enlace, _ = armar(r)
    a = punto_medio(r)
    g.enviar(*a)
    g.reiniciar()
    g.reiniciar()
    assert enlace.servos == [a] and g.ultima == a and not g.detenida


def test_gs10_tras_reiniciar_la_misma_orden_negada_se_niega_otra_vez_con_la_misma_clave(r):
    g, enlace, reloj = armar(r)
    a = punto_medio(r)
    g.enviar(*a)
    mala = mueve(r, a, 40)
    reloj.t += 1
    with pytest.raises(OrdenNegada) as e1:
        g.enviar(*mala)
    g.reiniciar()
    with pytest.raises(OrdenNegada) as e2:
        g.enviar(*mala)
    assert e2.value.nueva is True  # ES UNA NEGATIVA NUEVA: LA PANTALLA PUEDE AVISAR OTRA VEZ
    assert (e2.value.clave, e2.value.motivo) == (e1.value.clave, e1.value.motivo)
    assert g.detenida and enlace.servos == [a]


def test_gs10_tras_reiniciar_una_orden_buena_si_sale(r):
    g, enlace, reloj = armar(r)
    a = punto_medio(r)
    g.enviar(*a)
    negar_una(g, reloj, r, a)
    g.reiniciar()
    reloj.t += 1
    b = mueve(r, a, 10)
    g.enviar(*b)
    assert enlace.servos == [a, b] and not g.detenida


def test_gs10_la_negativa_trae_motivo_en_espanol_clave_estable_y_los_datos(r):
    g, _, reloj = armar(r)
    a = punto_medio(r)
    g.enviar(*a)
    e = negar_una(g, reloj, r, a)
    v = e.violacion
    assert v.clave == "salto_excedido" and v.servo == 1 and v.tope == r.movimiento.salto_max_por_paso
    assert v.pedido == pytest.approx(40) and "servo 1" in e.motivo and "40" in e.motivo and "15" in e.motivo
    assert str(e) == e.motivo
    assert isinstance(e, Exception) and not isinstance(e, (OSError, ValueError))


@PROPIEDADES
@given(data=st.data())
def test_gs05_tras_una_negativa_no_sale_nada_hasta_que_alguien_reinicia(data):
    robot = robot_de(data.draw(st.sampled_from(NOMBRES_ROBOT)))
    g, enlace, reloj = armar(robot)
    detenida, primera = False, None
    for _ in range(data.draw(st.integers(1, 40))):
        reloj.t += data.draw(st.sampled_from([0, 0.001, 0.05, 1, 3600]))
        base = g.ultima or punto_medio(robot)
        if data.draw(st.integers(0, 5)) == 0:
            orden = tuple(data.draw(st.lists(st.one_of(st.floats(), st.none(), st.booleans()), max_size=robot.gdl + 1)))
        else:
            tope = data.draw(st.sampled_from([2, 15, 40]))
            orden = tuple(b + data.draw(st.floats(-tope, tope)) for b in base)
        if detenida and data.draw(st.integers(0, 3)) == 0:
            g.reiniciar()
            detenida = False
        antes = len(enlace.salidas)
        try:
            g.enviar(*orden)
            assert not detenida, "salió una orden con la guardia detenida"
            assert len(enlace.salidas) == antes + 1
        except OrdenNegada as e:
            assert len(enlace.salidas) == antes
            if detenida:  # YA ESTABA DETENIDA: MISMO MOTIVO Y CLAVE QUE LA QUE LA DETUVO, Y NO ES NUEVA
                assert (e.clave, e.motivo) == (primera.clave, primera.motivo) and e.nueva is False
            else:
                assert e.nueva is True
                primera, detenida = e, True
        assert g.detenida == detenida


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GS9: SI LA GUARDIA FALLA, NO SALE NADA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class RelojRoto:
    def __init__(self, devuelve=None, lanza=None):
        self.devuelve, self.lanza = devuelve, lanza

    def __call__(self):
        if self.lanza:
            raise self.lanza
        return self.devuelve


def comprobar_fallo_cerrado(g, enlace, r, causa=None):
    with pytest.raises(OrdenNegada) as e:
        g.enviar(*punto_medio(r))
    assert e.value.clave == "guardia_fallo" and e.value.nueva
    assert "falló" in e.value.motivo and e.value.violacion.detalle
    if causa is not None:
        assert e.value.__cause__ is causa
        assert type(causa).__name__ in e.value.violacion.detalle
    assert enlace.salidas == [] and g.detenida
    with pytest.raises(OrdenNegada) as otra:  # Y SIGUE DETENIDA, CON LA MISMA CLAVE
        g.enviar(*punto_medio(r))
    assert otra.value.clave == "guardia_fallo" and otra.value.nueva is False
    assert enlace.salidas == []


@pytest.mark.parametrize("lanza", [RuntimeError("reloj roto"), OSError("sin reloj"), ZeroDivisionError("x")], ids=repr)
def test_gs09_si_el_reloj_falla_no_sale_nada(r, lanza):
    enlace = EnlaceFalso()
    comprobar_fallo_cerrado(GuardiaDeSalida(enlace, r, RelojRoto(lanza=lanza)), enlace, r, lanza)


@pytest.mark.parametrize("hora", [float("nan"), float("inf"), None, "12", True, [1.0]], ids=repr)
def test_gs09_si_el_reloj_da_una_hora_que_no_es_hora_no_sale_nada(r, hora):
    enlace = EnlaceFalso()
    comprobar_fallo_cerrado(GuardiaDeSalida(enlace, r, RelojRoto(devuelve=hora)), enlace, r)


def test_gs09_si_la_revision_misma_lanza_no_sale_nada(r, monkeypatch, caplog):
    g, enlace, _ = armar(r)
    error = ZeroDivisionError("falla dentro de la revisión")

    def revisar_roto(*a, **k):
        raise error

    monkeypatch.setattr(modulo_guardia, "revisar_paso", revisar_roto)
    with caplog.at_level(logging.ERROR, logger=modulo_guardia.__name__):
        comprobar_fallo_cerrado(g, enlace, r, error)
    assert any("falló al revisar" in rec.getMessage() and rec.exc_info for rec in caplog.records)  # QUEDA EL MOTIVO


def test_gs09_si_la_revision_falla_en_la_segunda_orden_la_primera_ya_salio_y_la_segunda_no(r, monkeypatch):
    g, enlace, reloj = armar(r)
    a = punto_medio(r)
    g.enviar(*a)
    monkeypatch.setattr(modulo_guardia, "revisar_paso", lambda *a, **k: 1 / 0)
    reloj.t += 1
    with pytest.raises(OrdenNegada) as e:
        g.enviar(*mueve(r, a, 1))
    assert e.value.clave == "guardia_fallo" and enlace.servos == [a]


@pytest.mark.parametrize("devuelto", [None, "ok", object(), Revision(None, None, 0.0), Revision(None, (90.0,), 0.0)],
                         ids=lambda d: type(d).__name__ + repr(getattr(d, "servos", "")))
def test_gs09_si_la_revision_devuelve_algo_que_no_se_entiende_no_sale_nada(r, monkeypatch, devuelto):
    g, enlace, _ = armar(r)
    monkeypatch.setattr(modulo_guardia, "revisar_paso", lambda *a, **k: devuelto)
    comprobar_fallo_cerrado(g, enlace, r)


def test_gs09_con_un_robot_roto_no_sale_nada():
    class Roto:
        def __getattr__(self, nombre):
            raise AttributeError(nombre)

    enlace = EnlaceFalso()
    g = GuardiaDeSalida(enlace, Roto(), lambda: 1.0)
    with pytest.raises(OrdenNegada) as e:
        g.enviar(90, 90, 90)
    assert e.value.clave == "guardia_fallo" and enlace.salidas == []


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GS7: UNA REPRODUCCIÓN NORMAL PASA COMPLETA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def planes_reales(nombre):
    """(identificador, plan, servos de inicio): planes hechos con el planeador y los programas versionados."""
    robot = robot_de(nombre)
    inicio = robot.postura_home or punto_medio(robot)
    lejos = tuple(j.servo.minimo + 0.9 * (j.servo.maximo - j.servo.minimo) for j in robot.articulaciones)
    cerca = tuple(j.servo.minimo + 0.1 * (j.servo.maximo - j.servo.minimo) for j in robot.articulaciones)
    planes = [("articular_ida", tramo_articular(robot, inicio, lejos), inicio),
              ("articular_extremos", tramo_articular(robot, lejos, cerca), lejos),
              ("articular_lento", tramo_articular(robot, inicio, cerca, ms=500), inicio)]
    if robot.cinematica_inversa:  # SOLO EL 3 GDL TIENE SOLVER
        for k, punto in enumerate(robot.puntos_demo[:6]):
            plan, motivo = tramo_continuo(robot, inicio, punto)
            if plan:
                planes.append((f"continuo_{k}", plan, inicio))
        planes.append(("ir_y_volver", plan_ir_y_volver(robot, inicio, robot.puntos_demo[2]), inicio))
        planes.append(("cosecha", plan_cosecha(robot, inicio, robot.puntos_demo[:4])[0], inicio))
        for archivo in ("demo-cosecha", "ejemplo-grabacion"):
            programa = prog.cargar(f"{CARPETA_REPO}/programas/{archivo}.yaml", robot)
            plan = (plan_puntos if programa.tipo == "puntos" else plan_grabacion)(robot, inicio, programa)[0]
            planes.append((archivo, plan, inicio))
    return planes


def ruido_uniforme(semilla, ms=20.0):
    azar = random.Random(semilla)
    return lambda: azar.uniform(-ms, ms)


PERFILES = {
    "reloj_exacto": dict(),
    "reloj_con_variacion_de_20ms": dict(ruido=ruido_uniforme(7)),
    "temporizador_1ms_antes_de_tiempo": dict(acortar=1),
    "ambas_cosas": dict(ruido=ruido_uniforme(11, 12.0), acortar=1),
}


def reproducir(robot, plan, inicio, velocidad=100, ruido=None, acortar=0):
    """Reproduce el plan como la ventana: cada paso espera su intervalo; la lectura del reloj lleva `ruido` (ms) y
    nunca retrocede. Devuelve (guardia, enlace, pasos con servos). Si la guardia niega, la excepción sale."""
    g, enlace, reloj = armar(robot, t=0.0)
    g.enviar(*inicio)
    nominal, leida, con_servos = 5000.0, 0.0, 0   # EL PLAN ARRANCA 5 s DESPUÉS DE LA ORDEN DE INICIO
    for paso in plan:
        if paso.servos is not None:
            leida = max(leida, nominal + (ruido() if ruido else 0.0))
            reloj.t = leida / 1000.0
            g.enviar(*paso.servos)
            con_servos += 1
        nominal += max(0, intervalo_ms(robot, paso, velocidad) - acortar)
    return g, enlace, con_servos


@pytest.mark.parametrize("perfil", list(PERFILES))
@pytest.mark.parametrize("nombre", NOMBRES_ROBOT)
def test_gs07_una_reproduccion_normal_a_velocidad_100_pasa_completa_sin_una_negativa(nombre, perfil):
    robot = robot_de(nombre)
    planes = planes_reales(nombre)
    assert len(planes) >= 3
    for identificador, plan, inicio in planes:
        # EL SIMULADOR DEBE USAR EL MISMO RITMO QUE EL PEOR CASO REAL: PASOS DE HASTA EL TOPE CADA ms_por_paso
        g, enlace, con_servos = reproducir(robot, plan, inicio, 100, **PERFILES[perfil])
        assert len(enlace.salidas) == con_servos + 1, identificador
        assert not g.detenida, identificador


def test_gs07_los_planes_de_verdad_si_traen_pasos_del_tamano_del_tope(robot):
    """Que la prueba de arriba no sea trivial: hay pasos de ~15° y la punta corre mm entre pasos."""
    plan = dict((i, (p, s)) for i, p, s in planes_reales("3gdl"))["articular_ida"][0]
    saltos = [max(abs(a - b) for a, b in zip(x.servos, y.servos)) for x, y in zip(plan, plan[1:])]
    assert max(saltos) > 0.9 * robot.movimiento.salto_max_por_paso


@pytest.mark.parametrize("velocidad", [10, 25, 50, 99])
def test_gs07_a_velocidades_menores_tambien_pasa_completa(velocidad):
    robot = robot_de("3gdl")
    for identificador, plan, inicio in planes_reales("3gdl"):
        g, enlace, con_servos = reproducir(robot, plan, inicio, velocidad, **PERFILES["reloj_con_variacion_de_20ms"])
        assert len(enlace.salidas) == con_servos + 1, identificador


def test_gs07_los_planes_reales_tambien_pasan_por_la_funcion_unica_a_velocidad_100():
    for nombre in NOMBRES_ROBOT:
        robot = robot_de(nombre)
        for identificador, plan, inicio in planes_reales(nombre):
            informe = revisar_plan(robot, plan, inicio, velocidad=100)
            assert informe.ok, (nombre, identificador, informe.violacion)


def test_gs07_la_misma_reproduccion_30_por_ciento_mas_rapida_de_lo_que_el_servo_aguanta_si_se_niega(robot):
    """El tope por tiempo no es de adorno: el plan articular largo a velocidad «130 %» no cabe."""
    plan, inicio = dict((i, (p, s)) for i, p, s in planes_reales("3gdl"))["articular_ida"]
    with pytest.raises(OrdenNegada) as e:
        reproducir(robot, plan, inicio, velocidad=130)
    assert e.value.clave == "ritmo_excedido"
    assert revisar_plan(robot, plan, inicio, velocidad=130).violacion.clave == "ritmo_excedido"


def test_gs07_muchas_ordenes_chicas_muy_seguidas_se_niegan_con_la_guardia(r):
    g, enlace, reloj = armar(r)
    a = punto_medio(r)
    g.enviar(*a)
    negada = None
    for k in range(1, 100):
        reloj.t += 0.001
        try:
            g.enviar(*mueve(r, a, 1.0 * k))
        except OrdenNegada as e:
            negada = e
            break
    assert negada is not None and negada.clave == "ritmo_excedido" and negada.nueva
    assert len(enlace.salidas) < 25 and g.detenida


def test_gs07_dos_ordenes_con_la_misma_hora_el_segundo_paso_completo_no_sale(r):
    g, enlace, reloj = armar(r)
    a = punto_medio(r)
    g.enviar(*a)
    reloj.t += 1.0
    g.enviar(*mueve(r, a, 15))
    with pytest.raises(OrdenNegada) as e:  # MISMA HORA: dt = 0
        g.enviar(*mueve(r, a, 30))
    assert e.value.clave == "ritmo_excedido" and len(enlace.salidas) == 2


def test_gs07_la_hora_que_retrocede_no_da_tiempo_a_favor(r):
    g, enlace, reloj = armar(r)
    a = punto_medio(r)
    g.enviar(*a)
    reloj.t -= 10.0
    with pytest.raises(OrdenNegada) as e:
        g.enviar(*mueve(r, a, 15))
    assert e.value.clave == "ritmo_excedido"


def test_gs07_tras_una_pausa_larga_sale_un_paso_completo_pero_no_una_rafaga(r):
    g, enlace, reloj = armar(r)
    a = punto_medio(r)
    g.enviar(*a)
    reloj.t += 7200.0
    g.enviar(*mueve(r, a, 15))  # TRAS LA PAUSA, UN PASO COMPLETO: BIEN
    with pytest.raises(OrdenNegada) as e:  # PERO EL TIEMPO PARADO NO SE GUARDÓ PARA GASTARLO DESPUÉS
        g.enviar(*mueve(r, a, 30))
    assert e.value.clave == "ritmo_excedido"


def test_gs07_la_holgura_numerica_esta_declarada(robot):
    """Los números que el reporte declara: 5 % sostenido y 0.8 pasos de ráfaga; 300 °/s de ritmo máximo."""
    mov = robot.movimiento
    assert HOLGURA_RELATIVA == 0.05 and HOLGURA_RAFAGA_PASOS == 0.8
    assert mov.salto_max_por_paso / mov.ms_por_paso * 1000 * (1 + HOLGURA_RELATIVA) == pytest.approx(315)
    assert HOLGURA_RAFAGA_PASOS * mov.ms_por_paso == pytest.approx(40)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PROPIEDADES SOBRE SECUENCIAS AL AZAR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def corrida(data):
    """Una secuencia de órdenes al azar (buenas, límite y basura) con horas al azar. Devuelve (robot, enlace)."""
    robot = robot_de(data.draw(st.sampled_from(NOMBRES_ROBOT)))
    g, enlace, reloj = armar(robot)
    for _ in range(data.draw(st.integers(1, 40))):
        reloj.t += data.draw(st.sampled_from([0, 0.001, 0.005, 0.02, 0.04, 0.05, 0.05, 0.06, 0.2, 5, 3600]))
        base = g.ultima or punto_medio(robot)
        if data.draw(st.integers(0, 7)) == 0:
            orden = tuple(data.draw(st.one_of(
                st.lists(st.one_of(st.floats(), st.floats(0, 180), st.none(), st.booleans(), st.text(max_size=2)),
                         max_size=robot.gdl + 2),
                st.lists(st.floats(0, 180), max_size=robot.gdl + 2))))  # TAMBIÉN NÚMEROS BUENOS DE LA CANTIDAD EQUIVOCADA
        else:
            tope = data.draw(st.sampled_from([1, 5, 15, 16]))
            orden = tuple(b + data.draw(st.floats(-tope, tope)) for b in base)
        if g.detenida and data.draw(st.integers(0, 2)) == 0:
            g.reiniciar()
        try:
            g.enviar(*orden)
        except OrdenNegada:
            pass
    return robot, enlace


@PROPIEDADES
@given(data=st.data())
def test_inv01_todo_lo_que_la_guardia_dejo_salir_cumple_la_funcion_unica(data):
    robot, enlace = corrida(data)
    anterior, hora_anterior, deuda = None, None, 0.0
    for hora, servos in enlace.salidas:
        r = revisar_paso(robot, servos, anterior, None if hora_anterior is None else hora - hora_anterior, deuda)
        assert r.ok, r.violacion
        anterior, hora_anterior, deuda = r.servos, hora, r.deuda_ms


@PROPIEDADES
@given(data=st.data())
def test_gs02_gs03_y_gs11_nada_de_lo_que_salio_trae_numeros_malos_fuera_de_rango_ni_saltos(data):
    """Oráculo escrito aparte (sin revisar_paso): lo que llegó al enlace es finito, de la cantidad justa, en rango y sin salto."""
    robot, enlace = corrida(data)
    tope = robot.movimiento.salto_max_por_paso
    previa = None
    for _, servos in enlace.salidas:
        assert len(servos) == robot.gdl
        for s, j in zip(servos, robot.articulaciones):
            assert type(s) is float and s == s and abs(s) != float("inf")
            assert j.servo.minimo - 0.01 <= s <= j.servo.maximo + 0.01
            assert j.servo.minimo <= int(round(s)) <= j.servo.maximo
        if previa is not None:
            assert max(abs(a - b) for a, b in zip(servos, previa)) <= tope + 1e-6
        previa = servos


@PROPIEDADES
@given(data=st.data())
def test_gs07_en_cualquier_ventana_de_lo_que_salio_el_giro_cabe_en_el_tiempo_mas_la_rafaga(data):
    """Oráculo aparte: entre cualquier par de órdenes que salieron, el giro total por servo no pasa de
    1.05 * (tiempo + 40 ms) * tope / ms_por_paso."""
    robot, enlace = corrida(data)
    mov = robot.movimiento
    salidas = enlace.salidas
    for i in range(len(salidas)):
        for j in range(i + 1, len(salidas)):
            tramo = [s for _, s in salidas[i:j + 1]]
            dt_ms = (salidas[j][0] - salidas[i][0]) * 1000
            for k in range(robot.gdl):
                giro = sum(abs(a[k] - b[k]) for a, b in zip(tramo, tramo[1:]))
                permitido = (1 + HOLGURA_RELATIVA) * (max(dt_ms, 0) + HOLGURA_RAFAGA_PASOS * mov.ms_por_paso) \
                    * mov.salto_max_por_paso / mov.ms_por_paso
                assert giro <= permitido + 1e-6, (i, j, k, giro, permitido)


@PROPIEDADES
@given(data=st.data())
def test_gs09_con_cualquier_basura_la_guardia_solo_sale_o_levanta_orden_negada(data):
    robot = robot_de(data.draw(st.sampled_from(NOMBRES_ROBOT)))
    g, enlace, _ = armar(robot)
    cosas = st.recursive(st.one_of(st.none(), st.booleans(), st.integers(), st.floats(), st.text(max_size=3), st.binary(max_size=2)),
                         lambda hijos: st.lists(hijos, max_size=3) | st.tuples(hijos) | st.dictionaries(st.text(max_size=1), hijos, max_size=2),
                         max_leaves=6)
    for _ in range(data.draw(st.integers(1, 8))):
        orden = data.draw(st.lists(cosas, max_size=robot.gdl + 2))
        if g.detenida:
            g.reiniciar()
        try:
            g.enviar(*orden)
        except OrdenNegada as e:
            assert e.clave != "guardia_fallo", f"la revisión falló con {orden!r}"
    assert all(len(s) == robot.gdl for _, s in enlace.salidas)
