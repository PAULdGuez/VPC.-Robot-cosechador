"""«Reemplazar» con la punta en camino, cuando no se puede ir directo al punto pedido: punto de paso.

D7 (aprobado por Paul el 25-nov-2026). Lo que alcanza la punta no es una figura convexa: la recta desde donde VA la punta
hasta el punto pedido puede salirse del alcance aunque el rojo del slider (que se mide desde el punto AL QUE va) diga que sí
se puede. Antes el planeador se negaba y el slider volvía solo. Ahora:

  D7.1  la orden no se pierde: la punta termina lo que iba haciendo y luego va; el slider se queda donde se soltó
  D7.2  sale UN aviso que dice que no se pudo ir directo y que irá al terminar
  D7.3  si sí se puede ir directo, todo pasa igual que antes
  D7.4  «En cola» y «Cancelar» no cambian
"""
import math

import pytest

import cosechador.ui.ventana as V
import explorador as ex
from cosechador.movimiento.trayectorias import posicion
from cosechador.robot import cargar_robot
from explorador import Gesto
from helpers_ventana import sin_dialogos_de_verdad

ROBOT = cargar_robot()
PUNTA = Gesto("elegir", "PESTANAS_MOVIMIENTO", 0.5)
REEMPLAZAR, EN_COLA, CANCELAR = (Gesto("elegir", "SELECTOR_EN_CAMINO", v) for v in (0.1, 0.5, 0.9))
A_NEGARSE = Gesto("elegir", "SELECTOR_POSTURA", 0.9)
LLEGAR = Gesto("llegar", "", 0.0, 0, 0.3)
FINO = Gesto("tecla", "ARRASTRE_FINO", 0.0, 0, 0.0)   # UN GESTO QUE NO EXISTE: LO ATRAPA _con_arrastre_fino
# LOS DOS HALLAZGOS DE LA EXPLORACIÓN LARGA DEL 25-nov-2026: (lo que deja la punta en camino, el arrastre que rebotaba)
CASOS = {
    "girar despacio": ((PUNTA, REEMPLAZAR, Gesto("arrastrar", "SLIDER_Z", 0.5457821488113972, 477770, 0.3),
                        Gesto("arrastrar", "SLIDER_X", 0.8835869371580177, 34855, 0.3)),
                       Gesto("arrastrar", "SLIDER_Y", 0.14673451377324587, 607881, 0.0)),
    "negarse": ((A_NEGARSE, PUNTA, Gesto("arrastrar", "SLIDER_Y", 0.38591912681792206, 23154, 0.0), REEMPLAZAR,
                 Gesto("esperar", "", 0.0, 0, 0.05), Gesto("arrastrar", "SLIDER_Z", 0.734530518561858, 528255, 0.05)),
                Gesto("arrastrar", "SLIDER_X", 0.26604663315456467, 620442, 0.0)),
}


@pytest.fixture
def sesion():
    s = ex.Sesion(ROBOT)
    with sin_dialogos_de_verdad():
        yield s
    s.cerrar()


def _avisos_de_paso(w, desde=0):
    return [e for e in w.x.eventos[desde:] if e.origen == "Jog" and e.datos.get("de_paso")]


def _con_la_punta_en_camino(s, caso, modo=REEMPLAZAR):
    antes, ultimo = CASOS[caso]
    for g in antes:
        s.hacer(modo if g is REEMPLAZAR else g)
    assert s.w._destino_jog is not None, "la preparación ya no deja la punta en camino: estas pruebas no probarían nada"
    return ultimo


@pytest.mark.parametrize("caso", CASOS)
def test_d7_1_si_no_se_puede_ir_directo_la_punta_termina_su_tramo_y_luego_va(sesion, caso):
    s, w = sesion, sesion.w
    ultimo = _con_la_punta_en_camino(s, caso)
    eje, enviadas, iba_a = "XYZ".index(ultimo.control[-1]), len(w.x.enviados), tuple(w._destino_jog)
    slider, pendiente = w.sliders_xyz[eje], len(w.cola)
    # DÓNDE QUEDA EL SLIDER AL SOLTARLO Y, CON ESO, EL PUNTO PEDIDO: A DONDE LO LLEVA LA PERSONA (LA MISMA CUENTA QUE HACE EL
    # ARRASTRE DEL EXPLORADOR) O, SI ANTES HAY ROJO, EL BORDE DEL ROJO. SE CALCULAN AQUÍ, ANTES DE LA ORDEN Y SIN PREGUNTARLE
    # DESPUÉS A LA VENTANA, PARA QUE UN SLIDER QUE VOLVIÓ SOLO NO PASE POR BUENO
    soltado = slider.acotar(slider.minimum() + round(ultimo.v * (slider.maximum() - slider.minimum())), iba_a[eje])
    pedido = [*iba_a[:eje], float(soltado), *iba_a[eje + 1:]]
    assert pendiente and abs(soltado - iba_a[eje]) > 50, "la preparación ya no deja un tramo pendiente ni pide otro punto"
    s.hacer(ultimo)                                  # ANTES DEL ARREGLO, AQUÍ LA SESIÓN LEVANTABA EL HALLAZGO «REBOTE»
    assert slider.value() == soltado, "el slider no se quedó donde se soltó"
    assert w._destino_jog is not None and math.dist(w._destino_jog, pedido) < 1e-6, "la ventana no va al punto pedido"
    assert len(w.cola) > w._tramo_jog[1], "la última orden salió directa: el caso ya no pide lo que estas pruebas protegen"
    s.hacer(LLEGAR)                                  # SI ALGO BRINCA, SE NIEGA O REVIENTA, LA SESIÓN LO LEVANTA COMO HALLAZGO
    assert not w.cola and not w.ocupado and not w.x.negadas
    assert math.dist(w.posicion, pedido) < 1.0, "la punta no terminó en el punto pedido"
    recorrido = [posicion(ROBOT, orden[1]) for orden in w.x.enviados[enviadas:]]
    assert min(math.dist(p, iba_a) for p in recorrido) < 1.0, "la punta no terminó lo que iba haciendo antes de ir"
    assert slider.value() == soltado


def _con_arrastre_fino(monkeypatch, slider, paradas, pixeles):
    """Hace que el gesto FINO arrastre `slider` por las `paradas`, sin soltarlo, como un ratón de verdad: un paso cada 5 mm y 8 ms. Va por la
    sesión, para que revise lo que pase. En `pixeles` queda, de cada paso, (a dónde se llevó, dónde quedó, qué contestó el
    planeador en cada intento, el último tramo anotado por la ventana)."""
    original, real, intentos = ex.Sesion._ejecutar, V.tramo_continuo, []

    def espia(*a, **k):
        plan, motivo = real(*a, **k)
        intentos.append(plan is not None)
        return plan, motivo

    def _ejecutar(self, g):
        if g != FINO:
            return original(self, g)
        slider.setSliderDown(True)
        for hasta in paradas:
            sale, n = slider.value(), abs(hasta - slider.value()) // 5
            for i in range(1, n + 1):
                valor = int(round(sale + (hasta - sale) * i / n))
                intentos.clear()
                slider.setSliderPosition(valor)
                pixeles.append((valor, slider.value(), tuple(intentos), self.w._tramo_jog))
                ex.avanzar(self.w, 0.008)
        slider.setSliderDown(False)
    monkeypatch.setattr(V, "tramo_continuo", espia)
    monkeypatch.setattr(ex.Sesion, "_ejecutar", _ejecutar)


@pytest.mark.parametrize("caso", CASOS)
def test_d7_1_en_un_arrastre_fino_el_slider_no_se_atora_donde_deja_de_poderse_ir_directo(sesion, monkeypatch, caso):
    """Un ratón manda muchos pasos chicos. Los primeros sí se pueden ir directo (y reemplazan lo pendiente); desde cierto
    punto ya no. Antes del arreglo el slider se atoraba ahí: en estos dos casos, 24 de 48 pasos y 195 de 203."""
    s, w = sesion, sesion.w
    ultimo = _con_la_punta_en_camino(s, caso)
    eje, iba_a = "XYZ".index(ultimo.control[-1]), tuple(w._destino_jog)
    slider = w.sliders_xyz[eje]
    hasta = slider.acotar(slider.minimum() + round(ultimo.v * (slider.maximum() - slider.minimum())), iba_a[eje])
    pixeles, eventos = [], len(w.x.eventos)
    _con_arrastre_fino(monkeypatch, slider, (hasta,), pixeles)
    s.hacer(FINO)
    # CONTROL: A MEDIO ARRASTRE HAY UN PASO QUE SALE DIRECTO Y EL SIGUIENTE YA NO PUEDE (AHÍ ERA DONDE SE ATORABA)
    directo_y_luego_no = [(a, b) for a, b in zip(pixeles, pixeles[1:]) if a[2] == (True,) and b[2][:1] == (False,)]
    assert len(pixeles) > 20 and directo_y_luego_no, "el caso ya no pasa de «se puede directo» a «no se puede» a medio arrastre"
    assert [pixel[1] for pixel in pixeles] == [pixel[0] for pixel in pixeles], "el slider se atoró a medio arrastre"
    assert len(_avisos_de_paso(w, eventos)) == 1, "un arrastre, un aviso"
    s.hacer(LLEGAR)
    assert math.dist(w.posicion, [*iba_a[:eje], float(hasta), *iba_a[eje + 1:]]) < 1.0 and not w.x.negadas


def test_d7_1_si_a_medio_arrastre_se_regresa_el_slider_el_tramo_se_traza_otra_vez_desde_el_punto_de_paso(sesion, monkeypatch):
    """El tramo que va detrás del punto de paso se traza otra vez con cada paso del mismo arrastre (como en «En cola»): si
    el slider pasa por un punto, sigue hasta el fondo y regresa a ese punto, lo pendiente queda igual que la primera vez
    que pasó por ahí. La punta no recorre el ir y venir del ratón."""
    s, w = sesion, sesion.w
    ultimo = _con_la_punta_en_camino(s, "girar despacio")
    eje, iba_a = "XYZ".index(ultimo.control[-1]), tuple(w._destino_jog)
    slider = w.sliders_xyz[eje]
    fondo = slider.acotar(slider.minimum() + round(ultimo.v * (slider.maximum() - slider.minimum())), iba_a[eje])
    regreso, pixeles = fondo + round((iba_a[eje] - fondo) / 4), []
    assert fondo < regreso - 40 < iba_a[eje] - 80, "el caso ya no deja un tramo largo hacia valores menores"
    _con_arrastre_fino(monkeypatch, slider, (regreso, fondo, regreso), pixeles)
    s.hacer(FINO)
    primera, segunda = (i for i, pixel in enumerate(pixeles) if pixel[0] == regreso)
    # CONTROL: DESDE LA PRIMERA VEZ QUE PASA POR AHÍ NO SE PUEDE IR DIRECTO (SI SE PUDIERA, «REEMPLAZAR» TIRARÍA LO PENDIENTE)
    assert all(pixel[2][:1] == (False,) for pixel in pixeles[primera:]), "el caso ya no se queda en «no se puede ir directo»"
    (_, pasos_1, _, sale_1), (_, pasos_2, _, sale_2) = pixeles[primera][3], pixeles[segunda][3]
    assert pasos_2 == pasos_1 and math.dist(sale_2, sale_1) < 1e-6, "el tramo no se trazó otra vez desde el punto de paso"
    assert math.dist(sale_2, w._destino_jog) > 40, "el último tramo sale del paso anterior del ratón, no del punto de paso"
    s.hacer(LLEGAR)
    assert math.dist(w.posicion, [*iba_a[:eje], float(regreso), *iba_a[eje + 1:]]) < 1.0 and not w.x.negadas


@pytest.mark.parametrize("caso", CASOS)
def test_d7_2_sale_un_solo_aviso_que_dice_que_no_se_pudo_ir_directo_y_que_ira_al_terminar(sesion, caso):
    s, w = sesion, sesion.w
    ultimo = _con_la_punta_en_camino(s, caso)
    eventos = len(w.x.eventos)
    s.hacer(ultimo)
    s.hacer(LLEGAR)
    avisos = _avisos_de_paso(w, eventos)
    assert len(avisos) == 1, [e.mensaje for e in avisos]
    assert avisos[0].nivel == "aviso" and avisos[0].mensaje.startswith("No se puede ir directo a (")
    assert avisos[0].mensaje.endswith("desde donde va la punta: termina el tramo que lleva y luego va.")
    negativas = [e.mensaje for e in w.x.eventos[eventos:] if e.origen == "Jog" and e.mensaje.startswith("No se mueve")]
    assert not negativas, negativas


@pytest.mark.parametrize("caso", CASOS)
def test_d7_2_si_el_arrastre_topa_con_el_rojo_tambien_sale_el_aviso_del_punto_de_paso(sesion, caso):
    """Si el arrastre entra al rojo, el aviso del tope se pide antes que el del punto de paso. Con un solo aviso por
    arrastre se quedaba el del tope y la punta iba de paso sin decirlo (112 de 133 órdenes en la medición del 25-nov-2026).
    Salen los dos, una vez cada uno, aunque se siga empujando contra el rojo."""
    s, w = sesion, sesion.w
    ultimo = _con_la_punta_en_camino(s, caso)
    eje, iba_a = "XYZ".index(ultimo.control[-1]), tuple(w._destino_jog)
    slider, eventos = w.sliders_xyz[eje], len(w.x.eventos)
    extremo = slider.minimum() if ultimo.v < 0.5 else slider.maximum()
    tope = slider.acotar(extremo, iba_a[eje])
    assert tope != extremo and abs(tope - iba_a[eje]) > 50, "el caso ya no tiene rojo de ese lado: no probaría nada"
    slider.setSliderDown(True)
    for _ in range(3):                               # UN SALTO HASTA EL EXTREMO Y DOS EMPUJONES MÁS CONTRA EL ROJO, SIN SOLTAR
        slider.setSliderPosition(extremo)
        ex.avanzar(w, 0.008)
    slider.setSliderDown(False)
    jog = [e for e in w.x.eventos[eventos:] if e.origen == "Jog"]
    assert [bool(e.datos.get("de_paso")) for e in jog] == [False, True], [e.mensaje for e in jog]
    assert "tope_mm" in jog[0].datos and slider.value() == tope
    s.hacer(LLEGAR)
    assert math.dist(w.posicion, [*iba_a[:eje], float(tope), *iba_a[eje + 1:]]) < 1.0 and not w.x.negadas


def test_d7_2_si_el_aviso_del_punto_de_paso_sale_primero_el_del_tope_ya_no_sale(sesion, monkeypatch):
    """Lo aprobado no cambia: en un arrastre fino el aviso del punto de paso sale antes de llegar al rojo, y al topar
    después con el rojo ya no sale el del tope. Un arrastre no se llena de avisos."""
    s, w = sesion, sesion.w
    ultimo = _con_la_punta_en_camino(s, "girar despacio")
    eje, iba_a = "XYZ".index(ultimo.control[-1]), tuple(w._destino_jog)
    slider = w.sliders_xyz[eje]
    tope = slider.acotar(slider.minimum(), iba_a[eje])
    assert slider.minimum() < tope - 60, "el caso ya no tiene rojo de ese lado: no probaría nada"
    pixeles, eventos = [], len(w.x.eventos)
    _con_arrastre_fino(monkeypatch, slider, (tope - 60,), pixeles)      # 60 mm DENTRO DEL ROJO
    s.hacer(FINO)
    jog = [e for e in w.x.eventos[eventos:] if e.origen == "Jog"]
    assert [bool(e.datos.get("de_paso")) for e in jog] == [True], [e.mensaje for e in jog]
    assert slider.value() == tope and any(pixel[0] < tope for pixel in pixeles), "el arrastre no llegó a entrar al rojo"


def test_d7_2_cada_arrastre_que_va_de_paso_da_su_aviso(sesion, monkeypatch):
    """Una vez por arrastre, no una vez para siempre: soltar el slider y arrastrarlo otra vez hacia otro punto al que
    tampoco se puede ir directo es otra orden, y avisa otra vez."""
    s, w = sesion, sesion.w
    ultimo = _con_la_punta_en_camino(s, "girar despacio")
    eje, iba_a = "XYZ".index(ultimo.control[-1]), tuple(w._destino_jog)
    slider = w.sliders_xyz[eje]
    fondo = slider.acotar(slider.minimum() + round(ultimo.v * (slider.maximum() - slider.minimum())), iba_a[eje])
    regreso = fondo + round((iba_a[eje] - fondo) / 4)
    paradas, pixeles, eventos = [regreso], [], len(w.x.eventos)
    _con_arrastre_fino(monkeypatch, slider, paradas, pixeles)
    s.hacer(FINO)
    assert len(_avisos_de_paso(w, eventos)) == 1
    paradas[:], primer_arrastre = [fondo], len(pixeles)
    s.hacer(FINO)
    segundo = pixeles[primer_arrastre:]
    assert segundo and all(pixel[2][:1] == (False,) for pixel in segundo), "control: en el segundo arrastre ya se puede ir directo"
    assert len(_avisos_de_paso(w, eventos)) == 2, "el segundo arrastre fue de paso y no avisó"


def test_d7_3_si_se_puede_ir_directo_todo_pasa_igual_que_antes(sesion):
    s, w = sesion, sesion.w
    for g in (PUNTA, REEMPLAZAR, Gesto("arrastrar", "SLIDER_Z", 0.55, 7, 0.1)):
        s.hacer(g)
    assert w._destino_jog is not None, "la preparación ya no deja la punta en camino"
    eventos = len(w.x.eventos)
    s.hacer(Gesto("arrastrar", "SLIDER_X", 0.55, 8, 0.0))
    assert w._destino_jog is not None and not _avisos_de_paso(w, eventos)
    assert len(w.cola) <= w._tramo_jog[1], "quedó pendiente el tramo anterior: «Reemplazar» ya no reemplaza"
    pedido = tuple(w._destino_jog)
    s.hacer(LLEGAR)
    assert math.dist(w.posicion, pedido) < 1.0 and not _avisos_de_paso(w)


@pytest.mark.parametrize("modo", (EN_COLA, CANCELAR), ids=("en cola", "cancelar"))
@pytest.mark.parametrize("caso", CASOS)
def test_d7_4_en_cola_y_cancelar_siguen_sin_el_aviso_del_punto_de_paso(sesion, caso, modo):
    s, w = sesion, sesion.w
    ultimo = _con_la_punta_en_camino(s, caso, modo)
    s.hacer(ultimo)
    s.hacer(LLEGAR)
    assert not _avisos_de_paso(w), "el punto de paso es solo de «Reemplazar»"


@pytest.mark.parametrize("modo", (EN_COLA, CANCELAR), ids=("en cola", "cancelar"))
def test_d7_4_en_cola_y_cancelar_si_el_planeador_se_niega_se_niegan_como_antes(sesion, monkeypatch, modo):
    """En esos dos modos el planeador no se niega con la punta en camino (por eso D7 era solo de «Reemplazar»). Para ver que
    el punto de paso no se coló ahí, se siembra una negativa: en cada orden, el primer intento del planeador
    falla; un segundo intento (eso sería el punto de paso) sí saldría, y no debe haberlo."""
    s, w = sesion, sesion.w
    ultimo = _con_la_punta_en_camino(s, "girar despacio", modo)
    real, orden, intentos = V.tramo_continuo, w._jog_cartesiano, []

    def una_orden(*a, **k):                          # CADA ORDEN DE LA PUNTA EMPIEZA SU CUENTA DE INTENTOS
        intentos.append(0)
        return orden(*a, **k)

    def se_niega_el_primer_intento(*a, **k):
        intentos[-1] += 1
        return (None, "pasa por una zona sembrada para la prueba") if intentos[-1] == 1 else real(*a, **k)

    monkeypatch.setattr(w, "_jog_cartesiano", una_orden)
    monkeypatch.setattr(V, "tramo_continuo", se_niega_el_primer_intento)
    eventos, pendiente = len(w.x.eventos), list(w.cola)
    with pytest.raises(ex.Hallazgo) as rebote:       # LA NEGATIVA DE SIEMPRE: EL SLIDER VUELVE Y LA SESIÓN LO REPORTA
        s.hacer(ultimo)
    assert intentos and set(intentos) == {1} and rebote.value.clase == "rebote", (intentos, rebote.value)
    assert not _avisos_de_paso(w, eventos), "el punto de paso es solo de «Reemplazar»"
    negativas = [e for e in w.x.eventos[eventos:] if e.origen == "Jog" and e.mensaje.startswith("No se mueve")]
    assert len(negativas) == 1 and "zona sembrada" in negativas[0].mensaje
    assert list(w.cola) == pendiente[len(pendiente) - len(w.cola):], "la negativa tocó lo que ya iba en camino"
