"""Las reglas de un paso válido, en UN solo lugar (INV1).

Las usan la guardia de salida (cosechador/hardware/guardia.py), el explorador y las pruebas: nadie más opina de qué
es un paso válido. Es pura (sin Qt, sin placa, sin reloj propio): recibe todo lo que necesita.

Un paso es válido cuando cumple, en este orden:
1. FORMA: una lista o tupla (o arreglo de numpy de una dimensión).
2. CANTIDAD: exactamente tantos servos como tiene el robot (`servos_en_rango` usa `zip` y daría por buena una
   orden corta).
3. NÚMEROS: cada valor es un número real finito (ni NaN, ni infinito, ni None, ni texto, ni booleano).
4. RANGO: cada servo cae en su rango (con la misma tolerancia que el planeador) Y el entero que de verdad llega a la
   placa también cae. Los topes son datos del robot (config.Servo), no números de este archivo.
5. SALTO: contra el paso anterior, ningún servo gira más de `movimiento.salto_max_por_paso`.
6. RITMO: contra el tiempo transcurrido desde el paso anterior, el giro no rebasa lo que el servo aguanta por
   segundo (salto_max_por_paso cada ms_por_paso).

Del avance de la punta (mm entre dos pasos seguidos) solo se MIDE y se reporta: no rechaza nada (su tope no está
aprobado).
"""
import math
import numbers
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from cosechador.hardware.enlace import angulo_de_placa
from cosechador.movimiento.trayectorias import Paso, posicion
from cosechador.robot.cinematica import primer_servo_fuera
from cosechador.robot.cinematica_inversa import salto

# UN SERVO NO GIRA MÁS QUE EL TOPE POR PASO; ESTO ABSORBE SOLO EL RUIDO DE COMA FLOTANTE (EL PLANEADOR PARTE LOS
# TRAMOS CON UN 1e-9 DE MARGEN, ASÍ QUE PUEDE DEJAR UN PASO 1e-8 ° ARRIBA DEL TOPE).
TOLERANCIA_SALTO = 1e-6          # grados

# TOPE POR TIEMPO (GS7). LA VELOCIDAD MÁXIMA ES salto_max_por_paso CADA ms_por_paso (300 °/s CON LOS DATOS ACTUALES).
# DOS HOLGURAS, CADA UNA PARA UNA COSA DISTINTA:
HOLGURA_RELATIVA = 0.05          # 5 % SOSTENIDO: UN TEMPORIZADOR QUE DISPARA ~1 ms ANTES DE TIEMPO CADA PASO NO ACUMULA NEGATIVAS
HOLGURA_RAFAGA_PASOS = 0.8       # RÁFAGA: CUÁNTO PUEDE ADELANTARSE LA ORDEN AL RITMO, EN PASOS COMPLETOS (40 ms CON 50 ms/PASO).
#                                  ABSORBE QUE EL RELOJ SE LEA CON VARIACIÓN (EL TIEMPO DE DIBUJAR ANTES DE ENVIAR). SE QUEDA
#                                  < 1 PASO A PROPÓSITO: DOS ÓRDENES COMPLETAS CON LA MISMA HORA NO PASAN.


class Violacion(NamedTuple):
    """Por qué un paso no es válido. Todo lo que la persona o la pantalla necesitan, sin reinterpretar nada."""
    clave: str            # ESTABLE (no cambia con el texto): no_es_lista | cantidad_servos | no_es_numero | fuera_de_rango |
    #                       salto_excedido | ritmo_excedido | guardia_fallo | enlace_fallo
    motivo: str           # EN ESPAÑOL, LEGIBLE
    servo: int | None = None       # NÚMERO DEL SERVO (1 = el primero) O None SI NO ES DE UN SERVO
    pedido: float | None = None    # LO QUE SE PIDIÓ (ángulo, grados de giro o cantidad de servos)
    tope: float | None = None      # LO QUE SE PERMITÍA
    paso: int | None = None        # EN UN PLAN: ÍNDICE DEL PASO (desde 0)
    detalle: str | None = None     # TEXTO TÉCNICO (p. ej. la excepción de una falla de la guardia)

    def en_el_paso(self, indice):
        return self._replace(paso=indice)


class Revision(NamedTuple):
    """Resultado de revisar un paso: la violación (None si es válido), los servos ya saneados y la deuda de ritmo."""
    violacion: Violacion | None
    servos: tuple | None
    deuda_ms: float

    @property
    def ok(self):
        return self.violacion is None


@dataclass(frozen=True)
class InformePlan:
    """Resultado de revisar un plan: la primera violación (si hay) y cuánto avanzó la punta entre pasos (mm)."""
    violacion: Violacion | None
    avance_max_mm: float = 0.0       # LO MÁS QUE SE MOVIÓ LA PUNTA ENTRE DOS PASOS SEGUIDOS. SOLO SE MIDE
    avance_en_paso: int | None = None
    pasos_revisados: int = 0

    @property
    def ok(self):
        return self.violacion is None


def _a_float(valor):
    """El valor como float, o None si no es un número real finito (los booleanos NO cuentan como número)."""
    if isinstance(valor, bool) or not isinstance(valor, numbers.Real):
        return None
    try:
        x = float(valor)
    except (OverflowError, ValueError, TypeError):  # UN ENTERO ENORME NO CABE EN UN float
        return None
    return x if math.isfinite(x) else None


def _como_tupla(servos):
    if isinstance(servos, (list, tuple)):
        return tuple(servos)
    if isinstance(servos, np.ndarray) and servos.ndim == 1:
        return tuple(servos.tolist())
    return None


def _g(x):
    return f"{x:g}"


def _revisar_valores(robot, servos):
    """Forma, cantidad, números, rango y entero de la placa. Devuelve (violacion, servos saneados)."""
    crudo = _como_tupla(servos)
    if crudo is None:
        return Violacion("no_es_lista", f"La orden debe ser una lista de {robot.gdl} ángulos de servo "
                                        f"(llegó {type(servos).__name__}).", pedido=None, tope=robot.gdl), None
    if len(crudo) != robot.gdl:
        return Violacion("cantidad_servos", f"La orden trae {len(crudo)} servos y el robot tiene {robot.gdl}.",
                         pedido=len(crudo), tope=robot.gdl), None
    limpios = []
    for i, valor in enumerate(crudo):
        x = _a_float(valor)
        if x is None:
            return Violacion("no_es_numero", f"El servo {i + 1} no es un número válido (llegó {valor!r}).",
                             servo=i + 1), None
        limpios.append(x)
    limpios = tuple(limpios)
    articulaciones = robot.articulaciones
    i = primer_servo_fuera(robot, limpios)
    if i is None:  # LO QUE DE VERDAD LLEGA A LA PLACA SON ENTEROS: TAMBIÉN ELLOS DEBEN CABER, SIN TOLERANCIA
        enteros = tuple(float(angulo_de_placa(s)) for s in limpios)
        i = primer_servo_fuera(robot, enteros, tolerancia=0)
    if i is not None:
        s = articulaciones[i].servo
        return Violacion("fuera_de_rango",
                         f"El servo {i + 1} pidió {_g(limpios[i])}° y su rango es {_g(s.minimo)}° a {_g(s.maximo)}°.",
                         servo=i + 1, pedido=limpios[i],
                         tope=s.maximo if limpios[i] > s.maximo else s.minimo), None
    return None, limpios


def _revisar_salto(robot, anterior, servos):
    tope = robot.movimiento.salto_max_por_paso
    for i, (a, b) in enumerate(zip(anterior, servos)):
        giro = abs(a - b)
        if giro > tope + TOLERANCIA_SALTO:
            return Violacion("salto_excedido",
                             f"El servo {i + 1} giraría {_g(round(giro, 3))}° de una vez y lo más que puede girar "
                             f"por paso es {_g(tope)}°.", servo=i + 1, pedido=giro, tope=tope)
    return None


def _revisar_ritmo(robot, anterior, servos, dt_s, deuda_ms):
    """El giro contra el tiempo. Devuelve (violacion, deuda nueva).

    Cada paso «cuesta» el tiempo que el servo necesita para girar eso a la velocidad máxima (más holgura relativa).
    Si llega antes de ese tiempo, la diferencia se acumula como DEUDA; la deuda se paga con el tiempo que pasa y
    NUNCA se guarda a favor (una pausa larga no da permiso para una ráfaga después). Más deuda que la ráfaga
    permitida: se niega.
    """
    mov = robot.movimiento
    dt_ms = max(0.0, dt_s) * 1000.0
    giro = salto(anterior, servos)
    costo_ms = giro / mov.salto_max_por_paso * mov.ms_por_paso / (1.0 + HOLGURA_RELATIVA)
    nueva = max(0.0, deuda_ms + costo_ms - dt_ms)
    rafaga_ms = HOLGURA_RAFAGA_PASOS * mov.ms_por_paso
    if nueva <= rafaga_ms + 1e-9:
        return None, nueva
    # LOS GRADOS QUE ESE TIEMPO PERMITÍA GIRAR (PARA DECIRLE A LA PERSONA CUÁNTO ERA EL TOPE)
    permitido = max(0.0, (rafaga_ms - deuda_ms + dt_ms)) * (1.0 + HOLGURA_RELATIVA) * mov.salto_max_por_paso / mov.ms_por_paso
    servo = max(range(len(servos)), key=lambda i: abs(anterior[i] - servos[i]))
    tasa = mov.salto_max_por_paso / mov.ms_por_paso * 1000.0
    return Violacion("ritmo_excedido",
                     f"El servo {servo + 1} giraría {_g(round(giro, 3))}° en {_g(round(dt_ms, 3))} ms y en ese "
                     f"tiempo solo puede girar {_g(round(permitido, 3))}° (máximo {_g(round(tasa))} °/s).",
                     servo=servo + 1, pedido=giro, tope=permitido), deuda_ms


def revisar_paso(robot, servos, anterior=None, dt_s=None, deuda_ms=0.0):
    """LA función que decide si un paso es válido (INV1). Devuelve una Revision; nunca devuelve una excepción por
    una orden mala (las órdenes malas son su trabajo): solo falla por un `anterior` o `dt_s` mal armados.

    servos: lo que se quiere mandar (cualquier cosa; se revisa la forma).
    anterior: los servos del paso previo que SÍ salió (ya válidos), o None si no hay (primer paso: solo forma,
        cantidad, números y rango).
    dt_s: segundos desde ese paso previo, o None si no se conoce (solo se mide el salto, no el ritmo).
    deuda_ms: la deuda de ritmo que arrastraba el paso previo (ver _revisar_ritmo).
    """
    violacion, limpios = _revisar_valores(robot, servos)
    if violacion is not None:
        return Revision(violacion, None, deuda_ms)
    if anterior is None:
        return Revision(None, limpios, 0.0)
    previo = tuple(anterior)
    if len(previo) != robot.gdl or any(_a_float(v) is None for v in previo):
        raise ValueError(f"'anterior' debe tener {robot.gdl} números válidos (llegó {anterior!r}).")
    violacion = _revisar_salto(robot, previo, limpios)
    if violacion is not None:
        return Revision(violacion, None, deuda_ms)
    if dt_s is None:
        return Revision(None, limpios, deuda_ms)
    if _a_float(dt_s) is None or _a_float(deuda_ms) is None:
        raise ValueError(f"'dt_s' y 'deuda_ms' deben ser números finitos (llegó {dt_s!r}, {deuda_ms!r}).")
    violacion, nueva = _revisar_ritmo(robot, previo, limpios, float(dt_s), float(deuda_ms))
    if violacion is not None:
        return Revision(violacion, None, deuda_ms)
    return Revision(None, limpios, nueva)


def avance_punta_mm(robot, servos_a, servos_b):
    """Cuánto se mueve la punta (mm, en línea recta) al pasar de una pose a otra. Solo mide; no rechaza nada."""
    return float(np.linalg.norm(posicion(robot, servos_b) - posicion(robot, servos_a)))


def intervalo_ms(robot, paso, velocidad=100):
    """Cuánto espera la app tras un paso antes del siguiente (ms). Es la misma cuenta que hace la ventana al
    armar el temporizador; `velocidad` es la general, de 10 a 100 %."""
    ms = paso.ms if paso.ms is not None else (robot.movimiento.ms_por_paso if paso.servos is not None else 0)
    return max(1, int(round(ms * 100 / velocidad)))


def revisar_plan(robot, plan, servos_inicio, velocidad=100):
    """Revisa un plan completo desde `servos_inicio`: devuelve un InformePlan con la PRIMERA violación (con su
    índice de paso) y, aparte, el mayor avance de la punta entre dos pasos con servos seguidos.

    plan: lista de Paso (o de tuplas de servos, que se toman con la duración de un paso normal). Los tiempos se
    arman como los reproduce la app a esa velocidad: cada paso espera su `ms` antes del siguiente. El primer paso
    con servos no se mide por ritmo (no se sabe cuánto hace que se mandó `servos_inicio`), solo por salto.
    """
    violacion, inicio = _revisar_valores(robot, servos_inicio)
    if violacion is not None:
        return InformePlan(violacion.en_el_paso(-1))
    anterior, t_anterior, t, deuda = inicio, None, 0, 0.0
    avance_max, avance_en, revisados = 0.0, None, 0
    for i, p in enumerate(plan):
        paso = p if isinstance(p, Paso) else Paso(servos=p)
        if paso.servos is not None:
            dt = None if t_anterior is None else (t - t_anterior) / 1000.0
            r = revisar_paso(robot, paso.servos, anterior, dt, deuda)
            if not r.ok:
                return InformePlan(r.violacion.en_el_paso(i), avance_max, avance_en, revisados)
            avance = avance_punta_mm(robot, anterior, r.servos)
            if avance > avance_max:
                avance_max, avance_en = avance, i
            anterior, deuda, t_anterior, revisados = r.servos, r.deuda_ms, t, revisados + 1
        t += intervalo_ms(robot, paso, velocidad)
    return InformePlan(None, avance_max, avance_en, revisados)
