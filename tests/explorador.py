"""Explorador al azar: usa la ventana REAL como la usaría alguien con ganas de romperla y, después de CADA gesto, revisa
que nada se haya descompuesto. Es solo código de pruebas (no toca nada de cosechador/).

Cómo funciona
-------------
- Hypothesis (máquina de estados) decide los gestos. Cada gesto es un dato (`Gesto`: tipo, control, valor, semilla local,
  pausa) que se guarda en la BITÁCORA; con esa lista se repite lo mismo en otra ventana (`reproducir`) y se reduce a la
  lista más corta que lo sigue produciendo (`reducir`). Las macros (grabar, reproducir, editar, jog, cambiar de modo a media
  marcha) solo arman listas de gestos: lo que queda en la bitácora son siempre gestos sueltos.
- Los controles se DESCUBREN recorriendo la ventana (botones, sliders, selectores, casillas, campos, pestañas, listas): si
  cambia el acomodo de la pantalla, el explorador sigue sirviendo. Solo se toca lo que una persona podría tocar ahora
  (encendido y a la vista).
- El tiempo es falso y exacto: el temporizador del movimiento dispara los pasos a la hora a la que los dispararía la app
  (tests/helpers_ventana.py). Nunca se abre una ventana en la pantalla ni se toca nada de la persona: todo vive en una carpeta
  desechable y los diálogos se contestan por sus ganchos.

Lo que se revisa después de cada gesto (EX2) y la CLASE con la que se reporta:
  racha_de_avisos       3 avisos o más, idénticos y seguidos (salvo un mismo clic repetido por la persona)
  rebote                un slider que dejó entrar a donde el planeador se niega: el aviso «No se mueve» del jog
  brinco                la punta saltó más de BRINCO_ROJO_MM entre dos órdenes seguidas (entre BRINCO_REPORTE_MM y eso, solo
                        se mide y se reporta: el tope del defecto 4 no está aprobado)
  perdida               una grabación o un programa que desapareció o cambió sin preguntar
  excepcion             una excepción (en el gesto o dentro de un slot de Qt)
  orden_negada          lo que la guardia de salida (observadora) no habría dejado salir a la placa, con su clave
  no_termina            un movimiento que no acaba
  reposo                en reposo, la pantalla no quedó como un robot quieto: controles trabados o sliders que no dicen la pose

Se lanza (ver main): `python tests/explorador.py --largo` · en pytest: `EXPLORADOR_LARGO=1 pytest tests/test_explorador.py`.
"""
import argparse
import collections
import dataclasses
import hashlib
import os
import random
import shutil
import sys
import time
import traceback

CARPETA_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CARPETA_REPO not in sys.path:             # PARA LANZARLO COMO PROGRAMA (python tests/explorador.py), NO SOLO DESDE pytest
    sys.path.insert(0, CARPETA_REPO)

from hypothesis import HealthCheck, Phase, settings, strategies as st
from hypothesis import seed as semilla_de_hypothesis
from hypothesis.stateful import RuleBasedStateMachine, get_state_machine_test, initialize, rule
from PyQt5 import QtWidgets

from auditoria_interfaz import _es_interno
from cosechador.movimiento import programa as prog
from cosechador.movimiento.invariantes import avance_punta_mm
from cosechador.robot import cargar_robot
from cosechador.robot.cinematica import servos_en_rango
from helpers_ventana import (APP, DESCARTAR_NOMBRE, NoTermina, avanzar, borrar_desechable, carpeta_desechable, cerrar, correr,
                             sin_dialogos_de_verdad, ventana_aislada)
from rachas import MINIMO_DE_UNA_RACHA, rachas_de_avisos

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LO QUE SE DECLARA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DEL BRINCO DE LA PUNTA (defecto 4: 445 mm en un paso de un programa de puntos; su tope NO está aprobado):
BRINCO_REPORTE_MM = 300.0     # DESDE AQUÍ SE MIDE Y SE LISTA EN EL INFORME; NO PONE NADA EN ROJO
BRINCO_ROJO_MM = 900.0        # MÁS QUE ESTO ES UN HALLAZGO: NINGÚN PASO LEGAL (15° POR SERVO) LLEGA A TANTO (EL MÁXIMO MEDIDO CON
#                               20,000 POSES AL AZAR FUE 824 mm; test_ex02_el_tope_del_brinco_no_deja_fuera_un_paso_legal)
# CONTROLES QUE NO SE TOCAN (la ventana abre un diálogo del sistema sin gancho para contestarlo desde la prueba):
EXCLUIDOS = {
    "CARGA": "cargar_imagen abre QFileDialog.getOpenFileName y la ventana no tiene gancho para contestarlo",
    "BTN_EXPORTAR_REGISTRO": "exportar_registro abre QFileDialog.getSaveFileName sin gancho (y escribiría un archivo fuera de la carpeta desechable)",
}
# VER_VOLUMEN SOLO SE TOCA CON EL ROBOT QUIETO: su barra de progreso llama a processEvents y dispararía el temporizador REAL
VOLUMEN = "VER_VOLUMEN"
SEMBRADOS = ("demo-cosecha.yaml", "ejemplo-grabacion.yaml")   # LOS DOS PROGRAMAS DE EJEMPLO QUE EL REPO VERSIONA

# DEFECTOS REALES YA ENCONTRADOS Y SIN ARREGLAR (el explorador NO se afina para dejar de verlos): se siguen detectando, se cuentan
# y salen en el informe, pero no tumban la corrida (si no, un solo defecto abierto taparía todo lo demás que pueda haber). Cada uno
# tiene su prueba fija con el fallo esperado estricto (test_explorador.py, EX10). Al arreglarlo se borra de aquí y de allá.
@dataclasses.dataclass(frozen=True)
class Conocido:
    id: str
    clase: str
    contiene: str          # UN TROZO DEL DETALLE DEL HALLAZGO QUE LO IDENTIFICA
    motivo: str


CONOCIDOS = (
    Conocido("D7-rebote-reemplazar-con-la-punta-en-camino", "rebote", "punta en camino, modo «reemplazar», postura «despacio»",
             "riesgo DECLARADO de «Reemplazar» (aceptado al elegir los tres modos, 25-nov-2026): con la punta en camino el rojo "
             "de los sliders se mide desde el punto al que va, y el tramo nuevo sale en diagonal desde donde va la punta. Esa "
             "diagonal puede cruzar una zona que el robot no alcanza: el planeador se niega con un aviso veraz y el slider "
             "vuelve. Esperando a que llegue, o en «En cola» y «Cancelar», no pasa"),
)

TAMANOS = {   # (ejemplos de Hypothesis, pasos por ejemplo). Un «paso» puede ser una macro de varios gestos.
    "corto": (8, 12),
    "largo": (1600, 40),
}
SEMILLA = 20261125
PAUSAS = (0.0, 0.008, 0.05, 0.3, 1.5)            # LO QUE PASA DESPUÉS DE UN GESTO (s): UN RATÓN, UNA PERSONA, UNA PAUSA
TEXTOS_DE_CAMPO = ("0", "100", "-250", "900", "1500", "1e3", "", "abc", "1,5", "nan", "inf", "-0", "99999999999", "  12 ", "٣")
NOMBRES = (None, DESCARTAR_NOMBRE, "prueba", "Mi grabación", "prueba", "../fuera", "", "   ", "con/barra", "x" * 300,
           "ñandú", "<existente>", "demo-cosecha")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GESTOS Y HALLAZGOS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@dataclasses.dataclass(frozen=True)
class Gesto:
    """Un gesto de la persona sobre la ventana. Todo dato: se imprime, se guarda y se vuelve a ejecutar tal cual."""
    tipo: str                 # click | arrastrar | tecla | elegir | escribir | esperar | llegar
    control: str = ""         # objectName DEL CONTROL ("" si no es sobre un control)
    v: float = 0.0            # 0..1: a dónde se arrastra / qué renglón se elige
    s: int = 0                # SEMILLA LOCAL: de ella salen los detalles (cuántos eventos del ratón, qué se contesta a un diálogo)
    dt: float = 0.0           # SEGUNDOS QUE PASAN DESPUÉS

    def __str__(self):
        return f"{self.tipo}({self.control}, v={self.v:.3f}, s={self.s}, dt={self.dt})"


class Hallazgo(AssertionError):
    """Algo que se descompuso. Trae todo para repetirlo: la semilla de la corrida y la lista de gestos hasta aquí."""

    def __init__(self, clase, detalle, gestos=(), semilla=None):
        self.clase, self.detalle, self.gestos, self.semilla = clase, detalle, tuple(gestos), semilla
        super().__init__(f"[{clase}] {detalle}")

    def __str__(self):
        return (f"[{self.clase}] {self.detalle}\n  semilla: {self.semilla} · {len(self.gestos)} gestos:\n"
                + "\n".join(f"    {i + 1:>3}. {g!r}" for i, g in enumerate(self.gestos)))

    def como_prueba_fija(self, nombre="hallazgo"):
        """El texto de una prueba fija (ver tests/test_explorador.py, EX10) que repite este hallazgo."""
        gestos = "\n".join(f"    {g!r}," for g in self.gestos)
        return (f"    # {self.clase}: {self.detalle.splitlines()[0]}\n"
                f"    ({nombre!r}, {self.clase!r}, (\n{gestos}\n    )),\n")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DESCUBRIR LOS CONTROLES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
_TIPOS = ((QtWidgets.QAbstractSlider, "slider"), (QtWidgets.QAbstractButton, "boton"), (QtWidgets.QComboBox, "selector"),
          (QtWidgets.QLineEdit, "campo"), (QtWidgets.QTabWidget, "pestanas"), (QtWidgets.QListWidget, "lista"))


def descubrir(w):
    """Todos los controles que una persona puede tocar en la ventana, aunque ahora estén apagados o escondidos:
    {objectName: (tipo, widget)}. Un control sin nombre no se puede volver a encontrar, así que no entra (se cuenta aparte)."""
    encontrados, vistos = {}, set()
    for clase, tipo in _TIPOS:
        for c in w.findChildren(clase):
            if id(c) in vistos or _es_interno(c, w) or c.objectName().startswith("qt_"):
                continue
            vistos.add(id(c))
            if c.objectName():
                encontrados[c.objectName()] = (tipo, c)
    return dict(sorted(encontrados.items()))


def disponible(w, nombre, tipo, c):
    """¿Lo puede tocar ahora una persona? Encendido y a la vista (la pestaña y los paneles abiertos cuentan)."""
    if nombre in EXCLUIDOS or not c.isEnabled() or not c.isVisibleTo(w):
        return False
    if nombre == VOLUMEN and w.temporizador_movimiento.isActive():
        return False
    if tipo == "pestanas":
        return any(c.isTabEnabled(i) for i in range(c.count()))
    if tipo in ("selector",):
        return c.count() > 0
    if tipo == "lista":
        return c.count() > 0
    if tipo == "slider":
        return c.maximum() > c.minimum()
    return True


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% COBERTURA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Cobertura:
    """Qué se tocó y cuántas veces (EX8). Se acumula a lo largo de todas las ventanas de una corrida."""

    def __init__(self):
        self.gestos = 0
        self.ventanas = 0
        self.pasos_a_la_placa = 0
        self.segundos = 0.0
        self.controles = {}                         # nombre -> tipo (TODO LO DESCUBIERTO)
        self.tocados = collections.Counter()        # nombre -> veces
        self.tipos = collections.Counter()
        self.modos = collections.Counter()          # (postura, en camino) -> gestos hechos en ese modo
        self.estados = collections.Counter()        # pestaña, grabando, editando, en marcha…
        self.brincos = []                           # [(mm, origen, gesto)] > BRINCO_REPORTE_MM
        self.avance_max_mm = 0.0
        self.sin_nombre = 0
        self.conocidos = collections.Counter()      # defectos ABIERTOS ya conocidos (CONOCIDOS) que se volvieron a ver
        self.sin_efecto = 0                         # GESTOS SOBRE UN CONTROL APAGADO O ESCONDIDO: NO PASÓ NADA
        self.recorrido = []                         # TODOS los gestos, en orden (para comparar dos corridas: EX4)

    def ventana_nueva(self, w):
        self.ventanas += 1
        todos = descubrir(w)
        for nombre, (tipo, _) in todos.items():
            self.controles.setdefault(nombre, tipo)
        todas = {c for clase, _ in _TIPOS for c in w.findChildren(clase) if not _es_interno(c, w)}
        self.sin_nombre = max(self.sin_nombre, sum(1 for c in todas if not c.objectName()))

    def gesto(self, g, w):
        self.gestos += 1
        self.recorrido.append(g)
        self.tipos[g.tipo] += 1
        self.modos[(w.postura, w.en_camino)] += 1
        pestana = w.PESTANAS_MOVIMIENTO.tabText(w.PESTANAS_MOVIMIENTO.currentIndex())
        self.estados[f"pestaña {pestana}"] += 1
        for nombre, activo in (("grabando", w.grabadora.grabando), ("editor abierto", w.editando),
                               ("robot en marcha", w.temporizador_movimiento.isActive()),
                               ("velocidad menor a 100 %", w.velocidad < 100)):
            if activo:
                self.estados[nombre] += 1

    def nunca_tocados(self):
        return sorted(n for n in self.controles if n not in self.tocados and n not in EXCLUIDOS)

    def informe(self):
        L = ["══ EXPLORADOR: INFORME ══",
             f"Gestos: {self.gestos} en {self.ventanas} ventanas · {self.segundos:.1f} s"
             + (f" · {self.segundos / self.gestos * 1000:.0f} ms por gesto" if self.gestos else "")
             + f" · {self.pasos_a_la_placa} órdenes enviadas a la placa"]
        L.append(f"Gestos sin efecto (control apagado o escondido en ese momento): {self.sin_efecto}")
        L.append("Tipos de gesto: " + " · ".join(f"{k} {v}" for k, v in self.tipos.most_common()))
        L.append("Modos (postura / en camino), gestos hechos en cada uno:")
        for (postura, camino), n in sorted(self.modos.items()):
            L.append(f"    {postura:<9} / {camino:<10} {n}")
        L.append("Estados en los que se tocó: " + " · ".join(f"{k} {v}" for k, v in sorted(self.estados.items())))
        tocados = {n: v for n, v in self.tocados.items() if n in self.controles}
        L.append(f"Controles tocados: {len(tocados)} de {len(self.controles) - len([n for n in EXCLUIDOS if n in self.controles])}"
                 f" (descubiertos {len(self.controles)}, excluidos {len([n for n in EXCLUIDOS if n in self.controles])})")
        L.append("    " + " · ".join(f"{n} {v}" for n, v in sorted(tocados.items())))
        nunca = self.nunca_tocados()
        L.append(f"Controles NUNCA tocados ({len(nunca)}): " + (", ".join(nunca) if nunca else "ninguno"))
        if self.conocidos:
            L.append("Defectos abiertos ya conocidos que VOLVIÓ A VER: " + " · ".join(f"{k} ×{v}" for k, v in self.conocidos.items()))
        L.append("Excluidos a propósito: " + "; ".join(f"{n} ({m})" for n, m in EXCLUIDOS.items()))
        if self.sin_nombre:
            L.append(f"Controles sin nombre (no se pueden repetir, no se tocan): {self.sin_nombre}")
        L.append(f"Mayor avance de la punta entre dos órdenes seguidas: {self.avance_max_mm:.0f} mm "
                 f"(se lista desde {BRINCO_REPORTE_MM:.0f}; es hallazgo desde {BRINCO_ROJO_MM:.0f})")
        for mm, origen, gesto in sorted(self.brincos, reverse=True)[:5]:
            L.append(f"    brinco de {mm:.0f} mm · origen «{origen}» · gesto {gesto}")
        return "\n".join(L)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% UNA SESIÓN: UNA VENTANA Y SU BITÁCORA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _huellas(carpeta):
    """{archivo: huella} de lo que hay en la carpeta de programas."""
    return {f: hashlib.sha1(open(os.path.join(carpeta, f), "rb").read()).hexdigest()
            for f in sorted(os.listdir(carpeta)) if os.path.isfile(os.path.join(carpeta, f))}


def sembrar_programas(w):
    """Programas con los que arranca la ventana: los dos de ejemplo que el repo versiona, y nada más de la persona."""
    for nombre in SEMBRADOS:
        origen = os.path.join(CARPETA_REPO, "programas", nombre)
        if os.path.isfile(origen):
            shutil.copy(origen, os.path.join(w.carpeta_programas, nombre))
    w.recargar_programas()


class Sesion:
    """Una ventana aislada, su bitácora de gestos y las revisiones de después de cada gesto."""

    def __init__(self, robot, cobertura=None, semilla=None, tolerar_conocidos=False):
        self.cob = cobertura if cobertura is not None else Cobertura()
        self.semilla = semilla
        self.tolerar_conocidos = tolerar_conocidos
        self.carpeta = carpeta_desechable()
        self.w = ventana_aislada(robot, self.carpeta)
        sembrar_programas(self.w)
        self.w.x.autores = []                  # EL GESTO QUE ESCRIBIÓ CADA EVENTO DEL REGISTRO (PARALELO A x.eventos)
        self.w.x.gesto_actual = -1
        self.w.x.salidas = []                  # CUÁNTAS ÓRDENES HABÍAN SALIDO A LA PLACA AL ESCRIBIRSE CADA EVENTO (PARALELO)
        self.w.registro.suscribir(lambda e: (self.w.x.autores.append(self.w.x.gesto_actual),
                                             self.w.x.salidas.append(len(self.w.x.enviados))))
        self.gestos = []
        self.cob.ventana_nueva(self.w)

    def cerrar(self):
        cerrar(self.w)
        borrar_desechable(self.carpeta)

    # -- controles ---------------------------------------------------------------------------------------------
    def disponibles(self):
        return [(n, t, c) for n, (t, c) in descubrir(self.w).items() if disponible(self.w, n, t, c)]

    # -- ejecutar un gesto ---------------------------------------------------------------------------------------
    def hacer(self, g):
        """Hace el gesto y revisa. Levanta Hallazgo si algo se descompuso."""
        w, x = self.w, self.w.x
        antes = self._antes()
        self.gestos.append(g)
        x.gesto_actual = len(self.gestos) - 1
        self.cob.gesto(g, w)
        try:
            self._ejecutar(g)
        except NoTermina as e:
            raise self._hallazgo("no_termina", str(e)) from None
        except Hallazgo:
            raise
        except Exception as e:  # noqa: BLE001 -- AQUÍ SE BUSCAN JUSTO LAS EXCEPCIONES QUE NADIE ESPERABA
            raise self._hallazgo("excepcion", f"{type(e).__name__}: {e}\n{traceback.format_exc()}") from None
        finally:
            x.nombres.clear()
            x.confirmas.clear()
        self._revisar(antes, g)

    def _hallazgo(self, clase, detalle):
        return Hallazgo(clase, detalle, self.gestos, self.semilla)

    def _ejecutar(self, g):
        w, x = self.w, self.w.x
        if g.tipo == "esperar":
            avanzar(w, g.dt)
            return
        if g.tipo == "llegar":
            correr(w)
            avanzar(w, g.dt)
            return
        todos = descubrir(w)
        if g.control not in todos or not disponible(w, g.control, *todos[g.control]):
            self.cob.sin_efecto += 1     # LA PERSONA NO PUEDE TOCAR ESO AHORA: NO PASA NADA, PERO EL TIEMPO SIGUE
            avanzar(w, g.dt)
            return
        self.cob.tocados[g.control] += 1
        tipo, c = todos[g.control]
        rng = random.Random(g.s)
        if tipo == "boton":
            self._preparar_dialogos(rng)
            c.click()
        elif tipo == "slider":
            self._tocar_slider(c, g, rng)
        elif tipo in ("selector", "pestanas", "lista"):
            self._elegir(tipo, c, g)
        elif tipo == "campo":
            c.setText(TEXTOS_DE_CAMPO[g.s % len(TEXTOS_DE_CAMPO)])
        avanzar(w, g.dt)

    def _preparar_dialogos(self, rng):
        """Lo que va a contestar la persona si el clic abre un diálogo (se tira de la semilla del gesto)."""
        x = self.w.x
        existentes = [os.path.splitext(f)[0] for f in os.listdir(self.w.carpeta_programas)]
        for _ in range(rng.randint(1, 3)):
            nombre = rng.choice(NOMBRES)
            if nombre == "<existente>":
                nombre = rng.choice(existentes) if existentes else "prueba"
            x.nombres.append(nombre)
            x.confirmas.append(rng.random() < 0.5)

    def _tocar_slider(self, c, g, rng):
        w = self.w
        if g.tipo == "tecla":
            accion = (QtWidgets.QAbstractSlider.SliderSingleStepAdd, QtWidgets.QAbstractSlider.SliderSingleStepSub,
                      QtWidgets.QAbstractSlider.SliderPageStepAdd, QtWidgets.QAbstractSlider.SliderPageStepSub,
                      QtWidgets.QAbstractSlider.SliderToMinimum, QtWidgets.QAbstractSlider.SliderToMaximum)[g.s % 6]
            c.triggerAction(accion)
            return
        # UN ARRASTRE: AGARRA LA PERILLA, VA LLEGANDO UN EVENTO DEL RATÓN CADA POCO TIEMPO Y LA SUELTA
        destino = c.minimum() + round(g.v * (c.maximum() - c.minimum()))
        salida, eventos = c.value(), rng.randint(1, 12)
        entre = rng.choice((0.008, 0.008, 0.016, 0.05))
        c.setSliderDown(True)
        for i in range(1, eventos + 1):
            c.setSliderPosition(int(round(salida + (destino - salida) * i / eventos)))
            avanzar(w, entre)
        c.setSliderDown(False)

    @staticmethod
    def _elegir(tipo, c, g):
        n = c.count()
        indice = min(n - 1, int(g.v * n))
        if tipo == "pestanas":
            habilitadas = [i for i in range(n) if c.isTabEnabled(i)]
            c.setCurrentIndex(habilitadas[min(len(habilitadas) - 1, int(g.v * len(habilitadas)))])
        elif tipo == "lista":
            c.setCurrentRow(indice)
        else:
            c.setCurrentIndex(indice)

    # -- gestos al azar sobre lo que haya ahora ------------------------------------------------------------------------
    def libre(self, k, v, s, dt):
        """Un gesto sobre cualquier control que se pueda tocar ahora; qué gesto depende del tipo de control."""
        candidatos = self.disponibles()
        nombre, tipo, c = candidatos[k % len(candidatos)]
        if tipo == "boton":
            g = Gesto("click", nombre, v, s, dt)
        elif tipo == "slider":
            g = Gesto("tecla" if s % 5 == 0 else "arrastrar", nombre, v, s // 5, dt)
        elif tipo == "campo":
            g = Gesto("escribir", nombre, v, s, dt)
        else:
            g = Gesto("elegir", nombre, v, s, dt)
        self.hacer(g)

    # -- lo que se mira antes y después ---------------------------------------------------------------------------------
    def _antes(self):
        w, x = self.w, self.w.x
        return {"eventos": len(x.eventos), "enviados": len(x.enviados), "negadas": len(x.negadas),
                "excepciones": len(x.excepciones), "dialogos": len(x.dialogos),
                "archivos": _huellas(w.carpeta_programas), "grabando": w.grabadora.grabando,
                "con_movimiento": w.grabadora.muestras is not None, "editado": w.archivo_editado,
                "ultima_enviada": x.enviados[-1][1] if x.enviados else None}

    def _revisar(self, antes, g):
        w, x = self.w, self.w.x
        for revision in (self._excepciones, self._brinco, self._orden_negada, self._perdida, self._rebote, self._rachas,
                         self._reposo):
            hallazgo = revision(antes)
            if hallazgo is None:
                continue
            conocido = next((c for c in CONOCIDOS if c.clase == hallazgo[0] and c.contiene in hallazgo[1]), None)
            if conocido is not None and self.tolerar_conocidos:
                self.cob.conocidos[conocido.id] += 1      # SE VIO, SE CUENTA Y SE REPORTA; NO TUMBA LA CORRIDA
                continue
            raise self._hallazgo(*hallazgo)

    def _excepciones(self, antes):
        nuevas = self.w.x.excepciones[antes["excepciones"]:]
        if nuevas:
            tipo, valor, tb = nuevas[0]
            return "excepcion", f"dentro de un slot de Qt: {tipo.__name__}: {valor}\n" + "".join(traceback.format_tb(tb))

    def _orden_negada(self, antes):
        nuevas = self.w.x.negadas[antes["negadas"]:]
        if nuevas:
            hora, v, servos = nuevas[0]
            return "orden_negada", (f"[{v.clave}] {v.motivo} (servos {tuple(round(float(s), 2) for s in servos)}, "
                                    f"origen del plan «{self.w._origen_plan}»)")

    def _perdida(self, antes):
        w, x = self.w, self.w.x
        ahora = _huellas(w.carpeta_programas)
        nuevos = [d for d in x.dialogos[antes["dialogos"]:]]
        sobrescritos = {d[1] for d in nuevos if d[0] == "sobrescribir" and d[2]}
        for archivo, huella in antes["archivos"].items():
            if archivo not in ahora:
                return "perdida", f"el programa «{archivo}» desapareció"
            if ahora[archivo] != huella:
                # CAMBIA SIN PREGUNTAR SOLO EL QUE SE ESTÁ EDITANDO (EL EDITOR GUARDA CADA CAMBIO), O EL QUE SE PIDIÓ REEMPLAZAR
                if archivo not in sobrescritos and archivo not in (antes["editado"], w.archivo_editado):
                    return "perdida", f"el programa «{archivo}» cambió sin que se preguntara si se reemplazaba"
        if antes["grabando"] and antes["con_movimiento"] and not w.grabadora.grabando:
            guardada = any(a not in antes["archivos"] or ahora[a] != antes["archivos"][a] for a in ahora)
            descartada = any(d[0] == "confirmar" and d[1].startswith("Descartar") and d[2] for d in nuevos)
            if not guardada and not descartada:
                return "perdida", ("una grabación con movimiento dejó de grabarse sin guardarse y sin que se confirmara "
                                   "descartarla")

    def _rebote(self, antes):
        for e in self.w.x.eventos[antes["eventos"]:]:
            if e.origen == "Jog" and e.nivel == "aviso" and "destino_mm" in e.datos and "tope_mm" not in e.datos:
                w = self.w
                contexto = (f"punta en camino, modo «{w.en_camino}»" if w._destino_jog is not None else "punta quieta")
                return "rebote", (f"el slider dejó entrar a donde el planeador se niega y volvió solo ({contexto}, postura "
                                  f"«{w.postura}»): «{e.mensaje}»")

    def _rachas(self, antes):
        x = self.w.x
        nuevos_desde = antes["eventos"]
        if len(x.eventos) - nuevos_desde <= 0:
            return None
        cola = x.eventos[-200:]
        desfase = len(x.eventos) - len(cola)
        for r in rachas_de_avisos(cola, MINIMO_DE_UNA_RACHA):
            primero = desfase + r.primero
            ultimo = primero + r.longitud - 1
            if ultimo < nuevos_desde:
                continue                     # UNA RACHA QUE YA TERMINÓ ANTES DE ESTE GESTO
            autores = {self.gestos[a].tipo for a in x.autores[primero:ultimo + 1] if 0 <= a < len(self.gestos)}
            if autores == {"click"}:
                continue                     # LA PERSONA VOLVIÓ A PULSAR EL MISMO BOTÓN: ES SU DECISIÓN, NO UN DEFECTO
            if self._un_aviso_por_intento(primero, ultimo):
                continue                     # LA PERSONA VOLVIÓ A INTENTARLO Y CADA INTENTO TUVO SU ÚNICA RESPUESTA
            return "racha_de_avisos", f"{r.longitud} avisos idénticos seguidos ({r.nivel}, {r.origen}): «{r.mensaje}»"

    def _un_aviso_por_intento(self, primero, ultimo):
        """True si cada aviso de la racha es la única respuesta a un intento distinto de la persona: lo escribió un gesto
        DISTINTO, ese gesto es algo que la persona hace (no esperar) y entre un aviso y el siguiente el robot se movió o el aviso dice otra cosa (otro tope).
        La app calla el mismo aviso «hasta que algo se mueva»: si algo se movió y la persona insiste, vuelve a avisar, y
        eso es lo que debe hacer. Varios avisos del MISMO gesto, el mismo aviso letra por letra sin que nada se mueva, o avisos mientras la persona
        solo espera, siguen siendo una racha."""
        x = self.w.x
        autores, salidas = x.autores[primero:ultimo + 1], x.salidas[primero:ultimo + 1]
        textos = [e.mensaje for e in x.eventos[primero:ultimo + 1]]
        return (all(0 <= a < len(self.gestos) and self.gestos[a].tipo not in ("esperar", "llegar") for a in autores)
                and len(set(autores)) == len(autores)
                and all(salidas[i + 1] > salidas[i] or textos[i + 1] != textos[i] for i in range(len(autores) - 1)))

    def _brinco(self, antes):
        w, x = self.w, self.w.x
        previo = antes["ultima_enviada"]
        for _hora, servos, origen in x.enviados[antes["enviados"]:]:
            self.cob.pasos_a_la_placa += 1
            if previo is not None:
                mm = avance_punta_mm(w.robot, previo, servos)
                self.cob.avance_max_mm = max(self.cob.avance_max_mm, mm)
                if mm > BRINCO_ROJO_MM:
                    return "brinco", f"la punta saltó {mm:.0f} mm entre dos órdenes seguidas (origen «{origen}»)"
                if mm > BRINCO_REPORTE_MM:
                    self.cob.brincos.append((mm, origen, len(self.gestos)))
            previo = servos
        return None

    def _reposo(self, antes):
        w = self.w
        if w.temporizador_movimiento.isActive() or w.cola:
            return None
        fallas = []
        if w.ocupado:
            fallas.append("la ventana sigue «ocupada» con el robot quieto")
        if w.BTN_DETENER.isEnabled():
            fallas.append("el botón Detener sigue encendido con el robot quieto")
        if not servos_en_rango(w.robot, w.servos):
            fallas.append(f"los servos {tuple(round(float(s), 1) for s in w.servos)} están fuera de rango")
        if not any(s.isSliderDown() for s in w.sliders_servo):
            for i, (s, valor) in enumerate(zip(w.sliders_servo, w.servos)):
                if abs(s.value() - round(valor)) > 1:
                    fallas.append(f"el slider del servo {i + 1} dice {s.value()}° y el servo está en {valor:.1f}°")
        return ("reposo", "; ".join(fallas)) if fallas else None


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% REPRODUCIR Y REDUCIR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def reproducir(gestos, robot=None, semilla=None):
    """Repite una lista de gestos en una ventana nueva. Devuelve el Hallazgo que produce, o None si no pasa nada."""
    robot = robot or cargar_robot()
    sesion = Sesion(robot, semilla=semilla, tolerar_conocidos=False)
    try:
        with sin_dialogos_de_verdad():
            for g in gestos:
                sesion.hacer(g)
            correr(sesion.w)
    except Hallazgo as h:
        return h
    except NoTermina as e:
        return sesion._hallazgo("no_termina", str(e))
    finally:
        sesion.cerrar()
    return None


def reducir(hallazgo, robot=None, reproducir_con=None):
    """La lista de gestos más corta (ninguno sobra) que sigue produciendo un hallazgo DE LA MISMA CLASE."""
    repetir = reproducir_con or (lambda gs: reproducir(gs, robot, hallazgo.semilla))
    gestos = list(hallazgo.gestos)
    ultimo = repetir(gestos)
    if ultimo is None or ultimo.clase != hallazgo.clase:
        return hallazgo                          # NO SE REPITE: SE ENTREGA TAL CUAL, SIN REDUCIR
    gestos = list(ultimo.gestos)                 # LO QUE HAY DESPUÉS DEL GESTO QUE LO PROVOCA SOBRA
    trozo = max(1, len(gestos) // 2)
    while True:
        i, cambio = 0, False
        while i < len(gestos):
            candidato = gestos[:i] + gestos[i + trozo:]
            r = repetir(candidato) if candidato else None
            if r is not None and r.clase == hallazgo.clase:
                gestos, cambio = list(r.gestos), True
            else:
                i += trozo
        if trozo == 1:
            if not cambio:
                break
        else:
            trozo = max(1, trozo // 2)
    final = repetir(gestos)
    return final if final is not None else hallazgo


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LA MÁQUINA DE ESTADOS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _macro_modos(s):
    rng = random.Random(s)
    return [Gesto("elegir", "SELECTOR_POSTURA", rng.random(), 0, 0.0), Gesto("elegir", "selector_en_camino".upper(), rng.random(), 0, 0.0),
            Gesto("arrastrar", "SLIDER_VELOCIDAD", rng.random(), rng.randrange(10 ** 6), 0.0)]


def macros(s):
    """Las macros: listas de gestos que arman recorridos típicos. Cada una se arma desde su semilla."""
    rng = random.Random(s)
    pausa = lambda: rng.choice(PAUSAS)
    servo = lambda: rng.choice(("SLIDER_1", "SLIDER_2", "SLIDER_3"))
    eje = lambda: rng.choice(("SLIDER_X", "SLIDER_Y", "SLIDER_Z"))
    mover = lambda: [Gesto("arrastrar", servo() if rng.random() < 0.5 else eje(), rng.random(), rng.randrange(10 ** 6), pausa())
                     for _ in range(rng.randint(1, 3))]
    pestana = lambda i: Gesto("elegir", "PESTANAS_MOVIMIENTO", (i + 0.5) / 3, 0, 0.0)
    clic = lambda n: Gesto("click", n, 0.0, rng.randrange(10 ** 6), pausa())
    tipo = rng.randrange(6)
    if tipo == 0:      # GRABAR: EMPIEZA, SE MUEVE, DETIENE (Y EL DIÁLOGO SE CONTESTA AL AZAR: GUARDAR, CANCELAR, DESCARTAR…)
        return [pestana(rng.randrange(2)), clic("BTN_GRABAR")] + mover() + [pestana(2), clic("BTN_GRABAR")] + (
            [clic("BTN_GRABAR")] if rng.random() < 0.4 else [])
    if tipo == 1:      # REPRODUCIR UN PROGRAMA, Y A VECES DETENERLO A LA MITAD
        return [pestana(2), Gesto("elegir", "SELECTOR_PROGRAMA", rng.random(), 0, 0.0), clic("BTN_REPRODUCIR"),
                Gesto("esperar", "", 0.0, 0, rng.choice((0.2, 1.0, 4.0)))] + (
            [clic("BTN_DETENER")] if rng.random() < 0.5 else [Gesto("llegar", "", 0.0, 0, 0.0)])
    if tipo == 2:      # EDITAR PUNTOS
        pasos = [pestana(2), Gesto("elegir", "SELECTOR_PROGRAMA", rng.random(), 0, 0.0), clic("BTN_EDITAR_PUNTOS")]
        for _ in range(rng.randint(2, 6)):
            pasos += mover() if rng.random() < 0.4 else []
            pasos.append(clic(rng.choice(("BTN_PUNTO_NUEVO", "BTN_PUNTO_NUEVO", "BTN_REENSENAR", "BTN_QUITAR_PUNTO", "BTN_SUBIR_PUNTO",
                                          "BTN_BAJAR_PUNTO", "BTN_PASO_ADELANTE", "BTN_PASO_ATRAS", "BTN_VISTA_PREVIA"))))
            pasos.append(Gesto("elegir", "LISTA_PUNTOS", rng.random(), 0, 0.0))
            if rng.random() < 0.3:
                pasos.append(Gesto("elegir", "SELECTOR_TIPO_PUNTO", rng.random(), 0, 0.0))
            if rng.random() < 0.3:
                pasos.append(Gesto("llegar", "", 0.0, 0, 0.0))
        return pasos + ([clic("BTN_CERRAR_PUNTOS")] if rng.random() < 0.5 else [])
    if tipo == 3:      # JOG RÁPIDO CON LA PUNTA, CAMBIANDO DE MODO A MEDIA MARCHA
        pasos = [pestana(1)]
        for _ in range(rng.randint(2, 5)):
            pasos.append(Gesto("arrastrar", eje(), rng.random(), rng.randrange(10 ** 6), rng.choice((0.0, 0.008, 0.05))))
            if rng.random() < 0.4:
                pasos.append(Gesto("elegir", rng.choice(("SELECTOR_POSTURA", "SELECTOR_EN_CAMINO")), rng.random(), 0, rng.choice((0.0, 0.05))))
        return pasos
    if tipo == 4:      # ESCRIBIR LAS COORDENADAS Y CALCULAR, CON NÚMEROS BUENOS Y CON BASURA; A VECES SE DETIENE
        pasos = [pestana(2)]
        for n in ("TE_X", "TE_Y", "TE_Z"):
            pasos.append(Gesto("escribir", n, 0.0, rng.randrange(len(TEXTOS_DE_CAMPO)), 0.0))
        pasos += [clic(rng.choice(("CALCULAR", "COSECHAR"))), Gesto("esperar", "", 0.0, 0, rng.choice((0.1, 0.5, 2.0)))]
        return pasos + ([clic("BTN_DETENER")] if rng.random() < 0.4 else [Gesto("llegar", "", 0.0, 0, 0.0)])
    return _macro_modos(rng.randrange(10 ** 6)) + [clic("VER_VOLUMEN")] + mover()


def crear_maquina(robot, cobertura, semilla, tolerar=True):
    """La clase de la máquina de estados (cerrada sobre el robot, la cobertura y la semilla de la corrida)."""

    class Maquina(RuleBasedStateMachine):
        def __init__(self):
            super().__init__()
            self.sesion = Sesion(robot, cobertura, semilla, tolerar_conocidos=tolerar)

        @initialize(velocidad=st.floats(0, 1))
        def modos_iniciales(self, velocidad):
            # CADA VENTANA NUEVA ARRANCA EN UNO DE LOS 6 MODOS (2 posturas x 3 «en camino»), EN TURNO: ASÍ NINGÚN MODO SE QUEDA
            # SIN TOCAR POR PURA MALA SUERTE EN UNA CORRIDA CORTA. (Y DURANTE LA CORRIDA LOS GESTOS LOS CAMBIAN AL AZAR.)
            k = self.sesion.cob.ventanas - 1
            for nombre, indice, de in (("SELECTOR_POSTURA", k % 2, 2), ("SELECTOR_EN_CAMINO", (k // 2) % 3, 3)):
                self.sesion.hacer(Gesto("elegir", nombre, (indice + 0.5) / de, 0, 0.0))
            self.sesion.hacer(Gesto("tecla", "SLIDER_VELOCIDAD", 0.0, 4 if velocidad < 0.25 else 5, 0.0))

        @rule(k=st.integers(0, 10 ** 6), v=st.floats(0, 1), s=st.integers(0, 2 ** 31 - 1), dt=st.sampled_from(PAUSAS))
        def libre_a(self, k, v, s, dt):
            self.sesion.libre(k, v, s, dt)

        @rule(k=st.integers(0, 10 ** 6), v=st.floats(0, 1), s=st.integers(0, 2 ** 31 - 1), dt=st.sampled_from(PAUSAS))
        def libre_b(self, k, v, s, dt):
            self.sesion.libre(k, v, s, dt)

        @rule(s=st.integers(0, 2 ** 31 - 1))
        def macro_a(self, s):
            for g in macros(s):
                self.sesion.hacer(g)

        @rule(s=st.integers(0, 2 ** 31 - 1))
        def macro_b(self, s):
            for g in macros(s):
                self.sesion.hacer(g)

        @rule(dt=st.sampled_from((0.05, 0.3, 1.0, 3.0)))
        def esperar(self, dt):
            self.sesion.hacer(Gesto("esperar", "", 0.0, 0, dt))

        @rule()
        def llegar(self):
            self.sesion.hacer(Gesto("llegar", "", 0.0, 0, 0.0))

        def teardown(self):
            try:
                # AL FINAL, EL ROBOT TIENE QUE PODER LLEGAR: UN MOVIMIENTO QUE NO TERMINA ES UN HALLAZGO
                self.sesion.hacer(Gesto("llegar", "", 0.0, 0, 0.0))
            finally:
                self.sesion.cerrar()

    return Maquina


def configuracion(tamano):
    ejemplos, pasos = TAMANOS[tamano]
    return settings(max_examples=ejemplos, stateful_step_count=pasos, deadline=None, database=None, print_blob=False,
                    suppress_health_check=list(HealthCheck), phases=(Phase.generate,), report_multiple_bugs=False)


@dataclasses.dataclass
class Resultado:
    cobertura: Cobertura
    hallazgo: Hallazgo | None
    segundos: float


def explorar(tamano="corto", semilla=SEMILLA, robot=None, cobertura=None, con_reduccion=False, tolerar=True):
    """Una corrida del explorador. Devuelve el Resultado (cobertura, primer hallazgo o None, tiempo). `tolerar`: los defectos
    abiertos ya conocidos (CONOCIDOS) se cuentan pero no la detienen; con tolerar=False el primero que vea la detiene."""
    robot = robot or cargar_robot()
    cob = cobertura if cobertura is not None else Cobertura()
    maquina = crear_maquina(robot, cob, semilla, tolerar)
    prueba = get_state_machine_test(maquina, settings=configuracion(tamano), _min_steps=0, _flaky_state={"selecting_rule": False})
    prueba = semilla_de_hypothesis(semilla)(prueba)
    inicio, hallazgo = time.perf_counter(), None
    try:
        with sin_dialogos_de_verdad():
            prueba()
    except Hallazgo as h:
        hallazgo = h
    cob.segundos += time.perf_counter() - inicio
    if hallazgo is not None and con_reduccion:
        hallazgo = reducir(hallazgo, robot)
    return Resultado(cob, hallazgo, time.perf_counter() - inicio)


def main(argv=None):
    p = argparse.ArgumentParser(description="Explorador al azar de la ventana del brazo (sin pantalla).")
    p.add_argument("--largo", action="store_true", help="el tamaño largo (unos 25,000 gestos, ~15 min) en vez del corto")
    p.add_argument("--semilla", type=int, default=SEMILLA, help="semilla de la corrida: la misma semilla, el mismo recorrido")
    a = p.parse_args(argv)
    r = explorar("largo" if a.largo else "corto", a.semilla, con_reduccion=True)
    print(r.cobertura.informe())
    if r.hallazgo is None:
        print("\nSIN HALLAZGOS.")
        return 0
    print("\nHALLAZGO (ya reducido a la lista más corta que lo repite):\n" + str(r.hallazgo))
    print("\nComo prueba fija (pégalo en HALLAZGOS de tests/test_explorador.py):\n" + r.hallazgo.como_prueba_fija())
    return 1


if __name__ == "__main__":
    sys.exit(main())
