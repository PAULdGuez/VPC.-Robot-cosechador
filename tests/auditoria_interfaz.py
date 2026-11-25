"""Auditoría automática de la interfaz: mide cada elemento al píxel y devuelve los defectos encontrados.

La usan las pruebas (test_interfaz.py, sin abrir ventanas) y el recorrido de humo con la ventana real.
Revisa, en el estado en que esté la ventana, lo que un ojo se cansa de revisar:

  solape      dos elementos se enciman
  recorte     un elemento se sale de su recuadro o de la ventana
  texto       el texto no cabe en su elemento (se corta o se elide)
  borde       el texto toca el borde de su caja (medido en los píxeles dibujados)
  tamaño      un control es demasiado chico para atinarle (WCAG 2.2, criterio 2.5.8: 24 × 24 px)
  contraste   el texto no se distingue de su fondo (WCAG AA: 4.5:1; 3:1 en texto grande)
  fila        los elementos de una misma fila no están alineados
  rótulo      dentro de un dibujo (lateral, superior): dos rótulos se enciman, uno se sale o corta, o su letra es menor a la mínima

`auditar(ventana)` devuelve una lista de textos; vacía = sin defectos.
"""
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PyQt5 import QtCore, QtGui, QtWidgets
from pyvistaqt import QtInteractor

W = QtWidgets
CON_TEXTO = (W.QLabel, W.QPushButton, W.QCheckBox)
INTERACTIVOS = (W.QPushButton, W.QComboBox, W.QLineEdit, W.QCheckBox, W.QSlider, W.QToolButton, W.QTabBar)
# TAMBIÉN SE REVISAN LAS PESTAÑAS, LOS BOTONES DE HERRAMIENTA Y LOS LIENZOS DE LAS VISTAS (matplotlib y la 3D de pyvista)
HOJAS = CON_TEXTO + (W.QComboBox, W.QLineEdit, W.QSlider, W.QProgressBar, W.QListWidget, W.QTextBrowser,
                     W.QTabBar, W.QToolButton, FigureCanvasQTAgg, QtInteractor)

LADO_MINIMO = 24            # px: objetivo mínimo de un control
LETRA_MINIMA = 12           # px: la letra más chica que se acepta en un rótulo dibujado (la «letra_chica» del tema); LA FIJA EL AUDITOR
CONTRASTE_NORMAL = 4.5
CONTRASTE_GRANDE = 3.0      # texto ≥ 24 px, o ≥ 19 px en negritas
ROLES_CON_BORDE = ("chip",)  # QLabel con caja; los QPushButton siempre se revisan
ROLES_SIN_TEXTO = ("imagen",)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% QUÉ ELEMENTOS SE REVISAN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def _es_interno(w, ventana):
    """Piezas internas de otro control (la lista de un selector, las barras de una lista…)."""
    p = w.parentWidget()
    while p is not None and p is not ventana:
        if isinstance(p, (W.QComboBox, W.QAbstractScrollArea, W.QTabBar, W.QAbstractSpinBox)):
            return True
        p = p.parentWidget()
    return False


def hojas(ventana, tipos=HOJAS):
    return [w for w in ventana.findChildren(tipos)
            if w.isVisible() and not _es_interno(w, ventana) and not w.objectName().startswith("qt_")]


def _nombre(w):
    return w.objectName() or type(w).__name__


def _rol(w):
    return w.property("rol") or ""


def _rect(ventana, w):
    return QtCore.QRect(w.mapTo(ventana, QtCore.QPoint(0, 0)), w.size())


def _texto(w):
    texto = w.text() if hasattr(w, "text") else ""
    if texto and QtCore.Qt.mightBeRichText(texto):
        doc = QtGui.QTextDocument()
        doc.setHtml(texto)
        texto = doc.toPlainText()
    return texto.strip()


def _es_imagen(w):
    return isinstance(w, W.QLabel) and (_rol(w) in ROLES_SIN_TEXTO or w.hasScaledContents()
                                        or (w.pixmap() is not None and not w.pixmap().isNull()))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GEOMETRÍA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def solapes(ventana):
    elementos = [(w, _rect(ventana, w)) for w in hojas(ventana)]
    hallazgos = []
    for i, (a, ra) in enumerate(elementos):
        for b, rb in elementos[i + 1:]:
            if a.isAncestorOf(b) or b.isAncestorOf(a):
                continue
            cruce = ra.intersected(rb)
            if cruce.width() > 0 and cruce.height() > 0:
                hallazgos.append(f"solape: {_nombre(a)} y {_nombre(b)} se enciman {cruce.width()}×{cruce.height()} px")
    return hallazgos


def recortes(ventana):
    hallazgos = []
    for w in hojas(ventana):
        p = w.parentWidget()
        while p is not None:
            dentro = QtCore.QRect(w.mapTo(p, QtCore.QPoint(0, 0)), w.size())
            if not p.rect().contains(dentro):
                hallazgos.append(f"recorte: {_nombre(w)} se sale de {_nombre(p)} "
                                 f"({dentro.x()},{dentro.y()} {dentro.width()}×{dentro.height()} en {p.width()}×{p.height()})")
                break
            if p is ventana:
                break
            p = p.parentWidget()
    return hallazgos


def textos_cortados(ventana):
    hallazgos = []
    for w in hojas(ventana, CON_TEXTO):
        if _es_imagen(w) or not _texto(w):
            continue
        if isinstance(w, W.QLabel) and w.wordWrap():
            falta = w.heightForWidth(w.width()) - w.height()
            if falta > 1:
                hallazgos.append(f"texto: a {_nombre(w)} le faltan {falta} px de alto para «{_texto(w)[:40]}»")
            continue
        necesita = w.sizeHint()
        if necesita.width() > w.width() + 1:
            hallazgos.append(f"texto: «{_texto(w)[:40]}» no cabe en {_nombre(w)} (necesita {necesita.width()} px de ancho, tiene {w.width()})")
        if necesita.height() > w.height() + 1:
            hallazgos.append(f"texto: «{_texto(w)[:40]}» no cabe en {_nombre(w)} (necesita {necesita.height()} px de alto, tiene {w.height()})")
    for barra in ventana.findChildren(W.QTabBar):
        if not barra.isVisible():
            continue
        metrica = barra.fontMetrics()
        for i in range(barra.count()):
            ancho, texto = barra.tabRect(i).width(), barra.tabText(i)
            if ancho < metrica.horizontalAdvance(texto) + 8:
                hallazgos.append(f"texto: la pestaña «{texto}» mide {ancho} px y su texto {metrica.horizontalAdvance(texto)}")
    for combo in hojas(ventana, (W.QComboBox,)):
        metrica, disponible = combo.fontMetrics(), combo.width() - 44  # flecha 26 + sangría 10 + bordes y aire 8
        for i in range(combo.count()):
            ancho = metrica.horizontalAdvance(combo.itemText(i))
            if ancho > disponible:
                hallazgos.append(f"texto: «{combo.itemText(i)[:40]}» no cabe en {_nombre(combo)} (necesita {ancho} px, hay {disponible})")
    return hallazgos


def controles_chicos(ventana):
    hallazgos = []
    for w in hojas(ventana, INTERACTIVOS):
        if w.height() < LADO_MINIMO or w.width() < LADO_MINIMO:
            hallazgos.append(f"tamaño: {_nombre(w)} mide {w.width()}×{w.height()} px (mínimo {LADO_MINIMO}×{LADO_MINIMO})")
    return hallazgos


def filas_desalineadas(ventana):
    """En cada fila de un layout horizontal o de rejilla, los controles deben compartir el centro vertical."""
    hallazgos = []

    def revisar(widgets, donde):
        visibles = [w for w in widgets if w is not None and w.isVisible() and w.height() <= 48]
        if len(visibles) < 2:
            return
        centros = [_rect(ventana, w).center().y() for w in visibles]
        if max(centros) - min(centros) > 1:
            detalle = ", ".join(f"{_nombre(w)}={c}" for w, c in zip(visibles, centros))
            hallazgos.append(f"fila: desalineada en {donde} ({detalle})")

    for lay in ventana.findChildren(W.QHBoxLayout):
        revisar([lay.itemAt(i).widget() for i in range(lay.count())], lay.objectName())
    for lay in ventana.findChildren(W.QGridLayout):
        for fila in range(lay.rowCount()):
            revisar([lay.itemAtPosition(fila, c).widget() for c in range(lay.columnCount())
                     if lay.itemAtPosition(fila, c) is not None], f"{lay.objectName()} fila {fila}")
    return hallazgos


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PÍXELES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def _pixeles(ventana, w, rect=None):
    """El elemento tal como se ve: arreglo (alto, ancho, 3) y cuántos píxeles reales hay por px lógico.

    Se toma de la VENTANA (no del elemento suelto): un elemento con fondo transparente, capturado solo,
    sale sobre negro y da un contraste falso.
    """
    rect = rect if rect is not None else w.rect()
    zona = QtCore.QRect(w.mapTo(ventana, rect.topLeft()), rect.size())
    imagen = ventana.grab(zona).toImage().convertToFormat(QtGui.QImage.Format_RGB888)
    bits = imagen.constBits()
    bits.setsize(imagen.height() * imagen.bytesPerLine())
    filas = np.frombuffer(bits, np.uint8).reshape(imagen.height(), imagen.bytesPerLine())
    return filas[:, :imagen.width() * 3].reshape(imagen.height(), imagen.width(), 3).copy(), imagen.devicePixelRatio()


def _fondo(a):
    """El color más repetido de la región."""
    plano = a.reshape(-1, 3)
    claves = ((plano[:, 0].astype(np.int32) >> 3) << 10) | ((plano[:, 1].astype(np.int32) >> 3) << 5) | (plano[:, 2] >> 3)
    valores, cuentas = np.unique(claves, return_counts=True)
    return plano[claves == valores[cuentas.argmax()]].mean(axis=0)


def luminancia(rgb):
    c = np.asarray(rgb, dtype=float) / 255.0
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def contraste(a, b):
    """Razón de contraste WCAG entre dos colores (o entre un arreglo de colores y uno)."""
    la, lb = luminancia(a), luminancia(b)
    return (np.maximum(la, lb) + 0.05) / (np.minimum(la, lb) + 0.05)


def _contraste_del_texto(a):
    """Contraste entre el fondo de la región y su tinta más marcada; None si no hay texto dibujado."""
    if a.size == 0:
        return None
    por_pixel = contraste(a.reshape(-1, 3), _fondo(a))
    tinta = por_pixel[por_pixel > 1.25]
    if tinta.size < 6:
        return None
    return float(np.sort(tinta)[-max(3, tinta.size // 10):].mean())  # EL 10 % MÁS MARCADO: EL TRAZO, NO SU SUAVIZADO


def _umbral(w):
    fuente = w.font()
    px = QtGui.QFontInfo(fuente).pixelSize()
    return CONTRASTE_GRANDE if px >= 24 or (fuente.bold() and px >= 19) else CONTRASTE_NORMAL


def _con_borde(w):
    return isinstance(w, W.QPushButton) or _rol(w) in ROLES_CON_BORDE


def contrastes(ventana):
    hallazgos = []
    for w in hojas(ventana, CON_TEXTO + (W.QLineEdit, W.QComboBox)):
        texto = w.currentText() if isinstance(w, W.QComboBox) else _texto(w)
        if _es_imagen(w) or not texto or not w.isEnabled():
            continue
        a, dpr = _pixeles(ventana, w)
        m = int(4 * dpr) if (_con_borde(w) or isinstance(w, (W.QLineEdit, W.QComboBox))) else 0
        if isinstance(w, W.QComboBox):
            a = a[:, :a.shape[1] - int(30 * dpr)]  # SIN LA FLECHA
        if m and a.shape[0] > 3 * m and a.shape[1] > 3 * m:
            a = a[m:-m, m:-m]
        razon = _contraste_del_texto(a)
        if razon is not None and razon < _umbral(w) - 0.05:
            hallazgos.append(f"contraste: «{texto[:30]}» en {_nombre(w)} tiene {razon:.1f}:1 (mínimo {_umbral(w)}:1)")
    for barra in ventana.findChildren(W.QTabBar):
        if not barra.isVisible():
            continue
        for i in range(barra.count()):
            if not barra.isTabEnabled(i):
                continue
            a, dpr = _pixeles(ventana, barra, barra.tabRect(i).adjusted(3, 3, -3, -2))
            razon = _contraste_del_texto(a)
            if razon is not None and razon < CONTRASTE_NORMAL - 0.05:
                hallazgos.append(f"contraste: la pestaña «{barra.tabText(i)}» tiene {razon:.1f}:1 (mínimo {CONTRASTE_NORMAL}:1)")
    return hallazgos


def _tinta_en_los_lados(a, dpr, margen, esquina):
    """Lados (izquierdo, derecho, arriba, abajo) en los que hay tinta pegada al borde interior de la caja."""
    m, banda, e = int(round(margen * dpr)), max(2, int(round(2 * dpr))), int(round(esquina * dpr))
    alto, ancho = a.shape[:2]
    if alto <= 2 * (m + banda) + 2 or ancho <= 2 * (m + banda) + 2:
        return []
    tinta = contraste(a.reshape(-1, 3), _fondo(a[m:alto - m, m:ancho - m])).reshape(alto, ancho) > 1.5
    minimo = max(2, int(round(2 * dpr)))
    zonas = {"izquierdo": tinta[e:alto - e, m:m + banda], "derecho": tinta[e:alto - e, ancho - m - banda:ancho - m],
             "de arriba": tinta[m:m + banda, e:ancho - e], "de abajo": tinta[alto - m - banda:alto - m, e:ancho - e]}
    return [lado for lado, zona in zonas.items() if zona.size and int(zona.sum()) >= minimo]


def textos_pegados_al_borde(ventana):
    hallazgos = []
    for w in hojas(ventana, (W.QLabel, W.QPushButton)):
        if not _con_borde(w) or not _texto(w):
            continue
        a, dpr = _pixeles(ventana, w)
        for lado in _tinta_en_los_lados(a, dpr, margen=3, esquina=9):
            hallazgos.append(f"borde: el texto «{_texto(w)[:30]}» de {_nombre(w)} toca el borde {lado}")
    for barra in ventana.findChildren(W.QTabBar):
        if not barra.isVisible():
            continue
        for i in range(barra.count()):
            a, dpr = _pixeles(ventana, barra, barra.tabRect(i))
            for lado in _tinta_en_los_lados(a, dpr, margen=3, esquina=8):
                if lado in ("izquierdo", "derecho"):
                    hallazgos.append(f"borde: el texto de la pestaña «{barra.tabText(i)}» toca el borde {lado}")
    return hallazgos


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% RÓTULOS DENTRO DE LOS DIBUJOS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def rotulos_de(dibujo, letra_minima=LETRA_MINIMA, nombre="dibujo"):
    """Revisa los rótulos de un dibujo (planos.Dibujo): ninguno se encima con otro, ninguno se sale ni se corta en el borde del
    lienzo y ninguno baja de `letra_minima` px. Cada rótulo es (texto, x0, y0, x1, y1, letra) en px del lienzo."""
    ancho, alto = dibujo.px()
    rotulos, hallazgos = dibujo.rotulos(), []
    for i, a in enumerate(rotulos):
        if a.x0 < -0.5 or a.y0 < -0.5 or a.x1 > ancho + 0.5 or a.y1 > alto + 0.5:
            hallazgos.append(f"rótulo: «{a.texto[:30]}» de {nombre} se sale del lienzo de {ancho:.0f}×{alto:.0f} px "
                             f"({a.x0:.0f},{a.y0:.0f} a {a.x1:.0f},{a.y1:.0f})")
        if a.letra < letra_minima - 0.05:
            hallazgos.append(f"rótulo: «{a.texto[:30]}» de {nombre} tiene letra de {a.letra:g} px (mínimo {letra_minima} px)")
        for b in rotulos[i + 1:]:
            cruce_x, cruce_y = min(a.x1, b.x1) - max(a.x0, b.x0), min(a.y1, b.y1) - max(a.y0, b.y0)
            if cruce_x > 0.5 and cruce_y > 0.5:
                hallazgos.append(f"rótulo: «{a.texto[:30]}» y «{b.texto[:30]}» de {nombre} se enciman {cruce_x:.0f}×{cruce_y:.0f} px")
    return hallazgos


def rotulos_de_las_vistas(ventana):
    """Los rótulos de los dibujos que están a la vista en la ventana (los de un robot sin visor no existen)."""
    visor = getattr(ventana, "visor", None)
    hallazgos = []
    for dibujo in getattr(visor, "dibujos", ()):
        if dibujo.lienzo.isVisible():
            hallazgos.extend(rotulos_de(dibujo, nombre=f"{type(dibujo).__name__} ({dibujo.lienzo.objectName() or 'lienzo'})"))
    return hallazgos


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TODO JUNTO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

REVISIONES = (solapes, recortes, textos_cortados, textos_pegados_al_borde, controles_chicos, contrastes, filas_desalineadas,
              rotulos_de_las_vistas)


def auditar(ventana, revisiones=REVISIONES):
    W.QApplication.processEvents()
    hallazgos = []
    for revision in revisiones:
        hallazgos.extend(revision(ventana))
    return hallazgos


def auditar_estados(ventana, tamanos, poner_vision, pestanas, revisiones=REVISIONES):
    """Audita cada combinación de tamaño de ventana × visión abierta/cerrada × pestaña.

    Devuelve {descripción del estado: [hallazgos]} solo con los estados que tienen hallazgos.
    """
    con_defectos = {}
    for ancho, alto in tamanos:
        ventana.resize(ancho, alto)
        for abierta in (True, False):
            poner_vision(abierta)
            for i in range(pestanas.count()):
                pestanas.setCurrentIndex(i)
                W.QApplication.processEvents()
                hallazgos = auditar(ventana, revisiones)
                if hallazgos:
                    estado = (f"{ventana.width()}×{ventana.height()} · visión {'abierta' if abierta else 'cerrada'} · "
                              f"pestaña {pestanas.tabText(i)}")
                    con_defectos[estado] = hallazgos
    return con_defectos
