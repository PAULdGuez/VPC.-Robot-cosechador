"""El visor con pestañas: «Todas · 3D · Lateral · Superior · Ficha».

- «Todas»: la 3D grande a la izquierda; la lateral y la superior a la derecha, una sobre otra. Las tres se mueven a la vez.
- Una pestaña, o un clic (sin arrastrar) sobre una vista del mosaico, deja esa vista sola. Otro clic en la pestaña activa, o
  en «Todas», regresa a las tres. Arrastrar sobre la 3D sigue girando la cámara.
- «Ficha»: la hoja «lado a lado» (ficha.py) y, a la derecha de las pestañas, «Guardar ficha…».
- Nada se crea ni se destruye al cambiar de pestaña: las vistas existen siempre; solo se muestran, se esconden y cambian de celda.
"""
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PyQt5 import QtCore, QtWidgets

from . import ficha as hoja_de_ficha
from .planos import Geometria, Lateral, Superior

Qt = QtCore.Qt
W = QtWidgets
NOMBRES = ("Todas", "3D", "Lateral", "Superior", "Ficha")
TODAS, VISTA_3D, LATERAL, SUPERIOR, FICHA = range(5)
AYUDAS = ("Las tres vistas a la vez",) + ("Deja sola esta vista. Otro clic en la misma pestaña regresa a las tres.",) * 3 + \
         ("La hoja del robot: sus dibujos con medidas y sus datos. Se guarda como imagen.",)
SEPARACION = 4            # px ENTRE VISTAS DEL MOSAICO (SE VE EL GRIS DEL PANEL)
ARRASTRE = 4              # px: SI EL CURSOR SE MOVIÓ MÁS QUE ESTO ENTRE PRESIONAR Y SOLTAR, FUE UN ARRASTRE, NO UN CLIC
PROPORCION_TODAS = (3, 2)  # ANCHO DE LA 3D : ANCHO DE LA COLUMNA DE LOS DOS DIBUJOS
AJUSTE_PESTANA = "visor/pestana"


class Pestanas(W.QTabBar):
    """Tira de pestañas en la que un clic sobre la pestaña YA activa regresa a la primera («Todas»).

    Se decide al SOLTAR y comparando con la pestaña que estaba activa al PRESIONAR: así da igual que el estilo de Qt
    cambie de pestaña al presionar (Fusion, Windows) o al soltar (el de macOS).
    """

    def __init__(self, *a):
        super().__init__(*a)
        self._activa_al_presionar = -1

    def mousePressEvent(self, evento):
        self._activa_al_presionar = self.currentIndex() if evento.button() == Qt.LeftButton else -1
        super().mousePressEvent(evento)

    def mouseReleaseEvent(self, evento):
        super().mouseReleaseEvent(evento)
        era, self._activa_al_presionar = self._activa_al_presionar, -1
        if evento.button() == Qt.LeftButton and era > TODAS and self.tabAt(evento.pos()) == era and self.currentIndex() == era:
            self.setCurrentIndex(TODAS)


class Visor(W.QWidget):
    """Vive dentro del recuadro `padre` (PANTALLA del .ui) y toma la vista 3D de la escena que ya existe (no crea otra)."""
    vista_3d_a_la_vista = QtCore.pyqtSignal(bool)    # LOS BOTONES DE CÁMARA SOLO MUEVEN LA VISTA 3D

    def __init__(self, padre, escena, robot, ajustes=None, lazos=None):
        super().__init__(padre)
        self.setObjectName("VISOR_VISTAS")
        self.escena, self.robot, self.ajustes, self._presion = escena, robot, ajustes, None
        self.servos = tuple(robot.postura_home or [j.servo.centro for j in robot.articulaciones])
        self.lazos, self.volumen_visible = lazos, False
        self.geo = Geometria(robot, lazos)
        self.v3d = escena.plotter.interactor
        self.lateral, self.superior = Lateral(self.geo, FigureCanvasQTAgg, self.servos), Superior(self.geo, FigureCanvasQTAgg, self.servos)
        self.vistas = (self.v3d, self.lateral.lienzo, self.superior.lienzo)
        self.ficha = hoja_de_ficha.PaginaFicha(self.geo)
        self.dibujos = (self.lateral, self.superior, *self.ficha.dibujos)

        self.pestanas = Pestanas()
        self.pestanas.setObjectName("PESTANAS_VISTAS")
        self.pestanas.setExpanding(False)
        self.pestanas.setDrawBase(False)
        for nombre, ayuda in zip(NOMBRES, AYUDAS):
            self.pestanas.setTabToolTip(self.pestanas.addTab(nombre), ayuda)
        self.btn_guardar_ficha = W.QPushButton("Guardar ficha…")
        self.btn_guardar_ficha.setObjectName("BTN_GUARDAR_FICHA")
        self.btn_guardar_ficha.setProperty("rol", "mini")
        self.btn_guardar_ficha.setToolTip("Guardar la hoja del robot como imagen, con su vista 3D")
        self.btn_guardar_ficha.clicked.connect(lambda: self.guardar_ficha())
        tira = W.QHBoxLayout()
        tira.setObjectName("lay_pestanas_vistas")
        tira.setContentsMargins(0, 0, 0, 0)
        tira.setSpacing(8)
        tira.addWidget(self.pestanas, 0, Qt.AlignBottom)
        tira.addStretch(1)
        tira.addWidget(self.btn_guardar_ficha)

        self.mosaico = W.QWidget()
        self.mosaico.setObjectName("MOSAICO_VISTAS")
        self.rejilla = W.QGridLayout(self.mosaico)
        self.rejilla.setObjectName("lay_vistas")
        self.rejilla.setContentsMargins(0, 0, 0, 0)
        self.rejilla.setSpacing(SEPARACION)
        self.paginas = W.QStackedLayout()
        self.paginas.setObjectName("lay_paginas_visor")
        self.paginas.addWidget(self.mosaico)
        self.paginas.addWidget(self.ficha)
        cuerpo = W.QVBoxLayout(self)
        cuerpo.setObjectName("lay_visor")
        cuerpo.setContentsMargins(0, 0, 0, 0)
        cuerpo.setSpacing(0)
        cuerpo.addLayout(tira)
        cuerpo.addLayout(self.paginas, 1)
        for vista in self.vistas:
            vista.setParent(self.mosaico)
            vista.setMinimumSize(40, 40)
            vista.setSizePolicy(W.QSizePolicy.Ignored, W.QSizePolicy.Ignored)   # EL TAMAÑO LO DECIDE LA REJILLA, NO LA VISTA
            vista.installEventFilter(self)
        self.mosaico.setSizePolicy(W.QSizePolicy.Ignored, W.QSizePolicy.Ignored)
        diseno = padre.layout() or W.QVBoxLayout(padre)
        diseno.setContentsMargins(0, 0, 0, 0)
        diseno.addWidget(self)
        for control in [self, *self.findChildren(W.QWidget)]:                    # LO MISMO QUE HACE tema.aplicar()
            control.setAttribute(Qt.WA_LayoutUsesWidgetRect)
        inicial = self._pestana_recordada()
        self.pestanas.setCurrentIndex(inicial)
        self.pestanas.currentChanged.connect(self.acomodar)
        self.acomodar()

    def _pestana_recordada(self):
        try:
            indice = int(self.ajustes.value(AJUSTE_PESTANA, TODAS)) if self.ajustes is not None else TODAS
        except (TypeError, ValueError):
            indice = TODAS
        return indice if 0 <= indice < len(NOMBRES) else TODAS

    # %%%%% QUÉ SE VE Y DÓNDE %%%%%
    @property
    def principal(self):
        """La pestaña activa: 0 = todas, 1 = 3D, 2 = lateral, 3 = superior, 4 = ficha."""
        return self.pestanas.currentIndex()

    def ver(self, indice):
        self.pestanas.setCurrentIndex(indice)

    def vista_3d_visible(self):
        return self.principal in (TODAS, VISTA_3D)

    def acomodar(self, *_):
        principal = self.principal
        todas, en_ficha = principal == TODAS, principal == FICHA
        self.paginas.setCurrentWidget(self.ficha if en_ficha else self.mosaico)
        self.btn_guardar_ficha.setVisible(en_ficha)
        for vista in self.vistas:
            self.rejilla.removeWidget(vista)
        for k in range(3):
            self.rejilla.setColumnStretch(k, 0)
            self.rejilla.setRowStretch(k, 0)
        if todas:
            self.rejilla.addWidget(self.v3d, 0, 0, 2, 1)
            self.rejilla.addWidget(self.lateral.lienzo, 0, 1)
            self.rejilla.addWidget(self.superior.lienzo, 1, 1)
            for k, peso in enumerate(PROPORCION_TODAS):
                self.rejilla.setColumnStretch(k, peso)
            self.rejilla.setRowStretch(0, 1)
            self.rejilla.setRowStretch(1, 1)
        elif not en_ficha:
            self.rejilla.addWidget(self.vistas[principal - 1], 0, 0)
            self.rejilla.setColumnStretch(0, 1)
            self.rejilla.setRowStretch(0, 1)
        for k, vista in enumerate(self.vistas):
            vista.setVisible(todas or k == principal - 1)
            vista.setCursor(Qt.PointingHandCursor if todas and vista is not self.v3d else Qt.ArrowCursor)
            vista.setToolTip("Clic: dejar sola esta vista" if todas else "")
        for dibujo in (self.lateral, self.superior):  # EN EL MOSAICO CADA DIBUJO DICE QUÉ ES; SOLO, YA LO DICE SU PESTAÑA
            dibujo.con_titulo = todas
            if dibujo.lienzo.isVisible():
                dibujo.componer()
                dibujo.poner_pose(self.servos, parcial=False)
        if en_ficha:
            for dibujo in self.ficha.dibujos:
                dibujo.poner_pose(self.servos, parcial=False)
        if self.ajustes is not None:
            self.ajustes.setValue(AJUSTE_PESTANA, principal)
        self.vista_3d_a_la_vista.emit(self.vista_3d_visible())

    def eventFilter(self, vista, evento):
        """Clic (sin arrastrar) sobre una vista del mosaico: esa vista queda sola. El evento NO se consume: la vista 3D
        sigue recibiendo el arrastre para girar la cámara."""
        if "pestanas" not in self.__dict__:      # EL VISOR SE ESTÁ DESARMANDO (CIERRE): UNA EXCEPCIÓN AQUÍ TUMBARÍA TODA LA APP
            return False
        if self.principal == TODAS and vista in self.vistas:
            tipo = evento.type()
            if tipo == QtCore.QEvent.MouseButtonPress and evento.button() == Qt.LeftButton:
                self._presion = (vista, evento.pos())
            elif tipo == QtCore.QEvent.MouseButtonRelease and evento.button() == Qt.LeftButton and self._presion:
                de_quien, donde = self._presion
                self._presion = None
                if de_quien is vista and (evento.pos() - donde).manhattanLength() <= ARRASTRE:
                    # DESPUÉS DE QUE LA VISTA TERMINE DE ATENDER SU PROPIO EVENTO
                    QtCore.QTimer.singleShot(0, lambda k=self.vistas.index(vista) + 1: self.ver(k))
        return False

    # %%%%% VIVAS %%%%%
    def poner_servos(self, servos):
        """La postura nueva en los dibujos que están a la vista (los escondidos solo la anotan: al verse se ponen al día)."""
        self.servos = tuple(servos)
        for dibujo in self.dibujos:
            if dibujo.lienzo.isVisible():
                dibujo.poner_pose(servos)
            else:
                dibujo.anotar_pose(servos)

    def poner_volumen(self, lazos, visible=True):
        """El contorno del volumen (o None si el robot no tiene un corte plano que dibujar): los dibujos se arman de nuevo."""
        self.lazos = lazos
        self.volumen_visible = bool(visible)
        self.geo = Geometria(self.robot, lazos)
        for dibujo in (self.lateral, self.superior):
            dibujo.poner_geometria(self.geo, self.volumen_visible)
        self.ficha.poner_geometria(self.geo, self.volumen_visible)

    def ocultar_volumen(self):
        """Se quita el volumen de los dibujos; el contorno ya calculado se conserva para volver a verlo sin recalcular."""
        self.volumen_visible = False
        for dibujo in (self.lateral, self.superior):
            dibujo.poner_geometria(self.geo, False)
        self.ficha.poner_geometria(self.geo, False)

    # %%%%% LA FICHA %%%%%
    def guardar_ficha(self, ruta=None):
        """Guarda la hoja completa (1600×1000, con la vista 3D) como imagen. Sin `ruta`, la persona elige dónde."""
        if ruta is None:
            ruta, _ = W.QFileDialog.getSaveFileName(self, "Guardar ficha", f"ficha-{self.robot.nombre}.png", "Imagen PNG (*.png)")
            if not ruta:
                return None
        hoja_de_ficha.guardar_hoja(ruta, self.geo, self.servos, self.lazos, self.volumen_visible)
        return ruta
