#CLASE: PYVISTA - QT DESIGNER
# CINEMATICA DIRECTA (DENAVIT-HARTENBERG) E INVERSA DEL ROBOT COSECHADOR DE 3 GDL
import numpy as np
import pyvista as pv

# %%%%%%%%%%%%%%%%%%%%%%%%%%% MEDIDAS DEL ROBOT (mm) — UNICA FUENTE DE VERDAD %%%%%%%%%%%%%%%%%%%%%%%%%%%
# LA SIMULACION, LA CINEMATICA INVERSA Y EL VOLUMEN DE TRABAJO LEEN ESTAS TRES CONSTANTES.
# SI CAMBIAS EL ROBOT, SOLO SE CAMBIA AQUI.
D1 = 350    # ALTURA DE LA BASE (d del eslabon 1)
A2 = 1000   # LARGO DEL BRAZO (a del eslabon 2)
A3 = 700    # LARGO DEL ANTEBRAZO (a del eslabon 3)

# RANGO FISICO DE CADA SERVO (grados), EL MISMO QUE VALIDA EL FIRMWARE DE LA ESP32
LIMITES_SERVO = ((0, 180), (20, 180), (0, 180))

HOME = (0, 0, 1065)  # POSICION DE REPOSO DEL EFECTOR (mm)


def _envolver(angulo):
    """Lleva un angulo al rango [-180, 180)."""
    return ((angulo + 180) % 360) - 180


def dh_a_servo(t1, t2, t3):
    """Angulos de Denavit-Hartenberg -> angulos que se le mandan a cada servo."""
    # SE REDONDEA ANTES DE ENVOLVER: UN y = -1e-14 DARIA 359.9999° EN VEZ DE 0°
    t1, t2, t3 = (round(float(t), 9) for t in (t1, t2, t3))
    return t1 % 360, 180 - t2, _envolver(t3) + 180


def servo_a_dh(s1, s2, s3):
    """Angulos de cada servo -> angulos de Denavit-Hartenberg (inversa de dh_a_servo)."""
    return s1, 180 - s2, s3 + 180


def servos_en_rango(s1, s2, s3, tolerancia=1e-6):
    return all(lo - tolerancia <= s <= hi + tolerancia
               for s, (lo, hi) in zip((s1, s2, s3), LIMITES_SERVO))


class cinematica:
    def __init__(self):
        self.plotter = pv.Plotter()
        # ESTADO DE LA SIMULACION (POR INSTANCIA, NO COMPARTIDO ENTRE VENTANAS)
        self.CONTADOR = 0
        self.LISTA_DATOS_BASE = []
        self.LISTA_ARTICULACIONES = []
        self.DISTANCIA_RELATIVA = []
        self.actor_volumen = None

    def matriz_rotacion_z (self, grados):
        rad = grados/180*np.pi
        rotacion = np.array([
            [np.cos(rad), -np.sin(rad),          0,          0],
            [np.sin(rad),  np.cos(rad),          0,          0],
            [          0,            0,          1,          0],
            [          0,            0,          0,          1]
        ])
        return rotacion
    def matriz_rotacion_x (self, grados):
        rad = grados/180*np.pi
        rotacion = np.array([
            [         1,            0,            0,          0],
            [         0,  np.cos(rad), -np.sin(rad),          0],
            [         0,  np.sin(rad),  np.cos(rad),          0],
            [         0,            0,            0,          1]
        ])
        return rotacion

    #%%%%%%%%%%%%%%%%%%%%%%%%%%% MATRICES DE TRASLACION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def matriz_traslacion_x(self, x):
        return np.array([[1,0,0,x],
                         [0,1,0,0],
                         [0,0,1,0],
                         [0,0,0,1]])
    def matriz_traslacion_z(self, z):
        return np.array([[1,0,0,0],
                         [0,1,0,0],
                         [0,0,1,z],
                         [0,0,0,1]])

    def denavit_hartemberg(self, theta_i, d_i, a_i, alpha_i):
        return (self.matriz_rotacion_z(theta_i) @ self.matriz_traslacion_z(d_i)
                @ self.matriz_traslacion_x(a_i) @ self.matriz_rotacion_x(alpha_i))

    def matrices(self, t1, t2, t3):
        """Matrices acumuladas de cada articulacion: [base, 1, 2, efector]."""
        A0_1 = self.denavit_hartemberg(t1, D1, 0, 90)
        A1_2 = self.denavit_hartemberg(t2, 0, A2, 0)
        A2_3 = self.denavit_hartemberg(t3, 0, A3, 0)
        A0_2 = A0_1 @ A1_2
        return [np.eye(4), A0_1, A0_2, A0_2 @ A2_3]

    def CD(self, t1, t2, t3):
        """Cinematica directa: angulos DH -> posicion (x, y, z) del efector en mm."""
        return self.matrices(t1, t2, t3)[-1][:3, 3]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def COSECHADOR(self, theta_1, theta_2, theta_3):
        """Dibuja (o actualiza) el robot en los angulos DH dados."""
        puntos = self.matrices(theta_1, theta_2, theta_3)

        if self.CONTADOR == 0:
            malla = pv.Plane(i_size=4000, j_size=4000, i_resolution=20, j_resolution=20)
            self.plotter.add_mesh(malla, style="wireframe", color="gray", opacity=0.3)

            for i in range(len(puntos)-1):
                start = puntos[i][:3, 3]
                end = puntos[i+1][:3, 3]
                tubo = pv.Line(start, end).tube(radius=40 if i == 0 else 20)

                self.plotter.add_mesh(tubo, color="orange", smooth_shading=True)
                self.LISTA_DATOS_BASE.append(tubo)

                self.DISTANCIA_RELATIVA.append(start)
                esfera = pv.Sphere(radius=20, center=start)
                self.LISTA_ARTICULACIONES.append(self.plotter.add_mesh(esfera, color="orange"))

            self.CONTADOR = 1
        else:
            for i in range(len(puntos)-1):
                start = puntos[i][:3, 3]
                end = puntos[i+1][:3, 3]
                # UNA NUEVA LINEA DE LA CUAL SOLO OCUPAMOS SUS COORDENADAS PARA ACTUALIZAR EL TUBO ANTERIOR
                nuevo_tubo = pv.Line(start, end).tube(radius=40 if i == 0 else 20)
                self.LISTA_DATOS_BASE[i].points = nuevo_tubo.points
                self.LISTA_ARTICULACIONES[i].SetPosition(*(start - self.DISTANCIA_RELATIVA[i]))

        self.plotter.render()

    def CI_soluciones(self, x, y, z):
        """Todas las soluciones de cinematica inversa (hasta 4), en grados DH.

        Orden de preferencia: base mirando al punto con codo arriba, base mirando al punto con
        codo abajo, y luego la base girada 180° (el brazo se inclina hacia atras).
        Lanza ValueError si el punto esta fuera del alcance geometrico del brazo.
        """
        # Distancia plana y altura MEDIDA DESDE EL HOMBRO (encima de la base, a D1)
        r = np.sqrt(x**2 + y**2)
        s = z - D1

        # Ley del coseno para theta3
        D = (r**2 + s**2 - A2**2 - A3**2) / (2 * A2 * A3)
        # (TOLERANCIA PARA EL BRAZO TOTALMENTE ESTIRADO, DONDE EL REDONDEO DA D = 1.0000000001)
        if np.abs(D) > 1 + 1e-9:
            raise ValueError("El punto está fuera del alcance del robot.")
        D = np.clip(D, -1.0, 1.0)

        if r < 1e-6:
            # PUNTO SOBRE EL EJE DE LA BASE (P. EJ. HOME): CUALQUIER GIRO SIRVE. SE USA EL CENTRO DEL
            # RANGO DEL SERVO PARA QUE EL ROBOT (Y LA CAMARA) MIRE A LA MITAD DE SU ZONA DE TRABAJO.
            theta1 = float(sum(LIMITES_SERVO[0]) / 2)
        else:
            theta1 = np.degrees(np.arctan2(y, x))
        soluciones = []
        for t1, r_plano in ((theta1, r), (theta1 + 180, -r)):
            for signo in (-1, 1):  # -1: codo arriba (theta3 negativo), 1: codo abajo
                t3 = np.arctan2(signo * np.sqrt(1 - D**2), D)
                phi = np.arctan2(A3 * np.sin(t3), A2 + A3 * np.cos(t3))
                t2 = np.arctan2(s, r_plano) - phi
                soluciones.append((t1, np.degrees(t2), np.degrees(t3)))
        return soluciones

    def CI(self, x, y, z):
        """Cinematica inversa: posicion (mm) -> angulos DH en grados.

        Devuelve la primera solucion que los servos pueden ejecutar; si ninguna cabe en sus
        rangos, la de codo arriba. Lanza ValueError si el punto esta fuera del alcance del brazo.
        """
        soluciones = self.CI_soluciones(x, y, z)
        for sol in soluciones:
            if servos_en_rango(*dh_a_servo(*sol)):
                return sol
        return soluciones[0]

    def angulos_servo(self, x, y, z):
        """Angulos de servo para llegar a (x, y, z), o None si el robot real no puede.

        No alcanza si el punto esta fuera del brazo o si ningun acomodo cabe en el rango de los servos.
        """
        try:
            soluciones = self.CI_soluciones(x, y, z)
        except ValueError:
            return None
        for sol in soluciones:
            servos = dh_a_servo(*sol)
            if servos_en_rango(*servos):
                return servos
        return None

    def punto_en_volumen(self, x, y, z):
        """True si el robot real puede llegar al punto (alcance + limites de los servos)."""
        return self.angulos_servo(x, y, z) is not None

    def volumen_trabajo(self, progresbar=None, paso=5):
        """Dibuja la nube de puntos que el efector alcanza recorriendo el rango de cada servo."""
        rangos = [range(lo, hi + 1, paso) for lo, hi in LIMITES_SERVO]
        total = len(rangos[0]) * len(rangos[1]) * len(rangos[2])
        if progresbar is not None:
            progresbar.setRange(0, total)
            progresbar.setValue(0)

        puntos_efector = []
        hechos = 0
        for s1 in rangos[0]:
            for s2 in rangos[1]:
                for s3 in rangos[2]:
                    puntos_efector.append(self.CD(*servo_a_dh(s1, s2, s3)))
                hechos += len(rangos[2])
                if progresbar is not None:
                    progresbar.setValue(hechos)

        self.puntos_efector = np.array(puntos_efector)
        # SI YA SE HABIA DIBUJADO, SE REEMPLAZA EN VEZ DE APILAR OTRA NUBE
        if self.actor_volumen is not None:
            self.plotter.remove_actor(self.actor_volumen)
        self.actor_volumen = self.plotter.add_mesh(
            pv.PolyData(self.puntos_efector), color="cyan", point_size=2,
            render_points_as_spheres=True)
        self.plotter.show_grid()
        self.plotter.render()
        return self.puntos_efector

    def mostrar(self):
        self.plotter.show()
