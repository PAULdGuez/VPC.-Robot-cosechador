#CLASE: PYVISTA - QT DESIGNER
# DEFINICOMOS CLASE
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm 
import math
import pyvista as pv
import vtk
from scipy.spatial import KDTree
import numpy as np
from math import atan2, acos, sqrt, degrees, radians

class cinematica:
    CONTADOR = 0
    LISTA_SLABONES = []
    LISTA_ARTICULACIONES = []
    DISTANCIA_RELATIVA = []
    PROCESO = 0
    def __init__(self):
        self.plotter = pv.Plotter()
        
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
    def matriz_rotacion_y (self, grados):
        rad = grados/180*np.pi
        rotacion = np.array([
            [ np.cos(rad),         0, np.sin(rad),          0],
            [           0,         1,           0,          0],
            [-np.sin(rad),         0, np.cos(rad),          0],
            [           0,         0,           0,          1]
        ])
        return rotacion 

    #%%%%%%%%%%%%%%%%%%%%%%%%%%% MATRICES DE TRASLACION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def matriz_traslacion_x(self, x):
        traslacion = np.array([[1,0,0,x],
                               [0,1,0,0],
                               [0,0,1,0],
                               [0,0,0,1]])
        return traslacion
    def matriz_traslacion_y(self, y):
        traslacion = np.array([[1,0,0,0],
                               [0,1,0,y],
                               [0,0,1,0],
                               [0,0,0,1]])
        return traslacion
    def matriz_traslacion_z(self, z):
        
        traslacion = np.array([[1,0,0,0],
                               [0,1,0,0],
                               [0,0,1,z],
                               [0,0,0,1]])
        return traslacion
    
    def denavit_hartemberg(self, theta_i, d_i, a_i, alpha_i):
        MTH = self.matriz_rotacion_z(theta_i)@self.matriz_traslacion_z(d_i)@self.matriz_traslacion_x(a_i)@self.matriz_rotacion_x(alpha_i)
        #MTH = self.matriz_rotacion_x(theta_i)@self.matriz_traslacion_x(d_i)@self.matriz_traslacion_y(a_i)@self.matriz_rotacion_y(alpha_i)
        return MTH
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
    def COSECHADOR(self, theta_1,d_1,a_1,alpha_1,  theta_2,d_2,a_2,alpha_2,  theta_3,d_3,a_3,alpha_3):
        A0_0 = self.matriz_rotacion_z(0)
        A0_1 = self.denavit_hartemberg(theta_1, d_1, a_1, alpha_1)
        A1_2 = self.denavit_hartemberg(theta_2, d_2, a_2, alpha_2)
        A2_3 = self.denavit_hartemberg(theta_3, d_3, a_3, alpha_3)
        
        A_0_1_0 = A0_0 @ A0_1
        A_0_2 = A_0_1_0 @ A1_2
        A_0_3 = A_0_2 @ A2_3
 
        puntos = [A0_0, A_0_1_0, A_0_2, A_0_3]
        
        
        if self.CONTADOR == 0:
            self.LISTA_DATOS_BASE = []
            malla = pv.Plane(i_size=500, j_size=500, i_resolution=10, j_resolution=10)
            self.plotter.add_mesh(malla, style="wireframe", color="gray", opacity=0.3)

            for i in range(len(puntos)-1):
                start = puntos[i][:3, 3]
                end = puntos[i+1][:3, 3]
                
                linea = pv.Line(start, end)

                if i == 0:
                    tubo = linea.tube(radius=40)
                else:
                    tubo = linea.tube(radius=20)
                
                actor = self.plotter.add_mesh(tubo, color="orange", smooth_shading=True)
                #self.LISTA_SLABONES.append(actor)
                self.LISTA_DATOS_BASE.append(tubo)  
                
                self.DISTANCIA_RELATIVA.append(start)
                esfera = pv.Sphere(radius=20, center=start)

                actor_esfera = self.plotter.add_mesh(esfera, color="orange")
                self.LISTA_ARTICULACIONES.append(actor_esfera)

            self.CONTADOR = 1

        else:
            for i in range(len(puntos)-1):
                start = puntos[i][:3, 3]
                end = puntos[i+1][:3, 3]

                # UNA NUEVA LINEA DE LA CUAL SOLO OCUPAREMOS SU COORDENADAS PARA ACTUALIZAR EL ANTERIOR TUBO
                nueva_linea = pv.Line(start, end)
                if i == 0:
                    nuevo_tubo = nueva_linea.tube(radius=40)
                else:
                    nuevo_tubo = nueva_linea.tube(radius=20)

                self.LISTA_DATOS_BASE[i].points = nuevo_tubo.points

                NUEVA_POS = start-self.DISTANCIA_RELATIVA[i]
                self.LISTA_ARTICULACIONES[i].SetPosition(*NUEVA_POS)
     
        self.plotter.render()

    def CI(self, x, y, z):
        L1 = L2 = L3 = 100  # Longitudes de los eslabones

        # 1. theta1 (rotación en Z)
        theta1 = np.arctan2(y, x)

        # 2. Distancia plana y altura
        r = np.sqrt(x**2 + y**2)
        s = z

        # 3. Ley del coseno para theta3
        D = (r**2 + s**2 - L2**2 - L3**2) / (2 * L2 * L3)
        if np.abs(D) > 1:
            raise ValueError("El punto está fuera del alcance del robot.")

        # Solución 1: Codo Abajo (theta3 positivo)
        #theta3_down = np.arctan2(np.sqrt(1 - D**2), D)
        theta3_down = np.arctan2(np.sqrt(1 - D**2), D)
        phi1_down = np.arctan2(L3 * np.sin(theta3_down), L2 + L3 * np.cos(theta3_down))
        theta2_down = np.arctan2(s, r) - phi1_down

        # Solución 2: Codo Arriba (theta3 negativo)
        theta3_up = np.arctan2(-np.sqrt(1 - D**2), D)
        phi1_up = np.arctan2(L3 * np.sin(theta3_up), L2 + L3 * np.cos(theta3_up))
        theta2_up = np.arctan2(s, r) - phi1_up

        # Devolver ambas soluciones en grados
        return np.degrees(theta1), np.degrees(theta2_up), np.degrees(theta3_up)     # Codo arriba

    def CI_FILTRO(self, x, y, z):
        L1 = L2 = L3 = 100  # Longitudes de los eslabones

        # 1. theta1 (rotación en Z)
        theta1 = np.arctan2(y, x)

        # 2. Distancia plana y altura
        r = np.sqrt(x**2 + y**2)
        s = z

        # 3. Ley del coseno para theta3
        D = (r**2 + s**2 - L2**2 - L3**2) / (2 * L2 * L3)
        return np.abs(D) > 1

    def volumen_trabajo(self, progresbar=None):
        puntos_efector = []

        # Barrido de los ángulos
        for t1 in range(0, 181, 5):     # paso de 10° para θ1
            for t2 in range(0, 181, 5): # paso de 10° para θ2
                for t3 in range(0, 181, 5): # paso de 10° para θ3
                    try:
                        self.PROCESO +=1
                        progresbar.setValue(self.PROCESO)
                        A0_1 = self.denavit_hartemberg(t1, 100, 0, 90)
                        A1_2 = self.denavit_hartemberg(t2, 0, 100, 0)
                        A2_3 = self.denavit_hartemberg(t3, 0, 100, 0)

                        T = A0_1 @ A1_2 @ A2_3
                        x, y, z = T[0, 3], T[1, 3], T[2, 3]
                        puntos_efector.append([x, y, z])
                    except:
                        continue

        self.puntos_efector = np.array(puntos_efector)
        self.tree = KDTree(self.puntos_efector)

        nube = pv.PolyData(self.puntos_efector)
        self.plotter.add_mesh(nube, color="cyan", point_size=1, render_points_as_spheres=True)
        self.plotter.show_grid()  
        self.plotter.show()

      # Asegúrate de importar al inicio

    # Dentro de la clase 'cinematica':
    def punto_en_volumen(self, x, y, z, tolerancia=20.0):
        # ACA PONER LA FUNCION DE ci Y QUE AVALUE SI NO HAY PEDO SEN LLEGAR A ESE PUNTO (ESTO EN GENRAL YA QUE TRABAAJA CON TETAS DE -180 A 180)
        # SI PASA EL FILTRO, AHORA VERIFICAR SI ESTA DENTRO DE NUESTRO VOLUMEN DE TRBABAJO CALCULADO PARA THETA SDE 0 A 180
        if self.CI_FILTRO(x, y, z):
            print("El punto esta fuera del alcanze del robot")
        else:
            if not hasattr(self, 'tree'):
                raise ValueError("Primero debes generar el volumen de trabajo con 'volumen_trabajo()'.")
            dist, _ = self.tree.query([x, y, z])
            return dist <= tolerancia

    def mostrar(self):
        self.plotter.show()