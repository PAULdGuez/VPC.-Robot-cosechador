"""Enlace entre la interfaz y la ESP32: la placa real por serial, o una ESP32 virtual.

La interfaz siempre habla el MISMO protocolo que espera ROBOT_ESP32_CODIGO.ino:
una linea "theta1,theta2,theta3\\n" con los angulos de cada servo.

- Sin variable de entorno ROBOT_PUERTO -> ESP32Virtual (no hace falta hardware).
- Con ROBOT_PUERTO=/dev/cu.usbserial-XXXX -> la placa real por pyserial.
"""
import os
import re
import time

BAUDRATE = 115200

# MISMOS RANGOS QUE VALIDA EL FIRMWARE (ROBOT_ESP32_CODIGO.ino)
RANGOS_FIRMWARE = ((0, 180), (20, 180), (0, 180))


def _to_int_arduino(texto):
    """Imita String::toInt() de Arduino: lee el entero inicial; si no hay, devuelve 0."""
    m = re.match(r"\s*([+-]?\d+)", texto)
    return int(m.group(1)) if m else 0


def formatear_comando(*servos):
    """Angulos de servo -> linea que entiende el firmware ("t1,t2,t3\\n", enteros)."""
    return ",".join(str(int(round(s))) for s in servos) + "\n"


class ESP32Virtual:
    """Emula ROBOT_ESP32_CODIGO.ino: mismo parseo, mismas validaciones y mismos mensajes.

    Con los 3 rangos por omision es EXACTAMENTE ese firmware. Con otros `rangos` emula el mismo
    firmware generalizado a N servos (sirve para otros robots del YAML): espera N valores
    separados por comas; el ultimo se lee con toInt, como hace el firmware con el tercero.
    """

    nombre = "ESP32 virtual"

    def __init__(self, rangos=RANGOS_FIRMWARE):
        self.rangos = tuple(rangos)
        self.angulos = None          # ULTIMOS ANGULOS QUE "MOVIERON" LOS SERVOS
        self.historial = []          # TODOS LOS COMANDOS ACEPTADOS, PARA PRUEBAS O GRAFICAR
        formato = ",".join(f"theta{i + 1}" for i in range(len(self.rangos)))
        self._salida = ["", f"Envie angulos en formato: {formato}"]

    def write(self, datos):
        texto = datos.decode() if isinstance(datos, (bytes, bytearray)) else str(datos)
        for linea in texto.split("\n")[:-1] if texto.endswith("\n") else texto.split("\n"):
            self._procesar(linea.strip())
        return len(datos)

    def _procesar(self, entrada):
        # COMO EL FIRMWARE: BUSCA N-1 COMAS (indexOf), CADA COMA DESPUES DE LA ANTERIOR, Y LA PRIMERA > 0
        comas, desde = [], 0
        for _ in range(len(self.rangos) - 1):
            pos = entrada.find(",", desde)
            if pos <= 0:
                break
            comas.append(pos)
            desde = pos + 1
        if len(comas) == len(self.rangos) - 1:
            cortes = [-1] + comas + [len(entrada)]
            angulos = tuple(_to_int_arduino(entrada[cortes[i] + 1:cortes[i + 1]]) for i in range(len(self.rangos)))
            if all(lo <= a <= hi for a, (lo, hi) in zip(angulos, self.rangos)):
                self.angulos = angulos
                self.historial.append(angulos)
            else:
                self._salida.append("Los angulos deben estar entre 0-180°")
        else:
            formato = ",".join(f"angulo{i + 1}" for i in range(len(self.rangos)))
            self._salida.append(f"Formato incorrecto. Usar: {formato}")

    def leer_respuestas(self):
        salida, self._salida = self._salida, []
        return [s for s in salida if s]

    def close(self):
        pass


class ESP32Serial:
    """La placa real por puerto serial."""

    def __init__(self, puerto, baudrate=BAUDRATE):
        import serial  # pyserial; solo se necesita con hardware real
        self.nombre = f"ESP32 en {puerto}"
        self.puerto = serial.Serial(puerto, baudrate, timeout=0)
        time.sleep(2)  # LA ESP32 SE REINICIA AL ABRIR EL PUERTO

    def write(self, datos):
        return self.puerto.write(datos.encode() if isinstance(datos, str) else datos)

    def leer_respuestas(self):
        lineas = []
        while self.puerto.in_waiting:
            linea = self.puerto.readline().decode(errors="replace").strip()
            if linea:
                lineas.append(linea)
        return lineas

    def close(self):
        self.puerto.close()


class Enlace:
    """Lo que usa la interfaz: manda angulos de servo y devuelve lo que contesto la placa."""

    def __init__(self, dispositivo):
        self.dispositivo = dispositivo
        self.nombre = dispositivo.nombre

    def enviar(self, *servos):
        comando = formatear_comando(*servos)
        self.dispositivo.write(comando)
        return comando.strip(), self.dispositivo.leer_respuestas()

    def close(self):
        self.dispositivo.close()


def abrir_enlace(puerto=None, rangos=RANGOS_FIRMWARE):
    """Abre la placa real si hay puerto (argumento o ROBOT_PUERTO); si no, la ESP32 virtual.

    `rangos` (min, max) por servo configuran la virtual; deben coincidir con lo que valida el
    firmware real, que los trae escritos en el .ino.
    """
    puerto = puerto or os.environ.get("ROBOT_PUERTO")
    return Enlace(ESP32Serial(puerto) if puerto else ESP32Virtual(rangos))
