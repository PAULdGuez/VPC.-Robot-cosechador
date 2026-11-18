"""Enlace con la ESP32: el emulador debe portarse igual que el firmware."""
import pytest

from cosechador.hardware.enlace import ESP32Virtual, abrir_enlace, formatear_comando


@pytest.fixture
def esp():
    e = ESP32Virtual()
    e.leer_respuestas()  # DESCARTA EL MENSAJE DE BIENVENIDA
    return e


def test_saluda_como_el_firmware():
    assert ESP32Virtual().leer_respuestas() == ["Envie angulos en formato: theta1,theta2,theta3"]


def test_comando_valido_mueve_los_servos(esp):
    esp.write(b"90,45,120\n")
    assert esp.angulos == (90, 45, 120)
    assert esp.leer_respuestas() == []  # EL FIRMWARE NO CONTESTA NADA SI TODO SALIO BIEN


@pytest.mark.parametrize("comando", ["181,45,45\n", "90,19,45\n", "90,45,-1\n"])
def test_fuera_de_rango_no_mueve(esp, comando):
    esp.write(comando)
    assert esp.angulos is None
    assert esp.leer_respuestas() == ["Los angulos deben estar entre 0-180°"]


@pytest.mark.parametrize("comando", ["90 45 45\n", "90,45\n", ",45,45\n"])
def test_formato_incorrecto(esp, comando):
    esp.write(comando)
    assert esp.angulos is None
    assert esp.leer_respuestas() == ["Formato incorrecto. Usar: angulo1,angulo2,angulo3"]


def test_basura_numerica_se_lee_como_toInt(esp):
    # String::toInt("abc") == 0 en Arduino; "0" esta en rango para el servo 1
    esp.write("abc,45,45\n")
    assert esp.angulos == (0, 45, 45)


def test_varios_comandos_en_una_escritura(esp):
    esp.write("10,30,40\n20,50,60\n")
    assert esp.historial == [(10, 30, 40), (20, 50, 60)]


def test_formatea_enteros_redondeados():
    assert formatear_comando(0.0, 45.6, 179.5) == "0,46,180\n"


def test_sin_puerto_abre_la_virtual(monkeypatch):
    monkeypatch.delenv("ROBOT_PUERTO", raising=False)
    enlace = abrir_enlace()
    assert enlace.nombre == "ESP32 virtual"
    enlace.dispositivo.leer_respuestas()
    assert enlace.enviar(90, 45, 45) == ("90,45,45", [])
    assert enlace.dispositivo.angulos == (90, 45, 45)


def test_virtual_de_n_servos_no_se_traga_valores():
    # CON UN ROBOT DE 4 SERVOS LA VIRTUAL ESPERA 4 VALORES (ANTES SE QUEDABA CON 3 SIN AVISAR)
    esp = ESP32Virtual(rangos=[(0, 180)] * 4)
    esp.leer_respuestas()
    esp.write("90,150,90,45\n")
    assert esp.angulos == (90, 150, 90, 45)
    esp.write("90,150,90\n")
    assert esp.leer_respuestas() == ["Formato incorrecto. Usar: angulo1,angulo2,angulo3,angulo4"]


def test_tres_servos_sigue_siendo_el_firmware():
    # "90,45,45,45" CON EL FIRMWARE DE 3: EL TERCERO SE LEE CON toInt("45,45") = 45
    esp = ESP32Virtual()
    esp.write("90,45,45,45\n")
    assert esp.angulos == (90, 45, 45)
