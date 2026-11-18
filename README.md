# Robot cosechador de aguacates

Brazo de 3 GDL con visión por computadora: detecta aguacates, calcula el punto de corte del
pedúnculo en milímetros y planea la trayectoria del brazo. Interfaz en PyQt5, simulación 3D en
PyVista y firmware para ESP32. **Funciona sin hardware**: hay una ESP32 virtual que habla el mismo
protocolo que el firmware.

## Correrlo

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv -r requirements.txt
.venv/bin/python -m cosechador                     # o: .venv/bin/python R_COSECHADOR_AUTONOMO.py
.venv/bin/python -m pytest -q tests                # pruebas
```

| Variable | Para qué |
|---|---|
| `ROBOT_CONFIG` | Otro robot: ruta a su YAML (también `--robot ruta.yaml`) |
| `ROBOT_PUERTO` | La ESP32 real, p. ej. `/dev/cu.usbserial-0001`. Sin ella se usa la virtual |
| `ROBOT_CAMARA` | La cámara: URL de la app del celular o índice de webcam. Sin ella, usa CARGAR IMAGEN |

## Cómo está organizado

```
robots/cosechador_3gdl.yaml   TODO lo propio del robot: tabla DH, servos, piezas, cámara, visión
cosechador/
  robot/        descripción (YAML), cinemática directa e inversa, volumen. Sin Qt ni PyVista
  vision/       detección por color y cámara pinhole (píxel -> mm del robot)
  hardware/     enlace con la ESP32 (real por serial o virtual)
  movimiento/   planea trayectorias; devuelve la lista de pasos
  ui/           ventana Qt y escena 3D: lo único que conoce Qt y PyVista
modelos/        STL de las piezas (ver modelos/README.md)
ROBOT_ESP32_CODIGO.ino         firmware
```

La regla: **cada capa solo depende de las de arriba**. `robot/` no sabe que existe una ventana;
`ui/` no calcula nada, solo ejecuta lo que planea `movimiento/` y dibuja lo que dice `robot/`.

## Otro robot

1. Copia `robots/cosechador_3gdl.yaml` y cambia la tabla DH, los servos y las piezas.
2. `python -m cosechador --robot robots/mi_robot.yaml`.

Si hay un error en el archivo, la app dice qué campo está mal. Cualquier brazo serial descrito con
DH se carga, se dibuja y se mueve con los sliders. Calcular trayectorias necesita un solver de
cinemática inversa para su topología (`cinematica_inversa:`). Hoy existe `antropomorfico_3r`.
Ejemplo de otro robot: `tests/robots/brazo_4gdl.yaml`.
