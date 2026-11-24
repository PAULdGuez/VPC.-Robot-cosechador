# Piezas 3D del robot

Pon aquí un STL por pieza. **Qué archivo es cada pieza lo dice el YAML del robot**
(`robots/cosechador_3gdl.yaml`, campos `pieza:` de la base, de cada articulación y del efector).
La simulación usa el STL si existe y, si no, una pieza provisional armada con la tabla DH.
**No hay que tocar código**: exportas, guardas aquí y vuelves a abrir la app.

Para el cosechador:

| Archivo | Qué es | Se mueve con |
|---|---|---|
| `base.stl` | La base fija | — |
| `columna.stl` | La torre que gira | servo 1 |
| `brazo.stl` | Del hombro al codo | servo 2 |
| `antebrazo.stl` | Del codo a la punta | servo 3 |
| `efector.stl` | La tijera o cortador | (va con el antebrazo) |

## La regla de oro: dónde poner el origen de cada pieza

Cada pieza se modela **en su propio documento (o Part Studio)**, con el origen en su eje de giro.
Si el origen queda en otro lado, la pieza aparece desfasada o girada.

- **base**: origen en el centro del piso; Z hacia arriba.
- **columna**: origen en el piso, sobre el eje de giro; Z hacia arriba. Su punta llega a **350 mm**
  (ahí va el eje del hombro, que es horizontal).
- **brazo**: origen **en el eje del hombro**; el brazo se extiende sobre **+X** hasta **1000 mm**
  (el codo). El eje del hombro es **Z**.
- **antebrazo**: origen **en el eje del codo**; se extiende sobre **+X** hasta **700 mm**. El eje
  del codo es **Z**.
- **efector**: origen en el extremo del antebrazo, apuntando a **+X**; su punta queda a `efector.largo` mm del origen
  (el largo que dice el archivo del robot: a esa punta van las coordenadas que se piden).

En general, para cualquier robot del YAML: **la pieza de la articulación i tiene su origen en el eje
de giro i (que es Z), sube `d` sobre Z y luego avanza `a` sobre +X** hasta el eje de la siguiente.

Las medidas 350 / 1000 / 700 viven en la tabla DH del YAML. Si tu diseño cambia una medida,
cámbiala ahí, para que la cinemática y el dibujo sigan coincidiendo.

## Exportar desde Onshape

1. Clic derecho en la pieza (en la lista de Parts) → **Export**.
2. Format: **STL**. Units: **Millimeter**. Resolution: **Fine** (o Medium si pesa mucho).
3. Guárdalo aquí con el nombre de la tabla.

En Fusion 360: clic derecho en el cuerpo → **Save As Mesh** → STL, unidades en **mm**.

**Para comprobarlo:** corre `.venv/bin/python -m pytest -q tests`. Al abrir la app, en la terminal
puedes revisar qué piezas tomó del STL con `ventana.escena.origen_piezas`.
