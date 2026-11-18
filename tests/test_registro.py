"""Registro estructurado: eventos con nivel, origen y datos, guardados en JSON Lines."""
import datetime as dt
import json

import numpy as np
import pytest

from cosechador import registro as reg


def reloj_fijo(*fechas):
    horas = iter(fechas)
    ultima = [None]

    def ahora():
        try:
            ultima[0] = next(horas)
        except StopIteration:
            pass
        return ultima[0]
    return ahora


def test_cada_evento_va_al_archivo_del_dia_y_se_relee_igual(tmp_path):
    r = reg.Registro(carpeta=tmp_path, reloj=reloj_fijo(*[dt.datetime(2026, 11, 18, 17, 21, 3, 120000)] * 4))
    r.registrar("aviso", "Jog", "No se mueve: zona inalcanzable",
                destino=np.array([1355.0, 117.0, 1398.0]), falla_en=(1336.4567, 117, 1398))
    ruta = tmp_path / "registro-2026-11-18.jsonl"
    linea = json.loads(ruta.read_text(encoding="utf-8").strip())
    assert linea == {"hora": "2026-11-18T17:21:03.120", "nivel": "aviso", "origen": "Jog",
                     "mensaje": "No se mueve: zona inalcanzable",
                     "datos": {"destino": [1355.0, 117.0, 1398.0], "falla_en": [1336.457, 117, 1398]}}
    leido = reg.leer_jsonl(ruta)[0]
    assert (leido.nivel, leido.origen, leido.mensaje) == ("aviso", "Jog", "No se mueve: zona inalcanzable")


def test_un_archivo_por_dia(tmp_path):
    r = reg.Registro(carpeta=tmp_path, reloj=reloj_fijo(dt.datetime(2026, 11, 18, 23, 59), dt.datetime(2026, 11, 18, 23, 59),
                                                         dt.datetime(2026, 11, 19, 0, 1), dt.datetime(2026, 11, 19, 0, 1)))
    r.registrar("info", "Sistema", "antes de medianoche")
    r.registrar("info", "Sistema", "después")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["registro-2026-11-18.jsonl", "registro-2026-11-19.jsonl"]


def test_nivel_desconocido_no_se_acepta():
    with pytest.raises(ValueError, match="Nivel desconocido"):
        reg.Registro(carpeta=None).registrar("grave", "Sistema", "x")


def test_suscriptores_y_filtro():
    r = reg.Registro(carpeta=None)
    vistos = []
    r.suscribir(vistos.append)
    for nivel in ("info", "ok", "aviso", "error"):
        r.registrar(nivel, "Prueba", nivel)
    assert [e.nivel for e in vistos] == ["info", "ok", "aviso", "error"]
    assert [e.nivel for e in r.filtrar(("aviso", "error"))] == ["aviso", "error"]


def test_la_memoria_no_crece_sin_fin_pero_el_archivo_guarda_todo(tmp_path, monkeypatch):
    monkeypatch.setattr(reg, "MAXIMO_EN_MEMORIA", 5)
    r = reg.Registro(carpeta=tmp_path)
    for i in range(12):
        r.registrar("info", "Prueba", f"evento {i}")
    assert [e.mensaje for e in r.eventos] == [f"evento {i}" for i in range(7, 12)]
    assert len(reg.leer_jsonl(r.archivo)) == 12


def test_exportar(tmp_path):
    r = reg.Registro(carpeta=None)
    r.registrar("ok", "Programa", "terminado", programa="demo")
    reg.exportar_jsonl(r.eventos, tmp_path / "x.jsonl")
    assert reg.leer_jsonl(tmp_path / "x.jsonl")[0].datos == {"programa": "demo"}
