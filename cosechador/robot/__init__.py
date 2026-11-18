"""Modelo del robot: descripción (YAML), cinemática y volumen de trabajo. Sin Qt ni PyVista."""
from .config import ErrorConfiguracion, Robot, cargar_robot, robot_desde_dict

__all__ = ["ErrorConfiguracion", "Robot", "cargar_robot", "robot_desde_dict"]
