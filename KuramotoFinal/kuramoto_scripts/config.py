"""
kuramoto_scripts.config
========================

Carga de parametros desde params.json.

Filosofia: TODO lo que el usuario quiere tocar vive en params.json. Este
modulo lo lee, ignora el bloque "_help" (que solo documenta cada campo),
y devuelve un objeto `Params` con acceso por atributo y por seccion.

Tambien deriva cantidades utiles:
    - max_steps = int(t_max / dt)   (tope de seguridad de la parada adaptativa)
"""

import os
import json


class Section:
    """Pequeño contenedor: convierte un dict en acceso por atributo.

    p.general.N  en vez de  p['general']['N'].
    """
    def __init__(self, d):
        self.__dict__.update(d)

    def __repr__(self):
        return f"Section({self.__dict__})"


class Params:
    """Contenedor de todos los parametros de la corrida.

    Atributos principales:
        sim_type   : int (0..4)
        general, convergence, K_sweep : Section
        tipo0_mean_field, tipo1_modular, tipo2_hierarchical,
        tipo3_connectome, tipo4_connectome_weighted : Section
        max_steps  : int derivado (t_max / dt)
    """
    def __init__(self, data):
        self.sim_type = int(data["sim_type"])

        self.general     = Section(data["general"])
        self.convergence = Section(data["convergence"])
        self.K_sweep     = Section(data["K_sweep"])

        self.tipo0_mean_field          = Section(data["tipo0_mean_field"])
        self.tipo1_modular             = Section(data["tipo1_modular"])
        self.tipo2_hierarchical        = Section(data["tipo2_hierarchical"])
        self.tipo3_connectome          = Section(data["tipo3_connectome"])
        self.tipo4_connectome_weighted = Section(data["tipo4_connectome_weighted"])

        # Cantidad derivada: tope de pasos de la integracion.
        self.max_steps = int(round(self.convergence.t_max / self.general.dt))

    def seccion_activa(self):
        """Devuelve la Section correspondiente al sim_type activo."""
        return {
            0: self.tipo0_mean_field,
            1: self.tipo1_modular,
            2: self.tipo2_hierarchical,
            3: self.tipo3_connectome,
            4: self.tipo4_connectome_weighted,
        }[self.sim_type]

    def as_dict_plano(self):
        """Diccionario plano (seccion.campo -> valor) para guardar en params.txt."""
        plano = {"sim_type": self.sim_type, "max_steps": self.max_steps}
        for nombre in ("general", "convergence", "K_sweep", self.nombre_seccion_activa()):
            sec = getattr(self, nombre)
            for k, v in sec.__dict__.items():
                plano[f"{nombre}.{k}"] = v
        return plano

    def nombre_seccion_activa(self):
        return {
            0: "tipo0_mean_field",
            1: "tipo1_modular",
            2: "tipo2_hierarchical",
            3: "tipo3_connectome",
            4: "tipo4_connectome_weighted",
        }[self.sim_type]


def load_params(ruta=None):
    """Lee params.json y devuelve un objeto Params.

    Si `ruta` es None, busca params.json junto al directorio que contiene
    este paquete (es decir, KuramotoFinal/params.json).
    """
    if ruta is None:
        aqui = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ruta = os.path.join(aqui, "params.json")

    with open(ruta, "r", encoding="utf-8") as f:
        data = json.load(f)

    # El bloque _help solo documenta; no es un parametro.
    data.pop("_help", None)

    return Params(data)
