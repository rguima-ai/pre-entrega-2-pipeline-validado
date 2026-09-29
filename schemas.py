"""Contrato de salida del pipeline: qué forma tiene que tener lo que devuelve el LLM."""
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class NivelCriticidad(str, Enum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class EntidadesTecnicas(BaseModel):
    # Las descripciones viajan al LLM como parte del schema de la herramienta.
    # Las restricciones NO: viven en los field_validator, y si el modelo no las cumple,
    # la salida se rechaza y la cadena reintenta.
    tecnologias: list[str] = Field(
        description="Tecnologías, frameworks, lenguajes o herramientas mencionadas explícitamente en el texto"
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        description="Gravedad del problema o relevancia de la arquitectura descripta: baja, media o alta"
    )
    resumen_tecnico: str = Field(
        description="Resumen técnico de 1 o 2 oraciones sobre el contenido del texto"
    )

    @field_validator("tecnologias")
    @classmethod
    def tecnologias_no_vacias_sin_duplicados(cls, v: list[str]) -> list[str]:
        # Duplicado = mismo nombre sin importar mayúsculas. Gana la primera aparición,
        # tal como estaba escrita: ["Redis", "redis"] -> ["Redis"].
        vistas: dict[str, str] = {}
        for tecnologia in (t.strip() for t in v):
            if tecnologia:
                vistas.setdefault(tecnologia.casefold(), tecnologia)
        if not vistas:
            raise ValueError("La lista de tecnologías no puede estar vacía")
        return list(vistas.values())

    @field_validator("nivel_de_criticidad", mode="before")
    @classmethod
    def normalizar_criticidad(cls, v):
        # "Alta " -> "alta". Un valor fuera del enum ("high", "crítica") lo sigue rechazando Pydantic.
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("resumen_tecnico")
    @classmethod
    def resumen_con_contenido(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("El resumen técnico no puede estar vacío")
        return v
