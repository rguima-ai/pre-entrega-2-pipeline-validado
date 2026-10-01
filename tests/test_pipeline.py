"""Tests offline: el LLM se reemplaza por un modelo falso que sigue un guion. Sin internet ni API keys.

Los errores de la API son los mismos que lanza ChatOpenAI en producción: las clases de
langchain_openai que envuelven a las del SDK (OpenAIRateLimitError, OpenAIAuthenticationError).
Un openai.RateLimitError crudo NO serviría: no hereda de ModelRateLimitError y with_retry no lo reconocería.
"""
import httpx2  # la librería HTTP que usan por debajo los SDKs actuales de openai y anthropic
import openai
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.exceptions import (
    ModelAuthenticationError,
    ModelRateLimitError,
    OutputParserException,
)
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import OpenAIAuthenticationError, OpenAIRateLimitError
from pydantic import ValidationError

from chain import PROMPT, SalidaInvalidaError, build_chain, get_model, process_text
from schemas import EntidadesTecnicas, NivelCriticidad

TEXTO = "La API en FastAPI tira timeouts porque Redis se satura."
VALIDO = {
    "tecnologias": ["FastAPI", "Redis"],
    "nivel_de_criticidad": "alta",
    "resumen_tecnico": "Timeouts en la API por saturación del caché.",
}


# ---------- pasos del guion: lo que devolvería with_structured_output(include_raw=True) ----------
def respuesta(datos: dict, finish_reason: str = "stop") -> dict:
    """Imita al parser real: si los datos no pasan el schema, parsed=None y parsing_error con el motivo."""
    raw = AIMessage(content="", response_metadata={"finish_reason": finish_reason})
    try:
        return {"raw": raw, "parsed": EntidadesTecnicas.model_validate(datos), "parsing_error": None}
    except ValidationError as e:
        return {"raw": raw, "parsed": None, "parsing_error": e}


def cortada() -> dict:
    # El JSON llegó a parsear, pero el modelo se quedó sin tokens: no se puede confiar en él.
    return respuesta(VALIDO, finish_reason="length")


def cortada_anthropic() -> dict:
    raw = AIMessage(content="", response_metadata={"stop_reason": "max_tokens"})
    return {"raw": raw, "parsed": None, "parsing_error": None}


def json_invalido() -> dict:
    raw = AIMessage(content='{"tecnologias": ["Redis"', response_metadata={"finish_reason": "stop"})
    return {"raw": raw, "parsed": None, "parsing_error": OutputParserException("Invalid json output")}


def _response(status: int) -> httpx2.Response:
    return httpx2.Response(status, request=httpx2.Request("POST", "https://api.openai.com/v1/chat/completions"))


def rate_limit() -> Exception:
    return OpenAIRateLimitError("Rate limit reached", response=_response(429), body=None)


def key_invalida() -> Exception:
    return OpenAIAuthenticationError("Incorrect API key provided", response=_response(401), body=None)


# ---------- modelo falso ----------
class FakeModel:
    """Sigue un guion: cada llamada consume un paso. Si el paso es una excepción, la lanza."""

    def __init__(self, *guion):
        self.guion = list(guion)
        self.llamadas = 0
        self.prompts = []  # lo que le llegó al modelo en cada llamada

    def with_structured_output(self, schema, method=None, include_raw=False):
        assert schema is EntidadesTecnicas and include_raw is True and method == "function_calling"

        def responder(prompt_value):
            self.llamadas += 1
            self.prompts.append(prompt_value)
            paso = self.guion.pop(0)
            if isinstance(paso, Exception):
                raise paso
            return paso

        return RunnableLambda(responder)


def cadena(*guion) -> tuple[FakeModel, object]:
    modelo = FakeModel(*guion)
    return modelo, build_chain(modelo, backoff_inicial_s=0)  # sin espera: los tests no tardan


# ---------- schema ----------
def test_enum_invalido_se_rechaza():
    with pytest.raises(ValidationError):
        EntidadesTecnicas(**{**VALIDO, "nivel_de_criticidad": "high"})


def test_criticidad_se_normaliza():
    assert EntidadesTecnicas(**{**VALIDO, "nivel_de_criticidad": " Alta "}).nivel_de_criticidad is NivelCriticidad.ALTA


@pytest.mark.parametrize("tecnologias", [[], ["", "   "]])
def test_lista_vacia_se_rechaza(tecnologias):
    with pytest.raises(ValidationError, match="no puede estar vacía"):
        EntidadesTecnicas(**{**VALIDO, "tecnologias": tecnologias})


def test_duplicados_se_eliminan_respetando_la_primera_aparicion():
    e = EntidadesTecnicas(**{**VALIDO, "tecnologias": ["Redis", "FastAPI", "redis", " REDIS ", "FastAPI"]})
    assert e.tecnologias == ["Redis", "FastAPI"]


@pytest.mark.parametrize("resumen", ["", "   "])
def test_resumen_vacio_se_rechaza(resumen):
    with pytest.raises(ValidationError):
        EntidadesTecnicas(**{**VALIDO, "resumen_tecnico": resumen})


def test_restricciones_no_estan_en_el_json_schema():
    # Van en field_validator: el schema que ve el LLM solo describe los campos.
    props = EntidadesTecnicas.model_json_schema()["properties"]
    assert "minItems" not in props["tecnologias"] and "minLength" not in props["resumen_tecnico"]


# ---------- prompt ----------
def test_prompt_tiene_roles_y_variable_texto():
    assert PROMPT.input_variables == ["texto"]  # instrucciones_formato ya quedó fija con .partial()
    mensajes = PROMPT.format_messages(texto=TEXTO)
    assert [m.type for m in mensajes] == ["system", "human"]
    assert "nivel_de_criticidad" in mensajes[0].content  # llegaron las instrucciones de formato
    assert TEXTO in mensajes[1].content


# ---------- Factory ----------
def test_factory_openai_temperature_0_sin_reintentos_propios(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-falsa")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    model = get_model("openai")
    assert isinstance(model, ChatOpenAI)
    assert (model.model_name, model.temperature, model.max_retries) == ("gpt-4o-mini", 0, 0)


def test_factory_anthropic_sin_temperature(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-falsa")
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    model = get_model("anthropic")
    assert isinstance(model, ChatAnthropic)
    assert (model.model, model.temperature, model.max_retries) == ("claude-haiku-4-5-20251001", None, 0)


def test_factory_lee_proveedor_y_modelo_del_entorno(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", " Anthropic ")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-falsa")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-otro")
    assert get_model().model == "claude-otro"


def test_factory_rechaza_proveedor_desconocido():
    with pytest.raises(ValueError, match="LLM_PROVIDER inválido"):
        get_model("gemini")


# ---------- pipeline ----------
async def test_camino_feliz():
    modelo, chain = cadena(respuesta(VALIDO))
    resultado = await process_text(TEXTO, chain=chain)
    assert resultado.tecnologias == ["FastAPI", "Redis"]
    assert resultado.nivel_de_criticidad is NivelCriticidad.ALTA
    assert modelo.llamadas == 1
    # el {texto} del ainvoke llegó al modelo, con los dos roles
    assert [m.type for m in modelo.prompts[0].to_messages()] == ["system", "human"]
    assert TEXTO in modelo.prompts[0].to_messages()[1].content


@pytest.mark.parametrize("paso_cortado", [cortada, cortada_anthropic], ids=["openai", "anthropic"])
async def test_respuesta_cortada_se_reintenta_y_se_recupera(paso_cortado):
    modelo, chain = cadena(paso_cortado(), respuesta(VALIDO))
    resultado = await process_text(TEXTO, chain=chain)
    assert resultado.resumen_tecnico == VALIDO["resumen_tecnico"]
    assert modelo.llamadas == 2


async def test_json_invalido_se_reintenta():
    modelo, chain = cadena(json_invalido(), respuesta(VALIDO))
    await process_text(TEXTO, chain=chain)
    assert modelo.llamadas == 2


async def test_salida_que_no_pasa_la_validacion_se_reintenta():
    modelo, chain = cadena(respuesta({**VALIDO, "tecnologias": []}), respuesta(VALIDO))
    await process_text(TEXTO, chain=chain)
    assert modelo.llamadas == 2


async def test_se_rinde_tras_3_intentos():
    modelo, chain = cadena(json_invalido(), cortada(), json_invalido())
    with pytest.raises(SalidaInvalidaError):
        await process_text(TEXTO, chain=chain)
    assert modelo.llamadas == 3


async def test_rate_limit_se_reintenta():
    modelo, chain = cadena(rate_limit(), rate_limit(), respuesta(VALIDO))
    await process_text(TEXTO, chain=chain)
    assert modelo.llamadas == 3


async def test_key_invalida_no_se_reintenta():
    modelo, chain = cadena(key_invalida(), respuesta(VALIDO))
    with pytest.raises(ModelAuthenticationError):
        await process_text(TEXTO, chain=chain)
    assert modelo.llamadas == 1


def test_errores_de_langchain_openai_heredan_de_langchain_core():
    # Si una versión futura cambia esta jerarquía, los reintentos dejarían de funcionar en silencio.
    assert isinstance(rate_limit(), ModelRateLimitError)
    assert isinstance(key_invalida(), ModelAuthenticationError)
    crudo = openai.RateLimitError("Rate limit", response=_response(429), body=None)
    assert not isinstance(crudo, ModelRateLimitError)  # por eso los tests no usan el error crudo


# ---------- integración con los modelos reales de LangChain (sin red) ----------
# Se reemplaza solo _agenerate, el punto donde ChatOpenAI/ChatAnthropic salen a la API.
# Así el parseo y el include_raw son los de verdad, no los del FakeModel.
@pytest.fixture(params=["openai", "anthropic"])
def modelo_real(request, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-falsa")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-falsa")
    clase = {"openai": ChatOpenAI, "anthropic": ChatAnthropic}[request.param]
    guion, pedidos = [], []

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        pedidos.append(kwargs)
        args, fin = guion.pop(0)
        meta = {"finish_reason": fin} if request.param == "openai" else {"stop_reason": fin}
        msg = AIMessage(content="", response_metadata=meta,
                        tool_calls=[{"name": "EntidadesTecnicas", "args": args, "id": "call_1"}])
        return ChatResult(generations=[ChatGeneration(message=msg)])

    monkeypatch.setattr(clase, "_agenerate", _agenerate)
    fin_ok, fin_cortado = ("stop", "length") if request.param == "openai" else ("tool_use", "max_tokens")
    return get_model(request.param), guion, pedidos, fin_ok, fin_cortado


async def test_modelo_real_usa_function_calling(modelo_real):
    # Con json_schema (el default de ChatOpenAI) el SDK valida dentro de la llamada y un
    # ValidationError saldría sin reintentarse. El pedido tiene que ir como herramienta.
    model, guion, pedidos, fin_ok, _ = modelo_real
    guion.append((VALIDO, fin_ok))
    await process_text(TEXTO, chain=build_chain(model, backoff_inicial_s=0))
    assert "tools" in pedidos[0] and "response_format" not in pedidos[0]


async def test_modelo_real_lista_vacia_se_reintenta(modelo_real):
    model, guion, pedidos, fin_ok, _ = modelo_real
    guion.extend([({**VALIDO, "tecnologias": []}, fin_ok), (VALIDO, fin_ok)])
    resultado = await process_text(TEXTO, chain=build_chain(model, backoff_inicial_s=0))
    assert resultado.tecnologias == VALIDO["tecnologias"] and len(pedidos) == 2


async def test_modelo_real_respuesta_cortada_se_reintenta(modelo_real):
    model, guion, pedidos, fin_ok, fin_cortado = modelo_real
    guion.extend([(VALIDO, fin_cortado), (VALIDO, fin_ok)])
    await process_text(TEXTO, chain=build_chain(model, backoff_inicial_s=0))
    assert len(pedidos) == 2


async def test_modelo_real_texto_sin_tecnologias_falla_tras_3_intentos(modelo_real):
    model, guion, pedidos, fin_ok, _ = modelo_real
    guion.extend([({**VALIDO, "tecnologias": []}, fin_ok)] * 3)
    with pytest.raises(SalidaInvalidaError):
        await process_text(TEXTO, chain=build_chain(model, backoff_inicial_s=0))
    assert len(pedidos) == 3


# ---------- logging ----------
async def test_loguea_inicio_validaciones_reintentos_y_resultado(caplog):
    caplog.set_level("INFO", logger="pipeline")
    _, chain = cadena(rate_limit(), json_invalido(), respuesta(VALIDO))
    await process_text(TEXTO, chain=chain)
    mensajes = [(r.levelname, r.getMessage()) for r in caplog.records]
    assert mensajes[0] == ("INFO", f"inicio caracteres={len(TEXTO)}")
    assert any(n == "WARNING" and "llamada_al_modelo=fallida" in m and "Rate limit" in m for n, m in mensajes)
    assert any(n == "WARNING" and "validacion=rechazada" in m for n, m in mensajes)
    assert any(n == "INFO" and m.startswith("validacion=ok") for n, m in mensajes)
    assert mensajes[-1][0] == "INFO" and mensajes[-1][1].startswith("ok ")


async def test_loguea_fallo_definitivo(caplog):
    caplog.set_level("INFO", logger="pipeline")
    _, chain = cadena(key_invalida())
    with pytest.raises(ModelAuthenticationError):
        await process_text(TEXTO, chain=chain)
    ultimo = caplog.records[-1]
    assert ultimo.levelname == "ERROR" and "fallo_definitivo tipo=no_reintentable" in ultimo.getMessage()
