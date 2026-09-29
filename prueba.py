"""Demo: un texto técnico claro y uno ambiguo (prueba de estrés) contra el proveedor de LLM_PROVIDER."""
import asyncio
import logging
import os

from dotenv import load_dotenv

from chain import build_chain, process_text

TEXTO_CLARO = """
Nuestra API en FastAPI está devolviendo timeouts intermitentes. El caché en Redis
parece saturarse en picos de tráfico y las conexiones a PostgreSQL se agotan
porque el pool de SQLAlchemy está mal dimensionado. Esto está afectando a usuarios en producción.
"""

TEXTO_AMBIGUO = "El sistema anda medio raro últimamente, no sé bien qué está pasando."


async def correr(nombre: str, texto: str, chain) -> None:
    print(f"\n=== {nombre} ===")
    try:
        resultado = await process_text(texto, chain=chain)
        print(resultado.model_dump_json(indent=2))
    except Exception as e:  # process_text ya logueó el detalle; acá solo mostramos que el programa sigue
        print(f"❌ No se pudo extraer: {type(e).__name__}: {e}")


async def main() -> None:
    load_dotenv()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # sin esto, la librería HTTP de los SDKs loguea cada request
    for ruidoso in ("httpx", "httpx2", "httpcore"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)

    try:
        chain = build_chain()
    except Exception as e:  # LLM_PROVIDER inválido o falta la API key
        print(f"⚠️ No se pudo configurar el proveedor: {type(e).__name__}: {e}")
        return

    await correr("Texto técnico claro", TEXTO_CLARO, chain)
    await correr("Texto ambiguo (prueba de estrés)", TEXTO_AMBIGUO, chain)


if __name__ == "__main__":
    asyncio.run(main())
