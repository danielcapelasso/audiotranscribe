import io
import os
import json
from typing import Optional, Literal

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI

# -------- Config --------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️  OPENAI_API_KEY não setada. Configure no painel do Render.")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="Audio Transcriber", version="1.0.0")


class TranscriptionResponse(BaseModel):
    language: Optional[str] = None
    raw_transcript: str
    cleaned_transcript: Optional[str] = None
    summary: Optional[str] = None


CLEAN_PROMPT = """
Você recebe a transcrição (possivelmente com repetições, ruídos e frases truncadas) de uma ligação em espanhol ou português.

Tarefas:
1) Gerar uma versão LIMPA do texto, removendo repetições óbvias, interjeições ("ah", "eh", "mmm"), cortes e frases duplicadas, mantendo o sentido.
2) Produzir um RESUMO curto (3–5 linhas) do que aconteceu na ligação.

Responda EXCLUSIVAMENTE em JSON com este formato:

{
  "cleaned_transcript": "...",
  "summary": "..."
}
"""


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(..., description="Arquivo de áudio (.mp3, .m4a, .wav, etc.)"),
    mode: Literal["literal", "clean"] = Form(default="literal"),
    language_hint: Optional[str] = Form(
        default=None, description="Opcional: 'es', 'pt', 'en'..."
    ),
):
    """
    1) Transcreve o áudio com gpt-4o-mini-transcribe (ou whisper-1 como fallback).
    2) Se mode='clean', gera uma versão limpa + resumo com gpt-4o-mini.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY não configurada no ambiente do Render.",
        )

    # --- Step 1: Transcrição ---
    try:
        audio_bytes = await file.read()
        with io.BytesIO(audio_bytes) as f:
            f.name = file.filename or "audio.wav"
            try:
                transcript = client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=f,
                    language=language_hint,
                )
                raw_text = transcript.text
                lang = getattr(transcript, "language", None) or language_hint
            except Exception as e:
                print(f"⚠️ Falha no gpt-4o-mini-transcribe, tentando whisper-1... ({e})")
                f.seek(0)
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language=language_hint,
                )
                raw_text = transcript.text
                lang = language_hint
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar áudio: {e}")

    # Se o usuário só quer literal, devolve direto
    if mode == "literal":
        return JSONResponse(
            content=TranscriptionResponse(
                language=lang,
                raw_transcript=raw_text,
                cleaned_transcript=None,
                summary=None,
            ).model_dump()
        )

    # --- Step 2: Limpeza simples (opcional) ---
    cleaned = None
    summary = None
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Você é um assistente que limpa transcrições de áudio e gera resumos curtos.",
                },
                {
                    "role": "user",
                    "content": CLEAN_PROMPT
                    + "\n\n--- TRANSCRIÇÃO BRUTA ---\n"
                    + raw_text,
                },
            ],
        )
        data = json.loads(completion.choices[0].message.content)
        cleaned = data.get("cleaned_transcript")
        summary = data.get("summary")
    except Exception as e:
        # Se der erro na limpeza, ainda assim devolvemos a transcrição bruta
        print(f"⚠️ Erro ao limpar transcrição: {e}")

    return JSONResponse(
        content=TranscriptionResponse(
            language=lang,
            raw_transcript=raw_text,
            cleaned_transcript=cleaned,
            summary=summary,
        ).model_dump()
    )


@app.get("/healthz")
def health():
    return {"ok": True}
