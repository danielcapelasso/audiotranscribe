
import io
import os
from typing import Optional, Literal

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI

# -------- Config --------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # Don't crash at import time on Render; just warn at runtime if missing
    print("⚠️  OPENAI_API_KEY not set. Set it in your Render service environment.")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="Audio Transcriber + Behavior Analyzer", version="1.0.0")

class AnalyzeResponse(BaseModel):
    language: Optional[str] = None
    raw_transcript: str
    cleaned_transcript: str
    summary: str
    speakers: list[str]
    questions_by_manager: list[str]
    common_actions_by_staff: list[str]
    intents_detected: list[str]
    recommended_agent_behaviors: list[str]

CLEAN_PROMPT = """
Você é um assistente que recebe a transcrição (possivelmente ruidosa) de uma ligação de cobrança/visita do Banco Azteca
envolvendo um gestor (field manager), um funcionário/atendente e às vezes o cliente.

Tarefas:
1) Limpar o texto: remover ruídos, repetições, preenchimentos ("é...", "ah..."), mantendo o sentido original.
2) Padronizar nomes dos falantes quando possível: Gestor, Funcionário, Cliente. Se incerto, use "Indefinido".
3) Produzir um RESUMO objetivo (3-6 linhas).
4) Extrair LISTAS:
   - Perguntas recorrentes do Gestor (em bullet points, frases curtas).
   - Ações ou passos recorrentes do Funcionário/Atendente (em bullets).
   - Intenções detectadas (ex.: localizar cliente, confirmar endereço, agendar visita, validar referência).
5) Sugerir 6-10 COMPORTAMENTOS para um Agente Inteligente que ajuda o gestor a encontrar o cliente. Foque em:
   - Perguntas iniciais de qualificação (endereço, ponto de referência, disponibilidade, telefone alternativo).
   - Estratégias quando o cliente não atende (contato com vizinho, referência, horário alternativo, replanejamento).
   - Confirmações e encerramento (rota, landmarks, confirmação de horário, recados).

Saída em JSON com as chaves:
language, cleaned_transcript, summary, speakers, questions_by_manager, common_actions_by_staff, intents_detected, recommended_agent_behaviors.
"""

ANALYZE_SYSTEM = "Você é um analista sênior de operações de campo do Banco Azteca."
ANALYZE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "analysis",
        "schema": {
            "type": "object",
            "properties": {
                "language": {"type": "string"},
                "cleaned_transcript": {"type": "string"},
                "summary": {"type": "string"},
                "speakers": {"type": "array", "items": {"type": "string"}},
                "questions_by_manager": {"type": "array", "items": {"type": "string"}},
                "common_actions_by_staff": {"type": "array", "items": {"type": "string"}},
                "intents_detected": {"type": "array", "items": {"type": "string"}},
                "recommended_agent_behaviors": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["cleaned_transcript","summary","speakers","questions_by_manager","common_actions_by_staff","intents_detected","recommended_agent_behaviors"],
            "additionalProperties": False
        }
    }
}

@app.post("/transcribe", response_model=AnalyzeResponse)
async def transcribe_and_analyze(
    file: UploadFile = File(..., description="Arquivo de áudio (.mp3, .m4a, .wav, etc.)"),
    mode: Literal["clean","literal"] = Form(default="clean"),
    language_hint: Optional[str] = Form(default=None, description="Opcional: 'es', 'pt', 'en'...")
):
    """
    1) Transcreve o áudio com gpt-4o-mini-transcribe (ou whisper-1 como fallback).
    2) Se mode='clean', roda uma limpeza + análise com gpt-4o-mini.
    3) Retorna JSON com transcrição crua, limpa e análises.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY não configurada no ambiente.")

    # --- Step 1: Transcrição ---
    try:
        audio_bytes = await file.read()
        # Passa para a API
        import io as _io
        with _io.BytesIO(audio_bytes) as f:
            f.name = file.filename or "audio.wav"
            try:
                transcript = client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=f,
                    language=language_hint  # pode ser None; a API detecta automaticamente
                )
                raw_text = transcript.text
                lang = getattr(transcript, "language", None) or language_hint
            except Exception as e:
                # Fallback para whisper-1
                print(f"⚠️ Falha no gpt-4o-mini-transcribe, tentando whisper-1... ({e})")
                f.seek(0)
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language=language_hint
                )
                raw_text = transcript.text
                lang = language_hint
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar áudio: {e}")

    # Se o usuário só quer literal, não processa análise completa
    if mode == "literal":
        return JSONResponse(
            content=AnalyzeResponse(
                language=lang,
                raw_transcript=raw_text,
                cleaned_transcript=raw_text,
                summary="",
                speakers=[],
                questions_by_manager=[],
                common_actions_by_staff=[],
                intents_detected=[],
                recommended_agent_behaviors=[]
            ).model_dump()
        )

    # --- Step 2: Limpeza + Análise ---
    try:
        # Usamos Responses API para gerar JSON estruturado com a limpeza + insights.
        resp = client.responses.create(
            model="gpt-4o-mini",
            reasoning={"effort":"medium"},
            input=[
                {"role":"system", "content": ANALYZE_SYSTEM},
                {"role":"user", "content": [
                    {"type":"text", "text": CLEAN_PROMPT},
                    {"type":"input_text", "text": raw_text}
                ]}
            ],
            response_format=ANALYZE_FORMAT
        )
        # Extrai o JSON de saída
        output = resp.output[0].content[0].input_json  # structured output
        cleaned = output.get("cleaned_transcript","").strip()
        summary = output.get("summary","").strip()
        speakers = output.get("speakers", [])
        questions = output.get("questions_by_manager", [])
        actions = output.get("common_actions_by_staff", [])
        intents = output.get("intents_detected", [])
        recs = output.get("recommended_agent_behaviors", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao analisar transcrição: {e}")

    return JSONResponse(
        content=AnalyzeResponse(
            language=lang,
            raw_transcript=raw_text,
            cleaned_transcript=cleaned or raw_text,
            summary=summary,
            speakers=speakers,
            questions_by_manager=questions,
            common_actions_by_staff=actions,
            intents_detected=intents,
            recommended_agent_behaviors=recs
        ).model_dump()
    )

@app.get("/healthz")
def health():
    return {"ok": True}
