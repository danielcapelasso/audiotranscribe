# Transcriber + Analyzer (Banco Azteca)

Serviço FastAPI pronto para deploy no **Render**. Faz:
1) Transcrição de áudio (`/transcribe`)
2) Limpeza + análise de comportamento (perguntas do gestor, ações do atendente, intenções, recomendações para agente inteligente)

## Deploy no Render
1. Crie um novo **Web Service** (Python) a partir deste ZIP ou repo.
2. Configure a env var **OPENAI_API_KEY**.
3. O `startCommand` já está no `render.yaml`: `uvicorn main:app --host 0.0.0.0 --port $PORT`.
4. Faça o deploy e teste em `/healthz`.

## Testes locais
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
uvicorn main:app --reload
```

## Uso (cURL)
Transcrever + analisar (modo default = clean):
```bash
curl -X POST "http://localhost:8000/transcribe"   -F "file=@/caminho/AudioCobranza1.mp3"   -F "mode=clean"   -F "language_hint=es"
```

Somente transcrição literal:
```bash
curl -X POST "http://localhost:8000/transcribe"   -F "file=@/caminho/AudioCobranza1.mp3"   -F "mode=literal"
```

### Retorno (JSON)
Contém: `language, raw_transcript, cleaned_transcript, summary, speakers, questions_by_manager, common_actions_by_staff, intents_detected, recommended_agent_behaviors`.

## Notas
- Usa `gpt-4o-mini-transcribe` (fallback para `whisper-1`).
- `language_hint` é opcional (`es`, `pt`, `en` etc.).
- Personalize o `CLEAN_PROMPT` em `main.py` conforme necessário.