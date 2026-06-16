# =============================================================================
# LABORATÓRIO: Pipeline RAG + LoRA — API RESTful com FastAPI
# =============================================================================
# Integra 4 modelos fine-tunados com LoRA:
#   - causal-pythia   → EleutherAI/pythia-160m   (text-generation)
#   - causal-gptneo   → EleutherAI/gpt-neo-125m  (text-generation)
#   - seq2seq-flant5  → google/flan-t5-base       (text2text-generation)
#   - seq2seq-mt5     → google/mt5-small          (text2text-generation)
#
# Fallback automático para o modelo base se os adaptadores LoRA não existirem.
# =============================================================================

import os
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# INSTÂNCIA FASTAPI
# =============================================================================
app = FastAPI(
    title="RAG + LoRA Lab — API de Modelos de Linguagem",
    description=(
        "API RESTful com 4 modelos fine-tunados via LoRA:\n"
        "- **causal-pythia**: EleutherAI/pythia-160m\n"
        "- **causal-gptneo**: EleutherAI/gpt-neo-125m\n"
        "- **seq2seq-flant5**: google/flan-t5-base\n"
        "- **seq2seq-mt5**: google/mt5-small\n\n"
        "Todos com fallback automático para o modelo base caso os adaptadores LoRA não existam."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# REGISTRO GLOBAL DOS MODELOS
# =============================================================================
MODELS: dict = {}

# Metadados descritivos (usados no endpoint /modelos)
MODELOS_META = {
    "causal-pythia": {
        "id": "causal-pythia",
        "nome": "Pythia 160M (Causal)",
        "descricao": "EleutherAI/pythia-160m — Causal LM fine-tunado com LoRA no domínio do dataset."
    },
    "causal-gptneo": {
        "id": "causal-gptneo",
        "nome": "GPT-Neo 125M (Causal)",
        "descricao": "EleutherAI/gpt-neo-125m — Causal LM com atenção Local+Global, fine-tunado com LoRA."
    },
    "seq2seq-flant5": {
        "id": "seq2seq-flant5",
        "nome": "Flan-T5 Base (Seq2Seq)",
        "descricao": "google/flan-t5-base — Encoder-Decoder instrução-tunado em 1800+ tasks, LoRA no domínio."
    },
    "seq2seq-mt5": {
        "id": "seq2seq-mt5",
        "nome": "mT5 Small (Seq2Seq)",
        "descricao": "google/mt5-small — Seq2Seq multilíngue (101 idiomas), fine-tunado com LoRA."
    },
}

# =============================================================================
# FUNÇÃO DE CARREGAMENTO DOS MODELOS
# =============================================================================

def carregar_modelo(nome_id: str, base_model_name: str, tipo: str,
                    path_lora: str, path_token: str) -> dict:
    """
    Carrega um modelo (causal ou seq2seq) com suporte a adaptadores LoRA.

    Parâmetros
    ----------
    nome_id        : chave no dicionário MODELS
    base_model_name: ID HuggingFace do modelo base (fallback)
    tipo           : 'causal' ou 'seq2seq'
    path_lora      : diretório local dos adaptadores LoRA
    path_token     : diretório local do tokenizador salvo

    Retorna
    -------
    dict com 'model', 'tokenizer', 'pipeline', 'tipo'
    """
    logger.info(f"Carregando '{nome_id}' ({base_model_name}) — tipo: {tipo}")

    # Determina se usa modelo fine-tunado ou base (fallback)
    if os.path.exists(path_lora):
        model_path = path_lora
        tok_path = path_token if os.path.exists(path_token) else path_lora
        logger.info(f"  → Usando adaptadores LoRA: {path_lora}")
    else:
        model_path = base_model_name
        tok_path = base_model_name
        logger.warning(
            f"  → '{path_lora}' não encontrado. Usando modelo base como fallback."
        )

    # Tokenizador
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    if tipo == "causal" and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("  → pad_token = eos_token (causal)")

    # Modelo
    if tipo == "causal":
        model = AutoModelForCausalLM.from_pretrained(model_path)
        pipe_type = "text-generation"
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        pipe_type = "text-generation"

    model.eval()
    pipe = pipeline(pipe_type, model=model, tokenizer=tokenizer, device=-1)

    logger.info(f"  ✓ '{nome_id}' carregado com sucesso! (pipeline: {pipe_type})")
    return {"model": model, "tokenizer": tokenizer, "pipeline": pipe, "tipo": tipo}


# =============================================================================
# EVENTO DE STARTUP — carrega todos os modelos uma vez
# =============================================================================

@app.on_event("startup")
async def startup_event():
    global MODELS
    logger.info("=" * 60)
    logger.info("  INICIANDO SERVIDOR — carregando 4 modelos LoRA")
    logger.info("=" * 60)

    MODELS["causal-pythia"] = carregar_modelo(
        "causal-pythia", "EleutherAI/pythia-160m", "causal",
        "./lora_causal_pythia_160m", "./tokenizer_pythia_160m"
    )
    MODELS["causal-gptneo"] = carregar_modelo(
        "causal-gptneo", "EleutherAI/gpt-neo-125m", "causal",
        "./lora_causal_gpt_neo_125m", "./tokenizer_gpt_neo_125m"
    )
    MODELS["seq2seq-flant5"] = carregar_modelo(
        "seq2seq-flant5", "google/flan-t5-base", "seq2seq",
        "./lora_seq2seq_flan_t5_base", "./tokenizer_flan_t5_base"
    )
    MODELS["seq2seq-mt5"] = carregar_modelo(
        "seq2seq-mt5", "google/mt5-small", "seq2seq",
        "./lora_seq2seq_mt5_small", "./tokenizer_mt5_small"
    )

    logger.info("=" * 60)
    logger.info(f"  ✓ {len(MODELS)} modelo(s) disponível(is): {list(MODELS.keys())}")
    logger.info("=" * 60)


# =============================================================================
# SCHEMAS PYDANTIC
# =============================================================================

class ChatRequest(BaseModel):
    modelo: str
    mensagem: str
    max_tokens: Optional[int] = 150
    temperatura: Optional[float] = 0.7


class ChatResponse(BaseModel):
    resposta: str
    modelo: str
    tokens_gerados: int


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/modelos", response_class=JSONResponse, tags=["Modelos"])
async def listar_modelos():
    """
    GET /modelos

    Retorna a lista de modelos disponíveis (apenas os carregados com sucesso).
    """
    disponiveis = [
        meta for key, meta in MODELOS_META.items()
        if key in MODELS
    ]
    return {"modelos": disponiveis}


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    POST /chat

    Envia uma mensagem para o modelo selecionado e retorna a resposta gerada.

    - Para modelos **Causais**: o texto gerado inclui o prompt; extraímos apenas
      a parte após a mensagem original.
    - Para modelos **Seq2Seq**: a saída é diretamente a resposta gerada.
    """
    if request.modelo not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo '{request.modelo}' não encontrado. "
                   f"Disponíveis: {list(MODELS.keys())}"
        )
    if not request.mensagem.strip():
        raise HTTPException(status_code=400, detail="A mensagem não pode ser vazia.")

    logger.info(f"[CHAT] modelo={request.modelo} | msg='{request.mensagem[:60]}...'")

    pipe_info = MODELS[request.modelo]
    pipe = pipe_info["pipeline"]
    tokenizer = pipe_info["tokenizer"]
    tipo = pipe_info["tipo"]

    try:
        resultado = pipe(
            request.mensagem,
            max_new_tokens=request.max_tokens,
            temperature=request.temperatura,
            do_sample=True,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
        )

        # O pipeline "text-generation" retorna prompt + geração para ambos na versão mais nova
        texto_completo = resultado[0]["generated_text"]
        
        # Garante que removemos o prompt da resposta final
        if texto_completo.startswith(request.mensagem):
            resposta = texto_completo[len(request.mensagem):].strip()
        else:
            resposta = texto_completo.strip()

        if not resposta:
            resposta = "[O modelo não gerou texto adicional. Tente aumentar max_tokens.]"

        tokens_gerados = len(tokenizer.encode(resposta))
        logger.info(f"  ✓ {tokens_gerados} tokens gerados")

        return ChatResponse(
            resposta=resposta,
            modelo=request.modelo,
            tokens_gerados=tokens_gerados
        )

    except Exception as e:
        logger.error(f"Erro na geração: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao gerar resposta: {e}")


@app.get("/health", tags=["Saúde"])
async def health_check():
    """
    GET /health

    Verifica quais modelos estão carregados e prontos para uso.
    """
    return {
        "status": "ok",
        "modelos_carregados": list(MODELS.keys()),
        "quantidade": len(MODELS),
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu"
    }


# =============================================================================
# FRONT-END ESTÁTICO
# =============================================================================
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse, tags=["Front-end"])
async def root():
    """Serve a interface web do chat."""
    html_path = os.path.join("static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
