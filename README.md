# 🪡 RAG & LoRA: QA Especializado em Acupuntura Clássica Chinesa

Este repositório contém o pipeline completo de Inteligência Artificial Generativa desenvolvido para responder a perguntas técnicas sobre Acupuntura Clássica Chinesa. O sistema emprega Geração Aumentada por Recuperação (RAG) para gerar um dataset sintético de instrução e, em seguida, aplica *Low-Rank Adaptation* (LoRA) para o *Fine-Tuning* paramétrico eficiente de quatro modelos pretreinados diferentes.

**Modelos Avaliados:**
* 🧠 **Pythia-160M** (Causal)
* 🧠 **GPT-Neo-125M** (Causal)
* 🏆 **Flan-T5 Base** (Seq2Seq) - *Modelo Campeão da Pesquisa*
* 🌐 **mT5 Small** (Seq2Seq)

---

## 📂 Estrutura do Projeto

* `notebooks/`: Cadernos Jupyter contendo a execução lógica das etapas do projeto.
  * `01_rag.ipynb`: Leitura do PDF, processamento e geração do dataset com Gemma-2b-it.
  * `02_lora.ipynb`: Treinamento LoRA dos quatro modelos selecionados.
  * `03_avaliacao_modelo_finetuned.ipynb`: Benchmarking com ROUGE, BLEU e Perplexidade.
* `main.py`: Código-fonte da API construída com FastAPI e inferência dinâmica do HuggingFace.
* `requirements.txt`: Dependências fixas e otimizadas para estabilidade do projeto.

*(Nota: Os arquivos pesados `.zip`, PDFs fontes, datasets e pesos LoRA não estão versionados devido ao limite de tamanho do GitHub).*

---

## 🛠 Instalação Local

Este projeto foi testado em ambiente Linux (Ubuntu). Recomenda-se o uso de um ambiente virtual para isolar as dependências.

1. **Clone o repositório:**
```bash
git clone https://github.com/pauloandrehxh/rag-lora-acupuntura.git
cd rag-lora-acupuntura
```

2. **Crie e ative o ambiente virtual:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Instale as dependências:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*(Caso os pacotes gerem conflito na importação de arquiteturas LoRA, rode `pip install -U peft`)*

---

## 🚀 Execução do Pipeline (Treinamento)

Caso deseje recriar os modelos e treinar o dataset do zero (Idealmente via Google Colab para acesso a GPUs):

1. Faça o upload do seu documento de origem em PDF para a raiz do Colab e rode o `01_rag.ipynb` de cima a baixo para gerar o arquivo `dataset_gerado.jsonl`.
2. Em seguida, rode o `02_lora.ipynb`. O código irá salvar os adaptadores e gerar o `modelos_treinados.zip`.
3. Finalmente, rode o `03_avaliacao_modelo_finetuned.ipynb` para ver a disparidade de performance entre arquiteturas Causais e Seq2Seq.

---

## ⚙️ Execução da API Local (Uso)

Para subir o sistema visual e testar os modelos já treinados, a sua máquina deve possuir as pastas exportadas pelo passo anterior (`lora_causal...` e `tokenizer...`) salvas na pasta raiz.

1. Inicie o servidor FastAPI:
```bash
uvicorn main:app --reload
```
2. O terminal indicará o boot da aplicação e o carregamento do modelo base hospedado no *HuggingFace* entrelaçado aos seus adaptadores LoRA locais.
3. Acesse a Interface de Chat:
   * **Navegador**: [http://localhost:8000](http://localhost:8000)

---

## 🔌 Exemplos de Uso da API via Terminal (cURL)

A API fornece um roteamento dinâmico que permite interrogar qualquer modelo pela porta lógica `/chat`. 

**Requisição de Exemplo:**
```bash
curl -X POST "http://127.0.0.1:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
           "mensagem": "Qual a localização do meridiano Ren-Mai?",
           "modelo": "seq2seq-flant5",
           "max_tokens": 150,
           "temperatura": 0.1
         }'
```

**Resposta (JSON):**
```json
{
  "resposta": "O meridiano Ren-Mai percorre a linha que se encontra entre a gengiva e a mucosa do lábio inferior.",
  "modelo_usado": "seq2seq-flant5",
  "tipo": "seq2seq"
}
```

---
*Projeto acadêmico desenvolvido para a disciplina de Tópicos Avançados em Inteligência Artificial A - UFRN / CERES / DCT.*
