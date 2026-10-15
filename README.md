# RAG PDF Chat

Chat com PDFs usando RAG (Retrieval-Augmented Generation). Você anexa um PDF, faz perguntas em linguagem natural e a IA responde **com base no conteúdo do documento**, lembrando do histórico da conversa.

## Como funciona

1. **Leitura**: o PDF é lido página por página com PyMuPDF (o número da página vai nos metadados).
2. **Chunking**: o texto é quebrado em pedaços de 500 caracteres (overlap de 50) com `RecursiveCharacterTextSplitter`.
3. **Embeddings + índice**: cada chunk vira um vetor com `BAAI/bge-m3` (HuggingFace) e é indexado no FAISS.
4. **Reformulação da pergunta**: um `history_aware_retriever` usa o histórico para transformar perguntas como "e o prazo dele?" em perguntas independentes antes de buscar.
5. **Resposta**: os 6 trechos mais parecidos entram no prompt e a LLM (Qwen via Groq) responde só com o que está no documento.
6. **Cache**: o hash MD5 do arquivo evita reprocessar o mesmo PDF a cada pergunta; o histórico fica em `st.session_state`.

## Stack

Python · Streamlit · LangChain · Groq (Qwen) · FAISS · HuggingFace (bge-m3) · PyMuPDF

## Como rodar

```bash
git clone https://github.com/Vivaraiders/rag-pdf-chat.git
cd rag-pdf-chat

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

python -m pip install -r requirements.txt
```

Crie um arquivo `.env` na raiz (use o `.env.example` como modelo) com sua chave da [Groq](https://console.groq.com/keys):

```
GROQ_API_KEY='sua-chave-aqui'
```

Rode:

```bash
streamlit run main.py
```

> Na primeira execução o modelo de embeddings `BAAI/bge-m3` é baixado (alguns GB), então a primeira pergunta demora mais.

## Observações

A resposta pode sair cortada, porque o limite de tokens é definido na criação da LLM (`max_tokens=500` no `ChatGroq`). Ajuste esse valor no `main.py` conforme a necessidade.

## Autor

Samuel Brandão · [GitHub](https://github.com/Vivaraiders)