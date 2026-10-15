from dotenv import load_dotenv
from os import getenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
import pymupdf
import hashlib

load_dotenv()

#se não existir um historico de chat no session_state, cria um
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

#para não precisar fazer todo o precessor de ler o pdf, quebrar em chunks, embedding e criar o indice FAISS
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

#armazena o hash do arquivo para verificar se ele foi alterado durante uma nova pergunta
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

GROQ_API_KEY = getenv("GROQ_API_KEY")
model = "qwen/qwen3.8-27b"

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=model,
    max_tokens=500,
    temperature=0.6,
    max_retries=2,
    timeout=None
)

system_prompt = """
Você é um assistente prestativo que responde a perguntas com base no conteúdo do documento fornecido.

Regras:
- Responda de forma clara, objetiva e completa.
- Utilize o contexto recuperado do documento para formular suas respostas.
- Se o contexto não contiver informações suficientes para responder, informe que não encontrou informações suficientes no documento.
- Não invente informações nem apresente suposições como fatos.
- Se a pergunta não estiver relacionada ao documento, explique educadamente que seu objetivo é responder a perguntas sobre o conteúdo do arquivo.
- Sempre que possível, preserve os valores, nomes, datas e outros detalhes presentes no documento.
"""



# adicionar o arquivo
file = st.file_uploader("Anexe o pdf", type=['pdf'])

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        st.write(f"**Você:** {message.content}")
    else:
        st.write(f"**IA:** {message.content}")

with st.form("chat form"):
    pergunta = st.text_input("Digite uma pergunta")
    enviar = st.form_submit_button("enviar")

if file:
    file_hash = hashlib.md5(file.getvalue()).hexdigest()

    if (
        st.session_state.vectorstore is None
        or st.session_state.file_hash != file_hash
    ):
        #abrir o pdf
        doc = pymupdf.open(stream=file.getvalue(), filetype='pdf')

        #lista de de documentos onde armazena o numero da pagina e os conteúdos
        documents = []

        #laço para incrementar os conteudos e as paginas na lista de documentos
        for i, pagina in enumerate(doc):
            documents.append(
                Document(
                    page_content=pagina.get_text(),
                    metadata={"page":i + 1}
                )
            )


        #quebra dos documentos e pedaços menores(chunks)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        embedding_model = "BAAI/bge-m3"
        embedding = HuggingFaceEmbeddings(model_name=embedding_model)


        st.session_state.vectorstore = FAISS.from_documents(
            chunks,
            embedding=embedding
        )

        st.session_state.file_hash = file_hash

        st.session_state.vectorstore.save_local("index_faiss")


    retriever =st.session_state.vectorstore.as_retriever(search_type='similarity', search_kwargs={'k':6})


    ######################################################   Alteração da pergunta do usuário para melhor entendimento da IA
    #da uma instrução pra ia para ler a pergunta do usuario e alterar se for preciso utilizando o historico de conversa
    context_q_system_prompt = """Given the following chat history and the follow-up question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""

    #cria um prompt juntando a instrução da ia, o historico de conversa e a pergunta do usuário
    context_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    #reformula a pergunta utilizando context_q_prompt e ja faz a busca no pdf(retriver) para recuperar trechos relevantes | se o historico tiver vazio, ele usa a pergunta original
    history_aware_retriever = create_history_aware_retriever(
        llm= llm, retriever= retriever, prompt=context_q_prompt
    )
    ########################################################
    

    # prompt para a IA pegar o template(agora ela realmente vai responder a pergunta), o historico, e por ultimo(a pergunta e os trechos relevantes trazidos pelo history aware)
    qa_prompt = ChatPromptTemplate([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "Pergunta: {input}\n\n Contexto: {context}")
    ])

    #criação da chain para conectar o prompt à llm, também formata a resposta recebida pelo history_aware para inserir ao context
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    #chain de conexão entre chains, primeiro ele pega os trechos relevantes, envia pra outra chain onde está o prompt
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)


    if enviar and pergunta.strip():
            resposta = rag_chain.invoke({
                'input': pergunta,
                "chat_history": st.session_state.chat_history
            })

            st.session_state.chat_history.append(
                HumanMessage(content=pergunta)
            )

            st.session_state.chat_history.append(
                AIMessage(content=resposta['answer'])
            )

            st.write(resposta['answer'])