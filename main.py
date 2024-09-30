from flask import Flask, request, jsonify
import bs4
import os
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory

app = Flask(__name__)

load_dotenv()

# Carregamento de documentos
loader = WebBaseLoader(web_paths=[
    'https://goias.gov.br/social/programa-de-agua-e-energia/',
    'https://goias.gov.br/social/programa-aprendiz-do-futuro/',
    'https://goias.gov.br/social/programa-auxilio-nutricional/',
    'https://goias.gov.br/social/carteira-de-identificacao-do-autista/',
    'https://goias.gov.br/social/cestas-basicas/',
    'https://goias.gov.br/social/cofinanciamento-estadual/',
    'https://goias.gov.br/social/programa-credito-social/',
    'https://goias.gov.br/social/dignidade/',
    'https://goias.gov.br/social/dignidade-menstrual/',
    'https://goias.gov.br/social/familia-acolhedora-goiana/',
    'https://goias.gov.br/social/goias-por-elas/',
    'https://goias.gov.br/social/programa-maes-de-goias-lista-de-beneficiarias/',
    'https://goias.gov.br/social/passaporte-do-idoso/',
    'https://goias.gov.br/social/passe-livre-da-pessoa-com-deficiencia/',
    'https://goias.gov.br/social/passe-livre-estudantil-2/',
    'https://goias.gov.br/social/registro-civil/',
    'https://goias.gov.br/social/vigilancia-socioassistencial/',
    'https://goias.gov.br/social/gestao-do-trabalho-e-educacao-permanente-no-suas/',
    'https://goias.gov.br/social/cadastro-unico-e-programa-bolsa-familia/'
],
                       bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                           class_=("entry-title", "entry-content"))))
docs = loader.load()

# Divisão de documentos
divisor = RecursiveCharacterTextSplitter(chunk_size=1500,
                                         chunk_overlap=300,
                                         add_start_index=True)
documento_dividido = divisor.split_documents(docs)

# Criação de vetores de documentos
vectorstore = Chroma.from_documents(documents=documento_dividido,
                                    embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# Definição dos prompts
contextualize_q_system_prompt = (
    '''Dado um histórico de chat e a pergunta mais recente do usuário, 
    que pode referenciar o contexto no histórico do chat, 
    formule uma pergunta independente que possa ser entendida sem o histórico do chat. 
    NÃO responda à pergunta, apenas reformule-a se necessário e, 
    caso contrário, retorne-a como está.''')

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

system_prompt = (
'''Sempre se apresente como GAL, a assistente virtual especializada em fornecer informações sobre programas do Goiás Social. Você deve responder apenas a perguntas relacionadas a saudações, criação de poesias, sua identidade e programas sociais dentro do contexto fornecido. Não responda a perguntas fora desse escopo. Se o usuário fizer uma pergunta fora do escopo, responda educadamente que você só pode fornecer informações sobre programas sociais. Caso não encontre diga que não encontrou o programa na sua base de dados.
'''
"\n\n"
"{context}")

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Configuração do LLM e dos chains
llm = ChatOpenAI(model="gpt-4o-mini")
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever,
                                   question_answer_chain)


# Rotas Flask
@app.route('/')
def homepage():
    return "A API ESTÁ NO AR"


@app.route('/ask', methods=['POST'])
def ask():
    chat_history = ChatMessageHistory()
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda: chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    if request.is_json:
        data = request.get_json()
        input_data = data.get('input')
        chat_history_data = data.get('chat_history', [])

        if not input_data:
            return jsonify({'error': 'Input is required'}), 400

        # Adiciona o histórico do chat à variável chat_history
        for message in chat_history_data:
            chat_history.add_message(message)

        response = conversational_rag_chain.invoke({"input": input_data})
        return jsonify({'answer': response['answer']})
    else:
        return jsonify({'error': 'Content-Type must be application/json'}), 415


app.run(host='0.0.0.0')
