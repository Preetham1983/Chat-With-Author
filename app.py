# from flask import Flask, request, jsonify
# import os
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_community.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import nltk
# from nltk.corpus import stopwords
# from langchain_community.llms import CTransformers
# from googletrans import Translator
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Load environment variables
# load_dotenv()
# google_api_key = os.getenv("GOOGLE_API_KEY")
# if not google_api_key:
#     raise ValueError("Google API key not found. Please check your environment variables.")

# # Download stopwords
# nltk.download('stopwords')
# stop_words = stopwords.words('english')
# custom_stopwords = ["what", "is", "how", "who", "explain", "about", "?", "please", "hey", "whatsup", "can u explain"]
# stop_words.extend(custom_stopwords)

# @app.route('/upload-pdf', methods=['POST'])
# def upload_pdf():
#     pdf_files = request.files.getlist('pdf_files')
#     if not pdf_files:
#         return jsonify({'error': 'No PDF files uploaded'}), 400
    
#     raw_text = get_pdf_text(pdf_files)
#     text_chunks = get_text_chunks(raw_text)
#     get_vector_store(text_chunks)
    
#     return jsonify({'message': 'PDFs uploaded and processed successfully'}), 200

# @app.route('/process-query', methods=['POST'])
# def process_query():
#     data = request.json
#     if 'user_question' not in data:
#         return jsonify({'error': 'Missing user_question field'}), 400
    
#     user_question = data['user_question']
#     response_language = data.get('response_language', 'en')
    
#     try:
#         raw_text = get_pdf_text()  # Fetch raw text from wherever it is stored
        
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#         docs = new_db.similarity_search(user_question)
        
#         gemini_chain = get_conversational_chain()
#         gemini_response = gemini_chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#         initial_response = gemini_response["output_text"]
        
#         similarity_score = calculate_cosine_similarity(raw_text, user_question)
    
#         if "The answer is not available in the context" in initial_response or "The provided context does not contain any information" in initial_response:
#             if similarity_score > 0.00125:
#                 refined_response = get_llama_response(user_question, no_words=500, blog_style="detailed", response_language="english")
#             else:
#                 refined_response = "I'm sorry, I cannot answer this question based on the provided context."
#         else:
#             refined_response = get_llama_response(initial_response, no_words=500, blog_style="detailed", response_language="english")
        
#         translated_response = translate_text(refined_response, response_language)
    
#         return jsonify({'generated_response': translated_response}), 200
    
#     except ValueError as ve:
#         return jsonify({'error': str(ve)}), 400  # Return 400 for client errors
#     except Exception as e:
#         app.logger.error(f"Error processing query: {e}")  # Log the error
#         return jsonify({'error': 'Internal Server Error'}), 500  # Return 500 for server errors

# def get_pdf_text(pdf_files):
#     text = ""
#     for pdf in pdf_files:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text() or ""
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     return text_splitter.split_text(text)

# def get_vector_store(text_chunks):
#     try:
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#         vector_store.save_local("faiss_index")
#     except Exception as e:
#         print(f"Error during embedding: {e}")

# def get_conversational_chain():
#     prompt_template = """
#     Please provide a detailed answer based on the provided context. If the necessary information to answer the question is not present in the context, respond with 'The answer is not available in the context'
    
#     Context:
#     {context}
    
#     Question:
#     {question}
    
#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# def get_llama_response(input_text, no_words, blog_style, response_language):
#     llm = CTransformers(
#         model='C:/Users/bandi/OneDrive/Desktop/LLama/chat-with-author-backend/models/llama-2-7b-chat.ggmlv3.q8_0.bin',
#         model_type='llama',
#         config={'max_new_tokens': 500, 'temperature': 0.01}
#     )
#     template = """
#     Given some information of '{input_text}', provide a concise summary suitable for a {blog_style} blog post in approximately {no_words} words. The total response should be in {response_language} language. Focus on key aspects and provide accurate information.
#     """
    
#     prompt = PromptTemplate(input_variables=["blog_style", "input_text", 'no_words', 'response_language'],
#                             template=template)
    
#     response = llm(prompt.format(input_text=input_text, no_words=no_words, blog_style=blog_style, response_language=response_language))
#     return response

# def calculate_cosine_similarity(raw_text, user_question):
#     vectorizer = TfidfVectorizer(stop_words=list(stop_words))
#     tfidf_matrix = vectorizer.fit_transform([raw_text, user_question])
#     cos_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
#     return cos_similarity

# def translate_text(text, dest_language):
#     translator = Translator()
#     translation = translator.translate(text, dest=dest_language)
#     return translation.text

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, request, jsonify
# import os
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_community.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import nltk
# from nltk.corpus import stopwords
# from langchain_community.llms import CTransformers
# from googletrans import Translator
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

# # Load environment variables
# load_dotenv()
# google_api_key = os.getenv("GOOGLE_API_KEY")
# if not google_api_key:
#     raise ValueError("Google API key not found. Please check your environment variables.")

# # Download stopwords
# nltk.download('stopwords')
# stop_words = stopwords.words('english')
# custom_stopwords = ["what", "is", "how", "who", "explain", "about", "?", "please", "hey", "whatsup", "can u explain"]
# stop_words.extend(custom_stopwords)

# # Variable to store extracted text globally (not recommended for production, use a database or file storage)
# global_raw_text = ""

# @app.route('/upload-pdf', methods=['POST'])
# def upload_pdf():
#     global global_raw_text
    
#     pdf_files = request.files.getlist('pdf_files')
#     if not pdf_files:
#         return jsonify({'error': 'No PDF files uploaded'}), 400
    
#     raw_text = get_pdf_text(pdf_files)
#     global_raw_text = raw_text  # Store raw text globally for use in queries
#     text_chunks = get_text_chunks(raw_text)
#     get_vector_store(text_chunks)
    
#     return jsonify({'message': 'PDFs uploaded and processed successfully'}), 200

# @app.route('/process-query', methods=['POST'])
# def process_query():
#     data = request.json
#     if 'user_question' not in data:
#         return jsonify({'error': 'Missing user_question field'}), 400
    
#     user_question = data['user_question']
   
    
#     try:
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#         docs = new_db.similarity_search(user_question)
        
#         global global_raw_text
#         global_raw_text = global_raw_text.strip()  # Ensure raw text is clean and stripped
#         similarity_score = calculate_cosine_similarity(global_raw_text, user_question)
        
#         gemini_chain = get_conversational_chain()
#         gemini_response = gemini_chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#         initial_response = gemini_response["output_text"]
        
#     except Exception as e:
#         initial_response = f"Error: {str(e)}"  # Log the specific error for debugging
    
#     if "The answer is not available in the context" in initial_response or "The provided context does not contain any information" in initial_response:
#         if similarity_score > 0.00125:  # Adjust this threshold as needed
#             refined_response = get_llama_response(user_question, no_words=500, blog_style="detailed")
#         else:
#             refined_response = ""
#     else:
#         refined_response = get_llama_response(initial_response, no_words=500, blog_style="detailed")
    
   
    
#     return jsonify({'generated_response': refined_response}), 200


# def get_pdf_text(pdf_files):
#     text = ""
#     for pdf in pdf_files:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text() or ""
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     return text_splitter.split_text(text)

# def get_vector_store(text_chunks):
#     try:
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#         vector_store.save_local("faiss_index")
#     except Exception as e:
#         print(f"Error during embedding: {e}")

# def get_conversational_chain():
#     prompt_template = """
#     Please provide a detailed answer based on the provided context. If the necessary information to answer the question is not present in the context, respond with 'The answer is not available in the context'
    
#     Context:
#     {context}
    
#     Question:
#     {question}
    
#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# def get_llama_response(input_text, no_words, blog_style):
#     llm = CTransformers(
#         model='C:/Users/bandi/OneDrive/Desktop/LLama/chat-with-author-backend/models/llama-2-7b-chat.ggmlv3.q8_0.bin',
#         model_type='llama',
#         config={'max_new_tokens': 500, 'temperature': 0.01}
#     )
#     template = """
#           Given some information of '{input_text}', provide a concise summary suitable for a {blog_style} blog post in approximately {no_words} words. Focus on key aspects and provide accurate information.
#     """
    
#     prompt = PromptTemplate(input_variables=["blog_style", "input_text", 'no_words'],
#                             template=template)
    
#     response = llm(prompt.format(input_text=input_text, no_words=no_words, blog_style=blog_style))
#     return response

# def calculate_cosine_similarity(text, user_question):
#     vectorizer = TfidfVectorizer(stop_words=list(stop_words))
#     tfidf_matrix = vectorizer.fit_transform([text, user_question])
#     cos_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
#     return cos_similarity



# if __name__ == '__main__':
#     app.run(debug=False)








from flask import Flask, request, jsonify
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from langchain_community.llms import CTransformers
from googletrans import Translator
from flask_cors import CORS
from pymongo import MongoClient

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("Google API key not found. Please check your environment variables.")

# Download stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')
custom_stopwords = ["what", "is", "how", "who", "explain", "about", "?", "please", "hey", "whatsup", "can u explain"]
stop_words.extend(custom_stopwords)


mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    raise ValueError("MongoDB URI not found. Please check your environment variables.")
client = MongoClient(mongo_uri)
db = client['pdf_database']
collection = db['text_chunks']


global_raw_text = ""

@app.route('/upload-pdf', methods=['POST'])
# def upload_pdf():
#     global global_raw_text
    
#     pdf_files = request.files.getlist('pdf_files')
#     if not pdf_files:
#         return jsonify({'error': 'No PDF files uploaded'}), 400
    
#     raw_text = get_pdf_text(pdf_files)
#     global_raw_text = raw_text
#     text_chunks = get_text_chunks(raw_text)
#     get_vector_store(text_chunks)
    
#     return jsonify({'message': 'PDFs uploaded and processed successfully'}), 200
def upload_pdf():
    global global_raw_text

    pdf_files = request.files.getlist('pdf_files')
    if not pdf_files:
        return jsonify({'error': 'No PDF files uploaded'}), 400

    raw_text = get_pdf_text(pdf_files)
    global_raw_text = raw_text
    text_chunks = get_text_chunks(raw_text)
    
    # Store text chunks in MongoDB
    store_chunks_in_mongodb(text_chunks)
    
    # Process vector store
    get_vector_store(text_chunks)
    
    return jsonify({'message': 'PDF processed and text chunks stored'}), 200

def store_chunks_in_mongodb(chunks):
    documents = [{"chunk": chunk} for chunk in chunks]
    collection.insert_many(documents)

@app.route('/process-query', methods=['POST'])
def process_query():
    data = request.json
    if 'user_question' not in data:
        return jsonify({'error': 'Missing user_question field'}), 400
    
    user_question = data['user_question']
   
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        global global_raw_text
        global_raw_text = global_raw_text.strip()  # Ensure raw text is clean and stripped
        similarity_score = calculate_cosine_similarity(global_raw_text, user_question)
        
        gemini_chain = get_conversational_chain()
        gemini_response = gemini_chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        initial_response = gemini_response["output_text"]
        
    except Exception as e:
        initial_response = f"Error: {str(e)}" 
    
    if "The answer is not available in the context" in initial_response or "The provided context does not contain any information" in initial_response:
        if similarity_score > 0.00125:  # Adjust this threshold as needed
            refined_response = get_llama_response(user_question, no_words=500, blog_style="detailed")
        else:
            refined_response = "The question is not related to the pdf...."
    else:
        refined_response = get_llama_response(initial_response, no_words=500, blog_style="detailed")
    
    return jsonify({'generated_response': refined_response}), 200




def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        print(f"Error during embedding: {e}")

def get_conversational_chain():
    prompt_template = """
    Please provide a detailed answer based on the provided context. If the necessary information to answer the question is not present in the context, respond with 'The answer is not available in the context'
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def get_llama_response(input_text, no_words, blog_style):
    llm = CTransformers(
        model='C:/Users/bandi/OneDrive/Desktop/os/LLama/chat-with-author-backend/models/llama-2-7b-chat.ggmlv3.q8_0.bin',
        model_type='llama',
        config={'max_new_tokens': 500, 'temperature': 0.01}
    )
    template="""
          Given some information of '{input_text}', provide a concise summary suitable for a {blog_style} blog post in approximately {no_words} words. Focus on key aspects and provide accurate information.
    """
    
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", 'no_words'],
                            template=template)
    
    response = llm(prompt.format(input_text=input_text, no_words=no_words, blog_style=blog_style))
    return response

def calculate_cosine_similarity(text, user_question):
    vectorizer = TfidfVectorizer(stop_words=list(stop_words))
    tfidf_matrix = vectorizer.fit_transform([text, user_question])
    cos_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return cos_similarity


if __name__ == '__main__':
    app.run(debug=False)
