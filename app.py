from flask import Flask, request, jsonify
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # HuggingFace Embeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from langchain_community.llms import CTransformers
from flask_cors import CORS
from pymongo import MongoClient

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- Configuration and Initialization ---
load_dotenv()

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
custom_stopwords = ["what", "is", "how", "who", "explain", "about", "?", "please", "hey", "whatsup", "can u explain"]
stop_words.update(custom_stopwords)

# Database connection
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    raise ValueError("MongoDB URI not found. Please set the MONGO_URI environment variable.")
try:
    client = MongoClient(mongo_uri)
    db = client['pdf_database']
    collection = db['text_chunks']
    client.server_info()
    print("Successfully connected to MongoDB.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit()

global_raw_text = ""
FAISS_INDEX_PATH = "C:/Users/bandi/OneDrive/Desktop/LLama/chat-with-author-backend/faiss_index"

# Initialize HuggingFace Embeddings
print("Initializing HuggingFace embedding model...")
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("HuggingFace embedding model loaded.")
except Exception as e:
    print(f"Error initializing HuggingFace embedding model: {e}")
    embeddings = None
    exit()

# Initialize local LLM (Llama 2 or your Llama 3 path)
print("Initializing local LLM...")
try:
    llm = CTransformers(
        model='C:/Users/bandi/OneDrive/Desktop/LLama/chat-with-author-backend/models/llama-2-7b-chat.ggmlv3.q8_0.bin',  # Change if you have llama3 weights
        model_type='llama',
        config={'max_new_tokens': 512, 'temperature': 0.1, 'context_length': 2500}
    )
    print("LLM loaded successfully.")
except Exception as e:
    print(f"Error loading local LLM: {e}")
    llm = None

# --- Helper functions ---

def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            print(f"Error reading PDF file {pdf.filename}: {e}")
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000, length_function=len)
    return splitter.split_text(text)

def create_and_save_vector_store(text_chunks):
    if not text_chunks:
        print("No text chunks to process for vector store.")
        return
    if not embeddings:
        print("Embedding model not available.")
        return
    try:
        print("Creating vector store from text chunks...")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        print(f"Vector store created and saved to '{FAISS_INDEX_PATH}'.")
    except Exception as e:
        print(f"Error creating vector store: {e}")

def store_chunks_in_mongodb(chunks):
    if not chunks:
        print("No chunks to store.")
        return
    try:
        collection.delete_many({})
        collection.insert_many([{"text": c} for c in chunks])
        print(f"Inserted {len(chunks)} chunks into MongoDB.")
    except Exception as e:
        print(f"Error storing in MongoDB: {e}")

def get_llama_response(question, context=""):
    if not llm:
        return "LLM is not available."
    template = """
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    prompt = PromptTemplate(template=template, input_variables=['context', 'question'])
    formatted = prompt.format(context=context, question=question)
    return llm(formatted)

def calculate_cosine_similarity(text, question):
    if not text or not question:
        return 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words=list(stop_words))
        matrix = vectorizer.fit_transform([text, question])
        return cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    except ValueError:
        return 0.0

# --- Flask routes ---

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    global global_raw_text
    if 'pdf_files' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    pdf_files = request.files.getlist('pdf_files')
    if not pdf_files or pdf_files[0].filename == '':
        return jsonify({"error": "No PDF files selected"}), 400

    print("Processing uploaded PDFs...")
    raw_text = get_pdf_text(pdf_files)
    global_raw_text = raw_text

    chunks = get_text_chunks(raw_text)
    store_chunks_in_mongodb(chunks)
    create_and_save_vector_store(chunks)

    return jsonify({"message": f"Successfully processed {len(pdf_files)} PDF(s). Index is ready."}), 200

@app.route('/process-query', methods=['POST'])
def process_query():
    data = request.json
    if 'user_question' not in data:
        return jsonify({"error": "Missing user_question field"}), 400

    user_question = data['user_question']
    print(f"Query: {user_question}")

    if not os.path.exists(FAISS_INDEX_PATH):
        return jsonify({"error": "Vector store not found. Please upload a PDF first."}), 400
    if not embeddings:
        return jsonify({"error": "Embedding model not available."}), 500

    try:
        print("Loading FAISS index...")
        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

        print("Searching close matches...")
        similar_docs = db.similarity_search(user_question, k=3)
        context = " ".join(doc.page_content for doc in similar_docs)

        if not context.strip():
            print("No relevant context found.")
            score = calculate_cosine_similarity(global_raw_text, user_question)
            if score > 0.1:
                return jsonify({"generated_response": "Found some related info, but couldn't pinpoint an exact answer. Please rephrase your question."}), 200
            else:
                return jsonify({"generated_response": "No answer found in the provided PDF."}), 200

        print("Generating response from local LLM...")
        answer = get_llama_response(user_question, context=context)
        return jsonify({"generated_response": answer}), 200

    except Exception as e:
        print(f"Query processing error: {e}")
        return jsonify({"error": f"Internal error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # Before running, install sentence-transformers package: pip install sentence-transformers
    app.run(host='0.0.0.0', port=5000, debug=False)
