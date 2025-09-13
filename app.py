from flask import Flask, request, jsonify, session, redirect, url_for, render_template
from flask_cors import CORS
import os
import json
import numpy as np
from datetime import datetime, timedelta
import requests
import pickle
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import google.auth.exceptions
import google.oauth2.credentials
from sentence_transformers import SentenceTransformer
import faiss
import re
from typing import List, Dict, Any, Optional
import logging
import hashlib
import time
from dotenv import load_dotenv
import os
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'cf9874baf14459ac4f89db0f2dd0659544b910e22f53061d33e522111386a510')
CORS(app, supports_credentials=True)

# Configuration from environment variables
SCOPES = ['https://www.googleapis.com/auth/documents.readonly', 'https://www.googleapis.com/auth/drive.readonly']
CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')
REDIRECT_URI = os.getenv('REDIRECT_URI', 'http://localhost:5000/oauth2callback')

# Gemini API configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent'

# Alternative: Hugging Face configuration
HF_API_KEY = os.getenv('HF_API_KEY', 'your-hugging-face-api-key')
HF_MODEL = os.getenv('HF_MODEL', 'microsoft/DialoGPT-medium')

class DocumentProcessor:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunk_size = 512
        self.overlap = 50
        
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            
        return chunks
    
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Create embeddings for text chunks"""
        return self.embedding_model.encode(chunks)

class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.metadata = []
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
    def add_documents(self, chunks: List[str], embeddings: np.ndarray, doc_metadata: Dict[str, Any]):
        """Add documents to the vector store"""
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)
            
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings.astype('float32'))
        
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.metadata.append({**doc_metadata, 'chunk_id': len(self.chunks) - 1})
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks"""
        if self.index is None or self.index.ntotal == 0:
            return []
            
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        scores, indices = self.index.search(query_embedding.reshape(1, -1).astype('float32'), k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid index
                results.append({
                    'chunk': self.chunks[idx],
                    'score': float(score),
                    'metadata': self.metadata[idx]
                })
        
        return results
    
    def clear(self):
        """Clear the vector store"""
        self.index = None
        self.chunks = []
        self.metadata = []

class LLMClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def generate_response_gemini(self, context: str, query: str, found_in_docs: bool) -> str:
        """Generate response using Gemini API"""
        if found_in_docs:
            prompt = f"""Based on the following context from the user's documents, answer the question:

Context: {context}

Question: {query}

Please provide a helpful and accurate response based on the context provided."""
        else:
            prompt = f"""The user asked: "{query}"

I could not find relevant information in their selected documents. Please provide a helpful response based on your general knowledge, and start your response by mentioning that the answer was not found in their documents."""

        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1024,
            }
        }
        
        try:
            response = requests.post(
                f"{GEMINI_API_URL}?key={self.api_key}",
                headers={'Content-Type': 'application/json'},
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return "I apologize, but I'm unable to generate a response at the moment."
                
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            return "I'm sorry, but I encountered an error while generating a response."
    
    def generate_response_hf(self, context: str, query: str, found_in_docs: bool) -> str:
        """Generate response using Hugging Face API"""
        API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        if found_in_docs:
            prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        else:
            prompt = f"I could not find information about '{query}' in the user's documents. Here's what I know from general knowledge:"
            
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 512, "temperature": 0.7}}
        
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
                # Remove the prompt from the response
                answer = generated_text.replace(prompt, '').strip()
                return answer if answer else "I apologize, but I'm unable to generate a response at the moment."
            else:
                return "I apologize, but I'm unable to generate a response at the moment."
                
        except Exception as e:
            logger.error(f"Error calling Hugging Face API: {str(e)}")
            return "I'm sorry, but I encountered an error while generating a response."

# Initialize components
doc_processor = DocumentProcessor()
vector_store = VectorStore()

# Initialize LLM client - choose between Gemini and Hugging Face
# Initialize LLM client - choose between Gemini and Hugging Face
if GEMINI_API_KEY and isinstance(GEMINI_API_KEY, str) and len(GEMINI_API_KEY) > 10:
    llm_client = LLMClient(GEMINI_API_KEY)
    use_gemini = True
    logger.info("Using Gemini API for LLM")
elif HF_API_KEY and HF_API_KEY != 'your-hugging-face-api-key':
    llm_client = LLMClient(HF_API_KEY) 
    use_gemini = False
    logger.info("Using Hugging Face API for LLM")
else:
    logger.error("No valid API key found. Please set either GEMINI_API_KEY or HF_API_KEY")
    raise ValueError("No valid LLM API key configured")

def get_google_credentials():
    """Get Google API credentials from session"""
    if 'credentials' not in session:
        return None
    
    credentials_dict = session['credentials']
    credentials = google.oauth2.credentials.Credentials(
        token=credentials_dict['token'],
        refresh_token=credentials_dict['refresh_token'],
        token_uri=credentials_dict['token_uri'],
        client_id=credentials_dict['client_id'],
        client_secret=credentials_dict['client_secret'],
        scopes=credentials_dict['scopes']
    )
    
    # Refresh token if expired
    if credentials.expired:
        try:
            credentials.refresh(Request())
            session['credentials'] = {
                'token': credentials.token,
                'refresh_token': credentials.refresh_token,
                'token_uri': credentials.token_uri,
                'client_id': credentials.client_id,
                'client_secret': credentials.client_secret,
                'scopes': credentials.scopes
            }
        except google.auth.exceptions.RefreshError:
            return None
    
    return credentials

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/auth/google')
def google_auth():
    """Initiate Google OAuth flow"""
    flow = Flow.from_client_config({
        "web": {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [REDIRECT_URI]
        }
    }, scopes=SCOPES)
    
    flow.redirect_uri = REDIRECT_URI
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true'
    )
    
    session['state'] = state
    return redirect(authorization_url)

@app.route('/oauth2callback')
def oauth2callback():
    """Handle OAuth callback"""
    state = session.get('state')
    
    flow = Flow.from_client_config({
        "web": {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [REDIRECT_URI]
        }
    }, scopes=SCOPES, state=state)
    
    flow.redirect_uri = REDIRECT_URI
    authorization_response = request.url
    flow.fetch_token(authorization_response=authorization_response)
    
    credentials = flow.credentials
    session['credentials'] = {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }
    
    return redirect('/')

@app.route('/api/auth/status')
def auth_status():
    """Check authentication status"""
    credentials = get_google_credentials()
    return jsonify({'authenticated': credentials is not None})

@app.route('/api/documents')
def list_documents():
    """List Google Docs"""
    credentials = get_google_credentials()
    if not credentials:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        drive_service = build('drive', 'v3', credentials=credentials)
        
        # Query for Google Docs
        results = drive_service.files().list(
            q="mimeType='application/vnd.google-apps.document'",
            pageSize=50,
            fields="files(id,name,modifiedTime,webViewLink)"
        ).execute()
        
        documents = results.get('files', [])
        
        # Format the response
        formatted_docs = []
        for doc in documents:
            formatted_docs.append({
                'id': doc['id'],
                'name': doc['name'],
                'modified': doc.get('modifiedTime', ''),
                'url': doc.get('webViewLink', '')
            })
        
        return jsonify({'documents': formatted_docs})
        
    except Exception as e:
        logger.error(f"Error fetching documents: {str(e)}")
        return jsonify({'error': 'Failed to fetch documents'}), 500

@app.route('/api/documents/add', methods=['POST'])
def add_documents():
    """Add selected documents to knowledge base"""
    credentials = get_google_credentials()
    if not credentials:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    document_ids = data.get('document_ids', [])
    
    if not document_ids:
        return jsonify({'error': 'No documents selected'}), 400
    
    try:
        # Clear existing vector store
        vector_store.clear()
        
        docs_service = build('docs', 'v1', credentials=credentials)
        drive_service = build('drive', 'v3', credentials=credentials)
        
        processed_docs = []
        
        for doc_id in document_ids:
            try:
                # Get document content
                document = docs_service.documents().get(documentId=doc_id).execute()
                
                # Get document metadata
                file_metadata = drive_service.files().get(fileId=doc_id).execute()
                
                # Extract text content
                content = extract_text_from_doc(document)
                
                if content.strip():
                    # Process document
                    chunks = doc_processor.chunk_text(content)
                    embeddings = doc_processor.create_embeddings(chunks)
                    
                    # Add to vector store
                    metadata = {
                        'document_id': doc_id,
                        'document_name': file_metadata['name'],
                        'source': 'google_docs'
                    }
                    
                    vector_store.add_documents(chunks, embeddings, metadata)
                    
                    processed_docs.append({
                        'id': doc_id,
                        'name': file_metadata['name'],
                        'chunks': len(chunks)
                    })
                
            except Exception as e:
                logger.error(f"Error processing document {doc_id}: {str(e)}")
                continue
        
        return jsonify({
            'message': f'Successfully added {len(processed_docs)} documents to knowledge base',
            'documents': processed_docs
        })
        
    except Exception as e:
        logger.error(f"Error adding documents: {str(e)}")
        return jsonify({'error': 'Failed to add documents'}), 500

def extract_text_from_doc(document):
    """Extract text content from Google Docs document"""
    content = document.get('body', {}).get('content', [])
    text = []
    
    for element in content:
        if 'paragraph' in element:
            paragraph = element['paragraph']
            for elem in paragraph.get('elements', []):
                if 'textRun' in elem:
                    text.append(elem['textRun']['content'])
        elif 'table' in element:
            # Handle tables
            table = element['table']
            for row in table.get('tableRows', []):
                for cell in row.get('tableCells', []):
                    for cell_element in cell.get('content', []):
                        if 'paragraph' in cell_element:
                            paragraph = cell_element['paragraph']
                            for elem in paragraph.get('elements', []):
                                if 'textRun' in elem:
                                    text.append(elem['textRun']['content'])
    
    return ''.join(text)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat queries"""
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        # Create query embedding
        query_embedding = doc_processor.embedding_model.encode([query])
        
        # Search for relevant chunks
        search_results = vector_store.search(query_embedding[0], k=5)
        
        if search_results and search_results[0]['score'] > 0.3:  # Threshold for relevance
            # Found relevant content
            context = '\n'.join([result['chunk'] for result in search_results[:3]])
            source_docs = list(set([result['metadata']['document_name'] for result in search_results[:3]]))
            
            if use_gemini:
                response = llm_client.generate_response_gemini(context, query, found_in_docs=True)
            else:
                response = llm_client.generate_response_hf(context, query, found_in_docs=True)
            
            return jsonify({
                'response': response,
                'sources': source_docs,
                'found_in_docs': True,
                'search_results_count': len(search_results)
            })
        else:
            # No relevant content found
            if use_gemini:
                response = llm_client.generate_response_gemini('', query, found_in_docs=False)
            else:
                response = llm_client.generate_response_hf('', query, found_in_docs=False)
            
            return jsonify({
                'response': response,
                'sources': [],
                'found_in_docs': False,
                'search_results_count': 0
            })
            
    except Exception as e:
        logger.error(f"Error processing chat query: {str(e)}")
        return jsonify({'error': 'Failed to process query'}), 500

@app.route('/api/knowledge-base/status')
def knowledge_base_status():
    """Get knowledge base status"""
    total_chunks = len(vector_store.chunks)
    documents = {}
    
    for metadata in vector_store.metadata:
        doc_name = metadata.get('document_name', 'Unknown')
        if doc_name not in documents:
            documents[doc_name] = 0
        documents[doc_name] += 1
    
    return jsonify({
        'total_chunks': total_chunks,
        'total_documents': len(documents),
        'documents': documents
    })

@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    vector_store.clear()
    return redirect('/')

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Validate configuration
    if not CLIENT_ID or CLIENT_ID == 'your-google-client-id':
        logger.error("Google OAuth CLIENT_ID not configured")
        print("❌ Error: Google OAuth CLIENT_ID not configured")
        print("Please set GOOGLE_CLIENT_ID environment variable or update CLIENT_ID in app.py")
        exit(1)
        
    if not CLIENT_SECRET or CLIENT_SECRET == 'your-google-client-secret':
        logger.error("Google OAuth CLIENT_SECRET not configured")
        print("❌ Error: Google OAuth CLIENT_SECRET not configured")
        print("Please set GOOGLE_CLIENT_SECRET environment variable or update CLIENT_SECRET in app.py")
        exit(1)
    
    print("🚀 Starting RAG Chatbot Server...")
    print("📋 Configuration:")
    print(f"   • Client ID: {CLIENT_ID[:20]}...")
    print(f"   • Redirect URI: {REDIRECT_URI}")
    print(f"   • Using {'Gemini' if use_gemini else 'Hugging Face'} API")
    print(f"   • Server: http://localhost:{os.getenv('PORT', 5000)}")
    print("\n🔧 Setup checklist:")
    print("   ✅ Install dependencies: pip install -r requirements.txt")
    print("   ✅ Google OAuth credentials configured")
    print("   ✅ API key configured")
    print("   ✅ Templates directory created")
    print("   📝 Make sure Google OAuth redirect URI is set to:", REDIRECT_URI)
    
    app.run(
        debug=os.getenv('FLASK_DEBUG', 'True').lower() == 'true',
        host=os.getenv('HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', 5000))
    )