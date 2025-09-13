# RAG-Powered Google Docs Chatbot

A complete RAG (Retrieval-Augmented Generation) chatbot that integrates with Google Docs using OAuth 2.0 authentication. Users can select documents from their Google Drive, add them to a knowledge base, and chat with their documents using AI.

## Features

- 🔐 **Google OAuth 2.0 Authentication**: Secure login with Google accounts
- 📄 **Google Docs Integration**: Fetch and display user's Google Docs
- 🧠 **Knowledge Base Management**: Add selected documents to a searchable knowledge base
- 🤖 **RAG Pipeline**: Retrieval-Augmented Generation using vector embeddings
- 💬 **Interactive Chat Interface**: Clean, responsive web interface
- 🔍 **Source Attribution**: Shows which documents were used for answers
- ⚡ **Fallback Mechanism**: Uses general knowledge when documents don't contain the answer

## Architecture

- **Backend**: Python Flask with Google APIs
- **Vector Database**: FAISS for similarity search
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **LLM**: Google Gemini API (with Hugging Face alternative)
- **Frontend**: Vanilla JavaScript with modern responsive design

## Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- Google Cloud Console account
- Gemini API key (or Hugging Face API key)

### 2. Google Cloud Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the following APIs:
   - Google Drive API
   - Google Docs API
4. Create OAuth 2.0 credentials:
   - Go to "Credentials" → "Create Credentials" → "OAuth 2.0 Client ID"
   - Choose "Web application"
   - Add authorized redirect URI: `http://localhost:5000/oauth2callback`
   - Download the client configuration

### 3. Get Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key for use in the application

### 4. Installation

```bash
# Clone or create the project directory
mkdir rag-chatbot
cd rag-chatbot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create templates directory
mkdir templates
```

### 5. Configuration

Edit `app.py` and update the following variables:

```python
CLIENT_ID = 'your-google-client-id'           # From Google Cloud Console
CLIENT_SECRET = 'your-google-client-secret'   # From Google Cloud Console
GEMINI_API_KEY = 'your-gemini-api-key'        # From Google AI Studio
app.secret_key = 'your-secret-key-here'       # Change this to a random string
```

### 6. File Structure

Ensure your project has this structure:

```
rag-chatbot/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── templates/
    └── index.html        # Frontend interface
```

### 7. Running the Application

```bash
# Make sure virtual environment is activated
python app.py
```

The application will start on `http://localhost:5000`

## Usage Guide

### 1. Authentication
- Click "Sign in with Google" to authenticate with your Google account
- Grant permissions to access your Google Docs and Drive

### 2. Document Management
- View all your Google Docs in the sidebar
- Select documents you want to add to the knowledge base
- Click "Add Selected Documents" to process them

### 3. Chatting
- Once documents are added, the chat input will be enabled
- Ask questions about your documents in natural language
- The system will:
  - Search your documents for relevant content
  - Generate answers using the found information
  - Fall back to general knowledge if no relevant content is found
  - Show source documents when applicable

### 4. Knowledge Base Status
- Monitor how many documents and chunks are in your knowledge base
- See which documents have been processed

## API Endpoints

- `GET /` - Main application page
- `GET /auth/google` - Initiate Google OAuth flow
- `GET /oauth2callback` - OAuth callback handler
- `GET /api/auth/status` - Check authentication status
- `GET /api/documents` - List user's Google Docs
- `POST /api/documents/add` - Add documents to knowledge base
- `POST /api/chat` - Send chat message and get response
- `GET /api/knowledge-base/status` - Get knowledge base statistics
- `GET /logout` - Logout and clear session

## Technical Details

### RAG Pipeline

1. **Document Processing**:
   - Extracts text from Google Docs (handles paragraphs and tables)
   - Splits text into overlapping chunks (512 tokens with 50 token overlap)
   - Creates embeddings using Sentence Transformers

2. **Vector Storage**:
   - Uses FAISS for efficient similarity search
   - Stores chunks with metadata (document name, ID, etc.)
   - Implements cosine similarity for relevance scoring

3. **Retrieval & Generation**:
   - Converts user queries to embeddings
   - Searches for top-k similar chunks (threshold: 0.3)
   - Provides context to Gemini API for answer generation
   - Falls back to general knowledge if no relevant content found

### Security Features

- OAuth 2.0 for secure authentication
- Session management with Flask sessions
- Token refresh handling
- CORS protection
- Input validation and error handling

## Alternative LLM Configuration

To use Hugging Face instead of Gemini:

1. Uncomment the Hugging Face sections in `app.py`
2. Replace `GEMINI_API_KEY` with `HF_API_KEY`
3. Update the `chat()` function to use `generate_response_hf()`

## Troubleshooting

### Common Issues

1. **OAuth Errors**: 
   - Verify redirect URI matches exactly
   - Check client ID and secret
   - Ensure APIs are enabled

2. **Import Errors**:
   - Make sure virtual environment is activated
   - Install all requirements: `pip install -r requirements.txt`

3. **API Errors**:
   - Verify Gemini API key is correct and active
   - Check API quotas and limits

4. **Document Processing Issues**:
   - Ensure documents are Google Docs (not other formats)
   - Check document permissions

### Logs and Debugging

The application logs important events and errors. Check the console output for:
- Authentication status
- Document processing progress
- API call results
- Error messages

## Deployment Notes

For production deployment:

1. Use environment variables for sensitive data
2. Set up proper SSL certificates
3. Configure production WSGI server (Gunicorn included)
4. Update redirect URIs for production domain
5. Implement proper session storage (Redis/Database)
6. Add rate limiting and monitoring

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.