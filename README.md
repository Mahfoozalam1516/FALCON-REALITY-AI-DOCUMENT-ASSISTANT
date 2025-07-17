# 🤖 RAG Document Assistant

A powerful AI-powered document analysis chatbot built with FastAPI, LangChain, and React. Upload PDF documents and have intelligent conversations about their content using Retrieval-Augmented Generation (RAG).

## ✨ Features

- 📄 **Multiple PDF Upload**: Upload and analyze multiple documents simultaneously
- 🧠 **Smart RAG System**: Context-aware responses using advanced retrieval techniques
- 💬 **ChatGPT-like Interface**: Clean, formatted responses with proper markdown rendering
- 🎨 **Beautiful Dark Theme**: Modern UI with smooth animations and responsive design
- 🔐 **Secure Configuration**: Environment-based setup with no hardcoded credentials
- 🚀 **High Performance**: Optimized embedding and retrieval pipeline
- 📊 **Source Attribution**: Shows which documents were referenced in responses
- 🌐 **Cross-Platform**: Works on Windows, macOS, and Linux

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│  React Frontend │◄──►│  FastAPI Backend│◄──►│Pinecone VectorDB│
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │                 │
                       │  Groq LLM API   │
                       │                 │
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+ installed
- Node.js 18+ installed
- Git installed
- Groq API account ([Get free API key](https://console.groq.com/))
- Pinecone account ([Get free API key](https://app.pinecone.io/))

### 1. Clone the Repository

```bash
git clone <repository-url>
cd rag-chatbot
```

### 2. Backend Setup

#### Create Virtual Environment
```bash
cd backend
python -m venv rag_env

# Activate virtual environment
# Windows:
rag_env\Scripts\activate
# macOS/Linux:
source rag_env/bin/activate
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Configure Environment Variables
Create a `.env` file in the `backend` directory:

```bash
# Required API Keys
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# Pinecone Configuration
PINECONE_INDEX_NAME=pdf-embeddings
PINECONE_DIMENSION=768
PINECONE_METRIC=cosine
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Model Configuration
EMBEDDING_MODEL=all-mpnet-base-v2
LLM_MODEL=llama-3.1-8b-instant
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=1000

# Document Processing
CHUNK_SIZE=500
CHUNK_OVERLAP=100
MAX_FILE_SIZE_MB=10
MAX_FILES_PER_REQUEST=5
SIMILARITY_THRESHOLD=0.5
TOP_K_RESULTS=8

# Security
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]
ALLOWED_FILE_TYPES=[".pdf"]

# Development
DEBUG=true
LOG_LEVEL=INFO
```

#### Run the Backend
```bash
python main.py
```

The backend will start on `http://localhost:8000`

### 3. Frontend Setup

#### Navigate to Frontend Directory
```bash
cd ../frontend
```

#### Install Dependencies
```bash
npm install
```

#### Configure Environment Variables
Create a `.env` file in the `frontend` directory:

```bash
# API Configuration
VITE_API_BASE_URL=http://localhost:8000

# App Configuration
VITE_APP_TITLE=AI Document Assistant
VITE_APP_SUBTITLE=Powered by RAG Technology

# Upload Configuration
VITE_MAX_FILE_SIZE_MB=10
VITE_MAX_FILES_PER_REQUEST=5
VITE_ALLOWED_FILE_TYPES=.pdf

# UI Configuration
VITE_TYPING_DELAY_MS=800
VITE_ANIMATION_DURATION_MS=600
VITE_ENABLE_ANIMATIONS=true

# Development
VITE_DEBUG=true
VITE_LOG_LEVEL=info
```

#### Run the Frontend
```bash
npm run dev
```

The frontend will start on `http://localhost:3000`

## 🔑 Getting API Keys

### Groq API Key
1. Visit [Groq Console](https://console.groq.com/)
2. Sign up for a free account
3. Navigate to "API Keys" section
4. Click "Create API Key"
5. Copy the key and paste it in your `.env` file

### Pinecone API Key
1. Visit [Pinecone Console](https://app.pinecone.io/)
2. Sign up for a free account
3. Go to "API Keys" section
4. Copy your API key
5. Paste it in your `.env` file

## 📱 How to Use

### 1. Start the Application
- Ensure both backend and frontend are running
- Open `http://localhost:3000` in your browser

### 2. Upload Documents
- Click "Choose Files" or drag & drop PDF files
- Upload up to 5 files at once (max 10MB each)
- Wait for processing to complete

### 3. Chat with Your Documents
- Type questions about your uploaded documents
- Get intelligent, context-aware responses
- See source attribution for each response

### 4. Example Questions
- "What are the main topics in these documents?"
- "Summarize the key findings"
- "Compare the different documents"
- "What does document X say about Y?"

## 🛠️ Project Structure

```
rag-chatbot/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── requirements.txt        # Python dependencies
│   └── .env                   # Backend configuration
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Main React component
│   │   ├── App.css            # Styling
│   │   ├── main.jsx           # Entry point
│   │   └── index.css          # Global styles
│   ├── package.json           # Node dependencies
│   ├── vite.config.js         # Vite configuration
│   ├── index.html             # HTML template
│   └── .env                   # Frontend configuration
├── .gitignore                 # Git ignore file
└── README.md                  # This file
```

## 🔧 Configuration Options

### Backend Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key (required) | - |
| `PINECONE_API_KEY` | Pinecone API key (required) | - |
| `CHUNK_SIZE` | Text chunk size for processing | 500 |
| `CHUNK_OVERLAP` | Overlap between chunks | 100 |
| `MAX_FILE_SIZE_MB` | Maximum file size in MB | 10 |
| `MAX_FILES_PER_REQUEST` | Maximum files per upload | 5 |
| `SIMILARITY_THRESHOLD` | Minimum similarity for retrieval | 0.5 |
| `TOP_K_RESULTS` | Number of chunks to retrieve | 8 |
| `DEBUG` | Enable debug logging | true |

### Frontend Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_BASE_URL` | Backend API URL | http://localhost:8000 |
| `VITE_APP_TITLE` | Application title | AI Document Assistant |
| `VITE_MAX_FILE_SIZE_MB` | Max file size (must match backend) | 10 |
| `VITE_ENABLE_ANIMATIONS` | Enable UI animations | true |
| `VITE_DEBUG` | Enable debug logging | true |

## 🐛 Troubleshooting

### Common Issues

#### 1. Backend Issues

**Error: "GROQ_API_KEY environment variable is required"**
- Ensure `.env` file exists in `backend` directory
- Check that `GROQ_API_KEY` is set in `.env`
- Restart the backend server

**Error: "Cannot connect to Pinecone"**
- Verify `PINECONE_API_KEY` is correct
- Check internet connection
- Ensure Pinecone account is active

**Error: "ModuleNotFoundError"**
- Make sure virtual environment is activated
- Run `pip install -r requirements.txt`

#### 2. Frontend Issues

**Error: "Cannot reach backend API"**
- Check `VITE_API_BASE_URL` in frontend `.env`
- Ensure backend is running on the correct port
- Check browser console for CORS errors

**Error: "npm install fails"**
- Update Node.js to version 18+
- Clear npm cache: `npm cache clean --force`
- Delete `node_modules` and reinstall

#### 3. Upload Issues

**Error: "File too large"**
- Check file size is under 10MB
- Ensure `MAX_FILE_SIZE_MB` is set correctly in both environments

**Error: "Invalid file type"**
- Only PDF files are supported
- Check file extension is `.pdf`

### Getting Help

1. Check the console logs for detailed error messages
2. Verify all environment variables are set correctly
3. Ensure API keys are valid and active
4. Check that all dependencies are installed

## 🚀 Production Deployment

### Backend Deployment

#### Using Railway
```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy
railway login
railway init
railway up
```

#### Using Docker
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

### Frontend Deployment

#### Using Vercel
```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
vercel --prod
```

#### Using Netlify
```bash
# Build the project
npm run build

# Deploy dist folder to Netlify
```

## 📊 Performance Tips

### For Better Speed
- Use `all-MiniLM-L6-v2` embedding model (smaller, faster)
- Reduce `CHUNK_SIZE` to 300
- Set `TOP_K_RESULTS` to 5

### For Better Accuracy
- Use `all-mpnet-base-v2` embedding model (better quality)
- Increase `TOP_K_RESULTS` to 10
- Lower `SIMILARITY_THRESHOLD` to 0.3


## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [LangChain](https://langchain.com/) - LLM application framework
- [Groq](https://groq.com/) - Ultra-fast LLM inference
- [Pinecone](https://pinecone.io/) - Vector database
- [React](https://react.dev/) - Frontend framework
- [Sentence Transformers](https://sentence-transformers.net/) - Embedding models
