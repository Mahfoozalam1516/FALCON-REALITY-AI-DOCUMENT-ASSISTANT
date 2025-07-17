from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fitz  # PyMuPDF
import io
import os
from typing import List, Optional, Union
import uvicorn
from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import uuid
import logging
from datetime import datetime
import re
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

# Environment Variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone Configuration
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pdf-embeddings")
PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "768"))
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Server Configuration
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))

# Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))

# Document Processing Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILES_PER_REQUEST = int(os.getenv("MAX_FILES_PER_REQUEST", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "8"))

# Security Configuration
CORS_ORIGINS = json.loads(os.getenv("CORS_ORIGINS", '["http://localhost:3000", "http://127.0.0.1:3000"]'))
ALLOWED_FILE_TYPES = json.loads(os.getenv("ALLOWED_FILE_TYPES", '[".pdf"]'))

# Development Configuration
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Validate required environment variables
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is required")

logger.info(f"Starting RAG Chatbot API with configuration:")
logger.info(f"- Embedding Model: {EMBEDDING_MODEL}")
logger.info(f"- LLM Model: {LLM_MODEL}")
logger.info(f"- Chunk Size: {CHUNK_SIZE}")
logger.info(f"- Max Files: {MAX_FILES_PER_REQUEST}")
logger.info(f"- Debug Mode: {DEBUG}")

# Global variables
embeddings_model = None
pc = None
index = None
llm = None
vector_store = None
chatbot = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global embeddings_model, pc, index, llm, vector_store, chatbot
    
    try:
        # Initialize embeddings model
        logger.info(f"Loading embeddings model: {EMBEDDING_MODEL}...")
        embeddings_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Initialize Pinecone
        logger.info("Initializing Pinecone...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Create or get index
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            try:
                pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=PINECONE_DIMENSION,
                    metric=PINECONE_METRIC,
                    spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
                )
                logger.info(f"Created Pinecone index: {PINECONE_INDEX_NAME}")
            except Exception as e:
                logger.error(f"Error creating Pinecone index: {e}")
                raise Exception(f"Error creating Pinecone index: {e}")
        
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Initialize Groq LLM
        logger.info(f"Initializing Groq LLM: {LLM_MODEL}...")
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
        
        # Initialize vector store
        vector_store = VectorStore(embeddings_model, pc, index)
        
        # Initialize chatbot
        chatbot = RAGChatbot(llm, vector_store)
        
        logger.info("All components initialized successfully!")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise e
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(title="RAG Chatbot API", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str
    sources_found: int

class UploadResponse(BaseModel):
    message: str
    total_chunks_processed: int
    files_processed: List[dict]
    total_files: int

class PDFProcessor:
    def __init__(self):
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP
        self.max_file_size = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes
        self.allowed_types = ALLOWED_FILE_TYPES
    
    def validate_file(self, file: UploadFile) -> bool:
        """Validate file type and size"""
        # Check file extension
        if not any(file.filename.endswith(ext) for ext in self.allowed_types):
            raise HTTPException(
                status_code=400, 
                detail=f"File type not supported. Allowed types: {', '.join(self.allowed_types)}"
            )
        
        return True
    
    def extract_text_from_pdf(self, pdf_bytes: bytes, filename: str) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            page_texts = []
            
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text.strip():  # Only add non-empty pages
                    page_texts.append(f"[Page {page_num + 1} of {filename}]\n{page_text}")
            
            text = "\n\n".join(page_texts)
            doc.close()
            
            if not text.strip():
                raise Exception("No text content found in PDF")
                
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {filename}: {e}")
            raise HTTPException(status_code=400, detail=f"Error processing PDF {filename}: {str(e)}")
    
    def chunk_text(self, text: str, filename: str) -> List[str]:
        """Chunk text into smaller pieces with filename context"""
        # Clean the text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split into sentences for better chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    # Add filename context to each chunk
                    chunk_with_context = f"[Document: {filename}]\n{current_chunk.strip()}"
                    chunks.append(chunk_with_context)
                
                # Start new chunk with overlap if possible
                current_chunk = sentence + " "
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunk_with_context = f"[Document: {filename}]\n{current_chunk.strip()}"
            chunks.append(chunk_with_context)
        
        # Ensure we have at least one chunk
        if not chunks and text.strip():
            chunk_with_context = f"[Document: {filename}]\n{text[:self.chunk_size]}"
            chunks.append(chunk_with_context)
        
        logger.info(f"Created {len(chunks)} chunks for {filename}")
        return chunks

class VectorStore:
    def __init__(self, embeddings_model, pinecone_client, index):
        self.embeddings_model = embeddings_model
        self.pc = pinecone_client
        self.index = index
    
    def add_chunks(self, chunks: List[str], metadata: dict = None) -> int:
        """Add chunks to vector store"""
        try:
            vectors = []
            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.embeddings_model.encode(chunk).tolist()
                
                # Create metadata
                chunk_metadata = {
                    "text": chunk,
                    "chunk_index": i,
                    "timestamp": datetime.now().isoformat(),
                    **(metadata or {})
                }
                
                # Create vector
                vector = {
                    "id": f"{metadata.get('file_name', 'unknown')}_{i}_{uuid.uuid4()}",
                    "values": embedding,
                    "metadata": chunk_metadata
                }
                vectors.append(vector)
            
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                logger.info(f"Upserted batch {i//batch_size + 1} with {len(batch)} vectors")
            
            logger.info(f"Successfully added {len(chunks)} chunks to vector store")
            return len(chunks)
        
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            raise HTTPException(status_code=500, detail=f"Error storing chunks: {str(e)}")
    
    def search_similar_chunks(self, query: str, top_k: int = None) -> List[dict]:
        """Search for similar chunks and return with metadata"""
        if top_k is None:
            top_k = TOP_K_RESULTS
            
        try:
            # Generate query embedding
            query_embedding = self.embeddings_model.encode(query).tolist()
            
            # Search in vector store
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Extract relevant chunks with scores
            relevant_chunks = []
            for match in results['matches']:
                if match['score'] > SIMILARITY_THRESHOLD:
                    relevant_chunks.append({
                        'text': match['metadata']['text'],
                        'score': match['score'],
                        'file_name': match['metadata'].get('file_name', 'Unknown'),
                        'chunk_index': match['metadata'].get('chunk_index', 0)
                    })
            
            # Sort by score (descending)
            relevant_chunks.sort(key=lambda x: x['score'], reverse=True)
            
            logger.info(f"Found {len(relevant_chunks)} relevant chunks for query: {query[:50]}...")
            return relevant_chunks
        
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []

class RAGChatbot:
    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        self.conversations = {}
        
        # Create a custom prompt template for RAG
        self.rag_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an AI assistant that helps users understand and analyze documents. You have access to content from uploaded PDF documents. You have context awareness and can answer questions based on the content provided.

Based on the following document content, please answer the user's question in a clear, well-structured format.

Document Content:
{context}

User Question: {question}

Instructions:
1. Provide clear, comprehensive answers based on the document content
2. Structure your response with proper formatting:
   - Use **bold** for important points
   - Use bullet points (-) for lists
   - Use numbered lists (1.) for steps or sequences
   - Use headers (## or ###) for major sections when appropriate
3. If you cannot find relevant information, clearly state this
4. Always cite which documents you're referencing when possible
5. Be specific and detailed when the information is available
6. Format your response for clarity and readability

Answer:"""
        )
    
    def get_conversation_memory(self, session_id: str):
        """Get or create conversation memory for session"""
        if session_id not in self.conversations:
            self.conversations[session_id] = ConversationBufferMemory(
                memory_key="history",
                return_messages=True
            )
        return self.conversations[session_id]
    
    def generate_response(self, query: str, session_id: str = None) -> dict:
        """Generate response using RAG"""
        try:
            # Search for relevant chunks
            relevant_chunks = self.vector_store.search_similar_chunks(query, TOP_K_RESULTS)
            
            if not relevant_chunks:
                return {
                    "response": "I don't have any relevant information in the uploaded documents to answer your question. Please make sure you have uploaded PDF documents and try asking questions related to their content.",
                    "sources_found": 0
                }
            
            # Create context from relevant chunks
            context_parts = []
            unique_files = set()
            
            for chunk in relevant_chunks:
                context_parts.append(f"From {chunk['file_name']} (relevance: {chunk['score']:.2f}):\n{chunk['text']}\n")
                unique_files.add(chunk['file_name'])
            
            context = "\n---\n".join(context_parts)
            
            # Generate prompt using the template
            prompt = self.rag_prompt.format(context=context, question=query)
            
            # Generate response using Groq LLM
            try:
                response = self.llm.invoke(prompt)
                
                # Extract content from the response
                if hasattr(response, 'content'):
                    final_response = response.content
                else:
                    final_response = str(response)
                
                # Add source information
                if len(unique_files) > 0:
                    source_info = f"\n\n📚 Sources: {', '.join(unique_files)}"
                    final_response += source_info
                
                return {
                    "response": final_response,
                    "sources_found": len(relevant_chunks)
                }
                
            except Exception as e:
                logger.error(f"Error with Groq LLM: {e}")
                return {
                    "response": "I'm sorry, I encountered an error while processing your request. Please try again.",
                    "sources_found": 0
                }
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": "I'm sorry, I encountered an error while processing your request. Please try again.",
                "sources_found": 0
            }

# Initialize components - moved to lifespan function above

# Initialize PDF processor
pdf_processor = PDFProcessor()

@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running!", "status": "healthy"}

@app.post("/upload", response_model=UploadResponse)
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """Upload and index multiple PDF files"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        if len(files) > MAX_FILES_PER_REQUEST:
            raise HTTPException(
                status_code=400, 
                detail=f"Too many files. Maximum {MAX_FILES_PER_REQUEST} files allowed per request."
            )
        
        # Validate all files
        for file in files:
            pdf_processor.validate_file(file)
        
        total_chunks = 0
        processed_files = []
        
        for file in files:
            try:
                # Read file content
                pdf_bytes = await file.read()
                
                # Check file size
                if len(pdf_bytes) > pdf_processor.max_file_size:
                    processed_files.append({
                        "filename": file.filename,
                        "status": "failed",
                        "reason": f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB",
                        "chunks": 0
                    })
                    continue
                
                # Extract text from PDF
                text = pdf_processor.extract_text_from_pdf(pdf_bytes, file.filename)
                
                if not text.strip():
                    processed_files.append({
                        "filename": file.filename,
                        "status": "failed",
                        "reason": "No text found in PDF",
                        "chunks": 0
                    })
                    continue
                
                # Chunk the text
                chunks = pdf_processor.chunk_text(text, file.filename)
                
                # Add chunks to vector store
                metadata = {
                    "file_name": file.filename,
                    "file_size": len(pdf_bytes),
                    "upload_time": datetime.now().isoformat()
                }
                
                chunks_processed = vector_store.add_chunks(chunks, metadata)
                total_chunks += chunks_processed
                
                processed_files.append({
                    "filename": file.filename,
                    "status": "success",
                    "chunks": chunks_processed,
                    "size_bytes": len(pdf_bytes)
                })
                
                logger.info(f"Successfully processed {file.filename}: {chunks_processed} chunks")
                
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}")
                processed_files.append({
                    "filename": file.filename,
                    "status": "failed",
                    "reason": str(e),
                    "chunks": 0
                })
        
        successful_files = [f for f in processed_files if f["status"] == "success"]
        failed_files = [f for f in processed_files if f["status"] == "failed"]
        
        message = f"Successfully processed {len(successful_files)} out of {len(files)} files."
        if failed_files:
            message += f" {len(failed_files)} files failed to process."
        
        return UploadResponse(
            message=message,
            total_chunks_processed=total_chunks,
            files_processed=processed_files,
            total_files=len(files)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in upload endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the RAG chatbot"""
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Generate response
        result = chatbot.generate_response(request.message, session_id)
        
        return ChatResponse(
            response=result["response"],
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            sources_found=result["sources_found"]
        )
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test vector store connection
        test_query = "test"
        test_embedding = embeddings_model.encode(test_query).tolist()
        
        # Get index stats
        index_stats = index.describe_index_stats()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "embeddings_model": embeddings_model is not None,
                "pinecone": pc is not None,
                "index": index is not None,
                "llm": llm is not None,
                "vector_store": vector_store is not None,
                "chatbot": chatbot is not None
            },
            "index_stats": {
                "total_vectors": index_stats.total_vector_count,
                "dimension": index_stats.dimension
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.delete("/clear")
async def clear_index():
    """Clear all vectors from the index"""
    try:
        # Delete all vectors in the index
        index.delete(delete_all=True)
        
        return {
            "message": "Successfully cleared all documents from the index",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error clearing index: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing index: {str(e)}")

def run_server():
    """Run the FastAPI server"""
    uvicorn.run("main:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=DEBUG)

if __name__ == "__main__":
    run_server()