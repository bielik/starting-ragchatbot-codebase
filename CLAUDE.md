# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Development Setup
```bash
# Install dependencies
uv sync

# Add new dependencies
uv add package_name

# Set up environment variables
cp .env.example .env
# Edit .env to add your ANTHROPIC_API_KEY
```

### Running the Application
```bash
# Quick start (recommended)
chmod +x run.sh
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Development Commands
```bash
# Run from backend directory
cd backend && uv run uvicorn app:app --reload --port 8000

# Test API endpoints directly
curl http://localhost:8000/api/courses
curl -X POST http://localhost:8000/api/query -H "Content-Type: application/json" -d '{"query":"your question here"}'
```

## Architecture Overview

This is a **RAG (Retrieval-Augmented Generation) System** with a three-layer architecture:

### Core RAG Pipeline
1. **Document Processing**: Course transcripts → chunked text with metadata
2. **Vector Storage**: ChromaDB stores embeddings for semantic search  
3. **AI Generation**: Claude API generates contextual responses using retrieved content
4. **Tool Integration**: AI can dynamically search the knowledge base using tools

### Key Components

**RAGSystem (`rag_system.py`)** - Central orchestrator that coordinates:
- Document processing and chunking
- Vector storage operations
- AI response generation with tool access
- Session management for conversation history

**Tool-Based Search Architecture** - The system uses a tool-based approach where:
- `ToolManager` registers and manages available tools
- `CourseSearchTool` performs content searches within course materials
- `CourseOutlineTool` retrieves complete course structures with lesson lists
- Claude API calls tools dynamically during response generation
- Tools return sources that are tracked and returned to frontend with links

**Data Models** (`models.py`):
- `Course`: Contains title, instructor, lessons list
- `CourseChunk`: Text chunks with course/lesson metadata for vector storage
- `Lesson`: Individual lessons with titles and optional links

### Configuration (`config.py`)
Key settings:
- `CHUNK_SIZE: 800` - Text chunk size for vector storage
- `CHUNK_OVERLAP: 100` - Overlap between chunks
- `MAX_RESULTS: 5` - Vector search result limit
- `MAX_HISTORY: 2` - Conversation memory depth

### Data Flow
1. Course documents in `docs/` are processed into `CourseChunk` objects
2. Chunks are embedded and stored in ChromaDB (`./chroma_db/`)
3. User queries trigger tool-based searches via Claude API
4. Retrieved chunks provide context for AI response generation
5. Session history maintains conversation continuity

### Frontend Integration
- FastAPI serves both API endpoints (`/api/*`) and static frontend files
- Frontend communicates via `/api/query` for chat and `/api/courses` for statistics
- CORS configured for development with live reload support

## Environment Requirements

Required environment variable:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

The system expects course documents in `docs/` folder as `.txt`, `.pdf`, or `.docx` files.
