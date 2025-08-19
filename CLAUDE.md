# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) chatbot system for course materials, using FastAPI, ChromaDB for vector storage, and Anthropic's Claude for AI responses. The system processes course transcripts, chunks them semantically, and provides intelligent Q&A with source citations.

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
chmod +x run.sh
./run.sh

# Manual start
cd backend
uv run uvicorn app:app --reload --port 8000
```

### Managing Background Server Processes
```bash
# Check if server is running on port 8000
lsof -i :8000

# Kill server running on port 8000
pkill -f "uvicorn app:app"

# Alternative: Kill by port number
lsof -i :8000 | grep LISTEN | awk '{print $2}' | xargs kill -9

# Check for any Python processes running uvicorn
ps aux | grep uvicorn

# Kill all uvicorn processes
pkill -f uvicorn
```

### Package Management
```bash
# Install dependencies (uses uv package manager)
uv sync

# Add new dependency
uv add <package_name>
```

### Running Tests
```bash
# Run all tests from backend directory
cd backend
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_vector_store.py -v

# Run tests with short traceback for cleaner output
uv run pytest tests/ -v --tb=short

# Run specific test by name pattern
uv run pytest tests/ -k "test_search" -v

# Run tests quietly (only show summary)
uv run pytest tests/ -q

# Run tests with coverage report
uv run pytest tests/ --cov=. --cov-report=term-missing
```

### Environment Setup
1. Ensure Python 3.13+ is installed
2. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. Create `.env` file with: `ANTHROPIC_API_KEY=your_key_here`
4. Access at `http://localhost:8000`

## Architecture Overview

### Core Query Flow
1. **Frontend** (`/frontend/script.js`) → POST to `/api/query` with query + session_id
2. **FastAPI** (`/backend/app.py:56-74`) → Routes to RAG system
3. **RAGSystem** (`/backend/rag_system.py:102-140`) → Orchestrates the pipeline:
   - Retrieves conversation history from SessionManager
   - Calls AIGenerator with tool definitions
4. **AIGenerator** (`/backend/ai_generator.py:43-135`) → Claude decides to use search tool
5. **CourseSearchTool** (`/backend/search_tools.py:52-86`) → Executes semantic search
6. **VectorStore** (`/backend/vector_store.py:61-100`) → Dual-collection ChromaDB search:
   - Course catalog collection for name resolution
   - Content collection for chunk retrieval
7. **Response Generation** → Claude processes results and generates answer
8. **Frontend** → Renders markdown response with collapsible sources

### Key Components

**Data Models** (`/backend/models.py`):
- `Course`: Title, instructor, lessons list
- `Lesson`: Number, title, optional link
- `CourseChunk`: Content with course/lesson metadata

**Document Processing** (`/backend/document_processor.py`):
- Parses structured course format (Title/Link/Instructor, then Lessons)
- Sentence-based chunking (800 chars, 100 overlap)
- Preserves lesson context in chunks

**Vector Storage** (`/backend/vector_store.py`):
- Two ChromaDB collections: `course_catalog` and `course_content`
- Semantic course name resolution
- Filtered search by course/lesson

**AI Integration** (`/backend/ai_generator.py`):
- Tool-calling architecture with CourseSearchTool
- System prompt enforces educational, concise responses
- Max 800 tokens, temperature 0

**Session Management** (`/backend/session_manager.py`):
- Tracks conversation history (max 2 exchanges)
- Session isolation for multi-user support

## Document Format Convention

Course documents must follow this structure:
```
Course Title: [title]
Course Link: [optional URL]
Course Instructor: [name]

Lesson 0: [title]
Lesson Link: [optional URL]
[lesson content...]

Lesson 1: [title]
[lesson content...]
```

## Configuration

Key settings in `/backend/config.py`:
- `ANTHROPIC_MODEL`: claude-sonnet-4-20250514
- `EMBEDDING_MODEL`: all-MiniLM-L6-v2
- `CHUNK_SIZE`: 800 characters
- `CHUNK_OVERLAP`: 100 characters
- `MAX_RESULTS`: 5 search results
- `MAX_HISTORY`: 2 conversation turns

## Important Implementation Details

### ChromaDB Persistence
- Database stored in `/backend/chroma_db/`
- Automatic deduplication by course title
- Documents loaded on startup from `/docs/`

### Tool-Calling Pattern
The system uses Anthropic's tool-calling feature where Claude decides when to search:
- One search maximum per query
- Search results injected as context
- Claude synthesizes results into educational responses

### Frontend State Management
- Session ID tracked globally in `currentSessionId`
- Loading states handled with dedicated message elements
- Markdown rendering via marked.js library

### Error Handling
- Graceful fallbacks for missing courses/lessons
- UTF-8 encoding with error handling for documents
- Empty result handling with clear user messages

## Working with the Codebase

When modifying this system:
1. **Adding new documents**: Place `.txt` files in `/docs/` following the format convention
2. **Modifying search**: Update `CourseSearchTool` and `VectorStore.search()` together
3. **Changing AI behavior**: Adjust `SYSTEM_PROMPT` in `AIGenerator`
4. **Frontend changes**: Remember to handle loading states and error cases
5. **New API endpoints**: Add to `app.py` with Pydantic models for validation

## Testing

The project includes a comprehensive test suite with 87+ tests covering:
- **Unit tests**: Individual component testing (search tools, AI generator, vector store)
- **Integration tests**: End-to-end RAG system testing
- **Edge case testing**: Malformed documents, missing fields, error handling

Test files are located in `/backend/tests/` with fixtures in `conftest.py`.

## Current Limitations

- No linting/formatting configuration
- Single-node deployment only (no distributed ChromaDB)
- Manual document upload required (no admin interface)
- No user authentication/authorization
- always use uv to run the server, do not use pip directly.
- use uv to run Python files.
- make sure to use uv to manage all dependencies