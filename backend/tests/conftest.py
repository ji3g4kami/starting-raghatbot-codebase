import os
import shutil
import sys
import tempfile
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any, Union, Optional
from fastapi.testclient import TestClient

import pytest

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from models import Course, CourseChunk, Lesson
from search_tools import CourseSearchTool
from vector_store import SearchResults, VectorStore
from session_manager import SessionManager
from rag_system import RAGSystem
from ai_generator import AIGenerator

@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    return Course(
        title="Test Course: Introduction to Testing",
        course_link="https://example.com/course",
        instructor="Test Instructor",
        lessons=[
            Lesson(
                lesson_number=0,
                title="Introduction",
                lesson_link="https://example.com/lesson0",
            ),
            Lesson(
                lesson_number=1,
                title="Basic Concepts",
                lesson_link="https://example.com/lesson1",
            ),
            Lesson(lesson_number=2, title="Advanced Topics"),
        ],
    )


@pytest.fixture
def sample_course_chunks():
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="This is lesson 0 content about introduction to testing concepts.",
            course_title="Test Course: Introduction to Testing",
            lesson_number=0,
            chunk_index=0,
        ),
        CourseChunk(
            content="This is lesson 1 content about basic testing concepts and methodologies.",
            course_title="Test Course: Introduction to Testing",
            lesson_number=1,
            chunk_index=1,
        ),
        CourseChunk(
            content="Advanced testing topics including mocking and integration testing.",
            course_title="Test Course: Introduction to Testing",
            lesson_number=2,
            chunk_index=2,
        ),
    ]


@pytest.fixture
def mock_vector_store():
    """Create a mock VectorStore for unit testing"""
    mock_store = Mock(spec=VectorStore)
    mock_store.max_results = 5
    return mock_store


@pytest.fixture
def mock_search_results_success():
    """Create successful search results for testing"""
    return SearchResults(
        documents=[
            "This is lesson 1 content about basic testing concepts.",
            "Advanced testing topics including mocking and integration testing.",
        ],
        metadata=[
            {
                "course_title": "Test Course: Introduction to Testing",
                "lesson_number": 1,
            },
            {
                "course_title": "Test Course: Introduction to Testing",
                "lesson_number": 2,
            },
        ],
        distances=[0.1, 0.2],
    )


@pytest.fixture
def mock_search_results_empty():
    """Create empty search results for testing"""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def mock_search_results_error():
    """Create error search results for testing"""
    return SearchResults.empty("Vector store connection error")


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client"""
    mock_client = Mock()

    # Mock response for non-tool queries
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [Mock(text="This is a test response")]

    # Mock response for tool use queries
    mock_tool_response = Mock()
    mock_tool_response.stop_reason = "tool_use"
    mock_tool_content = Mock()
    mock_tool_content.type = "tool_use"
    mock_tool_content.name = "search_course_content"
    mock_tool_content.input = {"query": "test query"}
    mock_tool_content.id = "test_tool_id"
    mock_tool_response.content = [mock_tool_content]

    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def test_config():
    """Create test configuration"""
    config = Config()
    config.ANTHROPIC_API_KEY = "test_api_key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.CHROMA_PATH = "./test_chroma_db"
    config.MAX_RESULTS = 5
    return config


@pytest.fixture
def temp_chroma_db():
    """Create temporary ChromaDB for integration testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def course_search_tool_with_mock(mock_vector_store):
    """Create CourseSearchTool with mock vector store"""
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def sample_test_document_content():
    """Sample course document content for testing"""
    return """Course Title: Test Course for Integration Testing
Course Link: https://example.com/test-course
Course Instructor: Test Instructor

Lesson 0: Introduction to Testing
Lesson Link: https://example.com/lesson0
Welcome to our comprehensive testing course. In this introduction, we'll cover the fundamentals of software testing, including unit testing, integration testing, and test-driven development.

Testing is a critical part of software development that ensures your code works as expected. We'll start with basic concepts and gradually move to more advanced topics.

Lesson 1: Unit Testing Fundamentals
Unit testing involves testing individual components or functions in isolation. This helps identify bugs early in the development process and makes code more maintainable.

Python's unittest framework provides a solid foundation for writing tests. We'll also explore pytest, which offers more powerful features and simpler syntax.

Lesson 2: Mocking and Test Doubles
When testing complex systems, you often need to simulate external dependencies. Mocking allows you to replace real objects with test doubles that behave in predictable ways.

This lesson covers different types of test doubles: mocks, stubs, spies, and fakes. We'll learn when to use each type and how to implement them effectively."""


@pytest.fixture
def anthropic_api_error():
    """Create mock Anthropic API error for testing error handling"""
    import anthropic
    return anthropic.APIError("Test API error", response=Mock(status_code=500), body={})

# API Testing Fixtures

@pytest.fixture
def mock_session_manager():
    """Create a mock SessionManager for API testing"""
    mock_sm = Mock(spec=SessionManager)
    mock_sm.create_session.return_value = "test-session-123"
    mock_sm.get_conversation_history.return_value = []
    mock_sm.add_exchange.return_value = None
    return mock_sm

@pytest.fixture
def mock_rag_system_complete():
    """Create a complete mock RAG system with all components"""
    mock_rag = Mock(spec=RAGSystem)
    
    # Mock session manager
    mock_rag.session_manager = Mock()
    mock_rag.session_manager.create_session.return_value = "test-session-123"
    mock_rag.session_manager.get_conversation_history.return_value = []
    
    # Mock vector store
    mock_rag.vector_store = Mock(spec=VectorStore)
    mock_rag.vector_store.search.return_value = SearchResults(
        documents=["Test content about testing"],
        metadata=[{"course_title": "Test Course", "lesson_number": 1}],
        distances=[0.1]
    )
    
    # Mock AI generator
    mock_rag.ai_generator = Mock(spec=AIGenerator)
    mock_rag.ai_generator.generate.return_value = (
        "This is a test response about testing.",
        [{"content": "Source content", "course": "Test Course", "lesson": "1"}]
    )
    
    # Mock query method
    mock_rag.query.return_value = (
        "This is a test answer based on the course materials.",
        [
            {"content": "Source content 1", "course": "Test Course", "lesson": "1"},
            {"content": "Source content 2", "course": "Test Course", "lesson": "2"}
        ]
    )
    
    # Mock get_course_analytics
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Course 1", "Course 2", "Course 3"]
    }
    
    return mock_rag

@pytest.fixture
def test_api_client(mock_rag_system_complete):
    """Create a test client with mocked RAG system for API testing"""
    from fastapi import FastAPI, HTTPException
    from typing import Optional
    from pydantic import BaseModel
    
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None
    
    class QueryResponse(BaseModel):
        answer: str
        sources: List[Union[str, Dict[str, Optional[str]]]]
        session_id: str
    
    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    app = FastAPI(title="Test RAG API")
    
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system_complete.session_manager.create_session()
            
            answer, sources = mock_rag_system_complete.query(request.query, session_id)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system_complete.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System API", "version": "1.0.0"}
    
    return TestClient(app)

@pytest.fixture
def test_documents_folder(tmp_path):
    """Create a temporary documents folder with test files"""
    docs_dir = tmp_path / "test_docs"
    docs_dir.mkdir()
    
    # Create test course file
    test_file = docs_dir / "test_course.txt"
    test_file.write_text("""Course Title: Test Course for Fixtures
Course Link: https://example.com/test-course
Course Instructor: Test Instructor

Lesson 0: Introduction
This is the introduction content.

Lesson 1: Basic Concepts
This is the basic concepts content.
""")
    
    return str(docs_dir)

@pytest.fixture
def mock_chromadb_client():
    """Create a mock ChromaDB client"""
    import chromadb
    mock_client = Mock()
    
    # Mock collection
    mock_collection = Mock()
    mock_collection.add.return_value = None
    mock_collection.query.return_value = {
        'documents': [["Test content"]],
        'metadatas': [[{"course_title": "Test Course", "lesson_number": 1}]],
        'distances': [[0.1]]
    }
    mock_collection.get.return_value = {
        'documents': ["Test content"],
        'metadatas': [{"course_title": "Test Course"}],
        'ids': ["test-id-1"]
    }
    mock_collection.count.return_value = 1
    
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client.create_collection.return_value = mock_collection
    mock_client.get_collection.return_value = mock_collection
    
    return mock_client

@pytest.fixture(autouse=True)
def cleanup_chromadb():
    """Automatically cleanup ChromaDB after each test"""
    yield
    # Cleanup any test ChromaDB instances
    test_db_path = "./test_chroma_db"
    if os.path.exists(test_db_path):
        shutil.rmtree(test_db_path, ignore_errors=True)

@pytest.fixture
def mock_tool_result():
    """Create a mock tool result for AI generator testing"""
    return {
        "results": ["Test search result 1", "Test search result 2"],
        "metadata": [
            {"course_title": "Test Course", "lesson_number": "1"},
            {"course_title": "Test Course", "lesson_number": "2"}
        ]
    }

@pytest.fixture
def integration_test_config():
    """Create configuration for integration tests"""
    config = Config()
    config.ANTHROPIC_API_KEY = "test_api_key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.CHROMA_PATH = "./test_integration_chroma_db"
    config.MAX_RESULTS = 5
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    return config

@pytest.fixture
def mock_fastapi_dependencies():
    """Mock all FastAPI app dependencies for testing"""
    mocks = {
        'static_files': Mock(),
        'cors_middleware': Mock(),
        'trusted_host_middleware': Mock()
    }
    return mocks
