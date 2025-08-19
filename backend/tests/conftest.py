import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any
import sys

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Course, Lesson, CourseChunk
from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem
from config import Config

@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    return Course(
        title="Test Course: Introduction to Testing",
        course_link="https://example.com/course",
        instructor="Test Instructor",
        lessons=[
            Lesson(lesson_number=0, title="Introduction", lesson_link="https://example.com/lesson0"),
            Lesson(lesson_number=1, title="Basic Concepts", lesson_link="https://example.com/lesson1"),
            Lesson(lesson_number=2, title="Advanced Topics")
        ]
    )

@pytest.fixture
def sample_course_chunks():
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="This is lesson 0 content about introduction to testing concepts.",
            course_title="Test Course: Introduction to Testing", 
            lesson_number=0,
            chunk_index=0
        ),
        CourseChunk(
            content="This is lesson 1 content about basic testing concepts and methodologies.",
            course_title="Test Course: Introduction to Testing",
            lesson_number=1, 
            chunk_index=1
        ),
        CourseChunk(
            content="Advanced testing topics including mocking and integration testing.",
            course_title="Test Course: Introduction to Testing",
            lesson_number=2,
            chunk_index=2
        )
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
            "Advanced testing topics including mocking and integration testing."
        ],
        metadata=[
            {"course_title": "Test Course: Introduction to Testing", "lesson_number": 1},
            {"course_title": "Test Course: Introduction to Testing", "lesson_number": 2}
        ],
        distances=[0.1, 0.2]
    )

@pytest.fixture 
def mock_search_results_empty():
    """Create empty search results for testing"""
    return SearchResults(
        documents=[],
        metadata=[], 
        distances=[]
    )

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