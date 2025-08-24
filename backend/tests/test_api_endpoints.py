import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import json
from typing import List, Optional, Dict, Union
from pydantic import BaseModel

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_system import RAGSystem
from config import Config


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


def create_test_app(rag_system):
    """Create a test FastAPI app with only API endpoints (no static files)"""
    app = FastAPI(title="Test RAG API")
    
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()
            
            answer, sources = rag_system.query(request.query, session_id)
            
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
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System API", "version": "1.0.0"}
    
    return app


@pytest.fixture
def mock_rag_system():
    """Create a mock RAG system for API testing"""
    mock_rag = Mock(spec=RAGSystem)
    
    # Mock session manager
    mock_rag.session_manager = Mock()
    mock_rag.session_manager.create_session.return_value = "test-session-123"
    
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
def test_client(mock_rag_system):
    """Create test client with mock RAG system"""
    app = create_test_app(mock_rag_system)
    return TestClient(app), mock_rag_system


class TestAPIEndpoints:
    """Test suite for API endpoints"""
    
    def test_root_endpoint(self, test_client):
        """Test the root endpoint returns API information"""
        client, _ = test_client
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["message"] == "Course Materials RAG System API"
        assert data["version"] == "1.0.0"
    
    def test_query_endpoint_success(self, test_client):
        """Test successful query processing"""
        client, mock_rag = test_client
        
        response = client.post(
            "/api/query",
            json={"query": "What is unit testing?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        
        # Verify mock was called correctly
        mock_rag.query.assert_called_once_with("What is unit testing?", "test-session-123")
        
        # Verify response content
        assert data["answer"] == "This is a test answer based on the course materials."
        assert len(data["sources"]) == 2
        assert data["session_id"] == "test-session-123"
    
    def test_query_endpoint_with_session_id(self, test_client):
        """Test query with existing session ID"""
        client, mock_rag = test_client
        
        response = client.post(
            "/api/query",
            json={
                "query": "Tell me more about testing",
                "session_id": "existing-session-456"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should use provided session ID
        mock_rag.query.assert_called_once_with("Tell me more about testing", "existing-session-456")
        assert data["session_id"] == "existing-session-456"
        
        # Session manager should not create new session
        mock_rag.session_manager.create_session.assert_not_called()
    
    def test_query_endpoint_empty_query(self, test_client):
        """Test query with empty string"""
        client, mock_rag = test_client
        
        # Mock empty query response
        mock_rag.query.return_value = ("Please provide a question.", [])
        
        response = client.post(
            "/api/query",
            json={"query": ""}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Please provide a question."
        assert data["sources"] == []
    
    def test_query_endpoint_error_handling(self, test_client):
        """Test error handling in query endpoint"""
        client, mock_rag = test_client
        
        # Mock query to raise exception
        mock_rag.query.side_effect = Exception("Database connection failed")
        
        response = client.post(
            "/api/query",
            json={"query": "What is testing?"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Database connection failed" in data["detail"]
    
    def test_query_endpoint_invalid_request(self, test_client):
        """Test query endpoint with invalid request body"""
        client, _ = test_client
        
        # Missing required field
        response = client.post(
            "/api/query",
            json={}
        )
        
        assert response.status_code == 422  # Unprocessable Entity
        data = response.json()
        assert "detail" in data
    
    def test_courses_endpoint_success(self, test_client):
        """Test successful course statistics retrieval"""
        client, mock_rag = test_client
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "total_courses" in data
        assert "course_titles" in data
        
        # Verify mock was called
        mock_rag.get_course_analytics.assert_called_once()
        
        # Verify response content
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Course 1" in data["course_titles"]
    
    def test_courses_endpoint_empty(self, test_client):
        """Test courses endpoint with no courses"""
        client, mock_rag = test_client
        
        # Mock empty analytics
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []
    
    def test_courses_endpoint_error_handling(self, test_client):
        """Test error handling in courses endpoint"""
        client, mock_rag = test_client
        
        # Mock to raise exception
        mock_rag.get_course_analytics.side_effect = Exception("Vector store not initialized")
        
        response = client.get("/api/courses")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Vector store not initialized" in data["detail"]
    
    def test_query_endpoint_with_complex_sources(self, test_client):
        """Test query with various source formats"""
        client, mock_rag = test_client
        
        # Mock complex sources
        mock_rag.query.return_value = (
            "Answer with multiple source types",
            [
                "Simple string source",
                {"content": "Dict source", "course": "Advanced Testing", "lesson": "5"},
                {"content": "Partial dict", "course": None, "lesson": None}
            ]
        )
        
        response = client.post(
            "/api/query",
            json={"query": "Complex query"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 3
        
        # Verify different source formats are handled
        assert isinstance(data["sources"][0], str)
        assert isinstance(data["sources"][1], dict)
        assert data["sources"][1]["course"] == "Advanced Testing"
    
    def test_query_endpoint_long_query(self, test_client):
        """Test query with very long input"""
        client, mock_rag = test_client
        
        long_query = "What is " + "testing " * 500 + "?"
        
        response = client.post(
            "/api/query",
            json={"query": long_query}
        )
        
        assert response.status_code == 200
        mock_rag.query.assert_called_once()
        
        # Verify the full query was passed
        called_query = mock_rag.query.call_args[0][0]
        assert len(called_query) > 1000
    
    def test_concurrent_queries(self, test_client):
        """Test handling multiple concurrent queries"""
        client, mock_rag = test_client
        
        # Reset mock to track multiple calls
        mock_rag.query.reset_mock()
        mock_rag.session_manager.create_session.reset_mock()
        
        # Simulate different sessions
        session_ids = ["session-1", "session-2", "session-3"]
        mock_rag.session_manager.create_session.side_effect = session_ids
        
        # Make multiple queries
        responses = []
        for i in range(3):
            response = client.post(
                "/api/query",
                json={"query": f"Query {i}"}
            )
            responses.append(response)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
        
        # Verify all queries were processed
        assert mock_rag.query.call_count == 3
        assert mock_rag.session_manager.create_session.call_count == 3
        
        # Verify different session IDs
        for i, response in enumerate(responses):
            data = response.json()
            assert data["session_id"] == session_ids[i]


class TestAPIIntegration:
    """Integration tests with real components (but mocked external services)"""
    
    @pytest.fixture
    def integration_rag_system(self, test_config, mock_anthropic_client):
        """Create RAG system with mocked external dependencies"""
        with patch('anthropic.Anthropic') as mock_anthropic_class:
            mock_anthropic_class.return_value = mock_anthropic_client
            
            # Create RAG system with test config
            rag_system = RAGSystem(test_config)
            
            # Mock vector store to avoid real ChromaDB
            mock_vector_store = Mock()
            mock_vector_store.search.return_value = Mock(
                documents=["Test content"],
                metadata=[{"course_title": "Test Course", "lesson_number": 1}],
                distances=[0.1]
            )
            mock_vector_store.get_course_analytics.return_value = {
                "total_courses": 1,
                "course_titles": ["Test Course"]
            }
            
            rag_system.vector_store = mock_vector_store
            
            return rag_system
    
    @pytest.fixture
    def integration_client(self, integration_rag_system):
        """Create test client with integration RAG system"""
        app = create_test_app(integration_rag_system)
        return TestClient(app), integration_rag_system
    
    def test_full_query_flow(self, integration_client):
        """Test complete query processing flow"""
        client, rag_system = integration_client
        
        response = client.post(
            "/api/query",
            json={"query": "What is testing?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have created session
        assert "session_id" in data
        assert data["session_id"] is not None
        
        # Should have answer (from mocked AI)
        assert "answer" in data
        assert len(data["answer"]) > 0
    
    def test_session_persistence(self, integration_client):
        """Test that sessions persist across queries"""
        client, rag_system = integration_client
        
        # First query creates session
        response1 = client.post(
            "/api/query",
            json={"query": "First question"}
        )
        
        assert response1.status_code == 200
        session_id = response1.json()["session_id"]
        
        # Second query with same session
        response2 = client.post(
            "/api/query",
            json={
                "query": "Follow-up question",
                "session_id": session_id
            }
        )
        
        assert response2.status_code == 200
        assert response2.json()["session_id"] == session_id
        
        # Verify session has history
        history = rag_system.session_manager.get_conversation_history(session_id)
        assert history is not None  # Should have conversation history


class TestAPIPerformance:
    """Performance and load testing for API endpoints"""
    
    def test_query_response_time(self, test_client):
        """Test that query endpoint responds within acceptable time"""
        import time
        client, _ = test_client
        
        start_time = time.time()
        response = client.post(
            "/api/query",
            json={"query": "Quick test query"}
        )
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Should respond within 1 second for mocked system
        response_time = end_time - start_time
        assert response_time < 1.0
    
    def test_courses_response_time(self, test_client):
        """Test that courses endpoint responds quickly"""
        import time
        client, _ = test_client
        
        start_time = time.time()
        response = client.get("/api/courses")
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Should respond within 0.5 seconds
        response_time = end_time - start_time
        assert response_time < 0.5
    
    def test_handle_many_sessions(self, test_client):
        """Test handling many concurrent sessions"""
        client, mock_rag = test_client
        
        # Create many sessions
        session_ids = []
        for i in range(100):
            mock_rag.session_manager.create_session.return_value = f"session-{i}"
            response = client.post(
                "/api/query",
                json={"query": f"Query {i}"}
            )
            assert response.status_code == 200
            session_ids.append(response.json()["session_id"])
        
        # All sessions should be unique
        assert len(set(session_ids)) == 100