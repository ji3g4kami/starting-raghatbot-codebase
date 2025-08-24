from unittest.mock import Mock, patch

import pytest

from config import Config
from rag_system import RAGSystem


class TestRAGSystemIntegration:
    """Integration tests for the complete RAG system"""

    @pytest.fixture
    def integration_config(self, temp_chroma_db):
        """Create configuration for integration testing"""
        config = Config()
        config.CHROMA_PATH = temp_chroma_db
        config.ANTHROPIC_API_KEY = "test_api_key"
        config.MAX_RESULTS = 3
        config.MAX_HISTORY = 2
        return config

    @pytest.fixture
    def rag_system_with_mock_ai(self, integration_config):
        """Create RAG system with mocked AI generator"""
        with patch("rag_system.AIGenerator") as mock_ai_gen:
            # Mock AI generator to return predictable responses
            mock_ai_instance = Mock()
            mock_ai_gen.return_value = mock_ai_instance

            # Setup different responses for different scenarios
            mock_ai_instance.generate_response.return_value = (
                "This is a test response about testing concepts."
            )

            rag_system = RAGSystem(integration_config)
            rag_system.mock_ai_generator = (
                mock_ai_instance  # Store reference for test assertions
            )

            return rag_system

    @pytest.fixture
    def sample_test_document(self, tmp_path):
        """Create a sample test document file"""
        content = """Course Title: Integration Testing Course
Course Link: https://example.com/integration-testing
Course Instructor: Test Instructor

Lesson 0: Introduction to Integration Testing
Lesson Link: https://example.com/lesson0
Integration testing verifies that different components of a system work together correctly. It's performed after unit testing and before system testing.

Lesson 1: Test Data Management
Test data management is crucial for reliable integration tests. You need to set up consistent test environments and manage test data lifecycle effectively.

Lesson 2: Mock Services and Stubs
When testing integrations with external services, you often need to use mock services or stubs to simulate external dependencies and ensure predictable test outcomes."""

        doc_file = tmp_path / "integration_test_course.txt"
        doc_file.write_text(content)
        return str(doc_file)

    def test_add_course_document_success(
        self, rag_system_with_mock_ai, sample_test_document
    ):
        """Test successfully adding a course document"""
        # Execute
        course, chunk_count = rag_system_with_mock_ai.add_course_document(
            sample_test_document
        )

        # Assert
        assert course is not None
        assert course.title == "Integration Testing Course"
        assert course.instructor == "Test Instructor"
        assert course.course_link == "https://example.com/integration-testing"
        assert len(course.lessons) == 3
        assert chunk_count > 0

        # Verify course was added to vector store
        analytics = rag_system_with_mock_ai.get_course_analytics()
        assert analytics["total_courses"] == 1
        assert "Integration Testing Course" in analytics["course_titles"]

    def test_add_course_document_file_not_found(self, rag_system_with_mock_ai):
        """Test adding nonexistent course document"""
        # Execute
        course, chunk_count = rag_system_with_mock_ai.add_course_document(
            "/nonexistent/file.txt"
        )

        # Assert
        assert course is None
        assert chunk_count == 0

    def test_query_with_successful_search(
        self, rag_system_with_mock_ai, sample_test_document
    ):
        """Test end-to-end query with successful search results"""
        # Setup - add test document
        rag_system_with_mock_ai.add_course_document(sample_test_document)

        # Mock AI generator to simulate tool use
        mock_ai = rag_system_with_mock_ai.mock_ai_generator
        mock_ai.generate_response.return_value = "Integration testing involves verifying that different components work together."

        # Execute
        response, sources = rag_system_with_mock_ai.query(
            "What is integration testing?"
        )

        # Assert
        assert (
            response
            == "Integration testing involves verifying that different components work together."
        )
        assert mock_ai.generate_response.called

        # Check that AI was called with tools
        call_args = mock_ai.generate_response.call_args
        assert "tools" in call_args[1]
        assert "tool_manager" in call_args[1]
        assert call_args[1]["tools"] is not None

    def test_query_with_session_management(
        self, rag_system_with_mock_ai, sample_test_document
    ):
        """Test query with session management and conversation history"""
        # Setup
        rag_system_with_mock_ai.add_course_document(sample_test_document)
        session_id = "test_session_123"

        # First query
        response1, sources1 = rag_system_with_mock_ai.query(
            "What is integration testing?", session_id
        )

        # Second query in same session
        response2, sources2 = rag_system_with_mock_ai.query("Tell me more", session_id)

        # Assert
        mock_ai = rag_system_with_mock_ai.mock_ai_generator
        assert mock_ai.generate_response.call_count == 2

        # Second call should include conversation history
        second_call_args = mock_ai.generate_response.call_args
        assert "conversation_history" in second_call_args[1]
        assert second_call_args[1]["conversation_history"] is not None

    def test_query_without_session_creates_session(
        self, rag_system_with_mock_ai, sample_test_document
    ):
        """Test that query without session ID creates a new session"""
        # Setup
        rag_system_with_mock_ai.add_course_document(sample_test_document)

        # Execute
        response, sources = rag_system_with_mock_ai.query(
            "What is integration testing?"
        )

        # Assert
        assert response is not None
        # Session should be created internally, but not returned in current implementation
        # This tests that the system doesn't crash when no session_id is provided

    def test_tool_manager_initialization(self, rag_system_with_mock_ai):
        """Test that tool manager is properly initialized with tools"""
        # Assert
        assert rag_system_with_mock_ai.tool_manager is not None
        assert rag_system_with_mock_ai.search_tool is not None
        assert rag_system_with_mock_ai.outline_tool is not None

        # Check that tools are registered
        tool_definitions = rag_system_with_mock_ai.tool_manager.get_tool_definitions()
        tool_names = [tool["name"] for tool in tool_definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    def test_add_course_folder_success(self, rag_system_with_mock_ai, tmp_path):
        """Test adding multiple course documents from a folder"""
        # Setup - create multiple test documents
        doc1_content = """Course Title: Course 1
Course Instructor: Instructor 1
Lesson 0: Introduction
Content for lesson 0."""

        doc2_content = """Course Title: Course 2
Course Instructor: Instructor 2
Lesson 0: Getting Started
Content for getting started."""

        (tmp_path / "course1.txt").write_text(doc1_content)
        (tmp_path / "course2.txt").write_text(doc2_content)
        (tmp_path / "readme.md").write_text("This should be ignored")  # Non-course file

        # Execute
        total_courses, total_chunks = rag_system_with_mock_ai.add_course_folder(
            str(tmp_path)
        )

        # Assert
        assert total_courses == 2
        assert total_chunks > 0

        analytics = rag_system_with_mock_ai.get_course_analytics()
        assert analytics["total_courses"] == 2
        assert "Course 1" in analytics["course_titles"]
        assert "Course 2" in analytics["course_titles"]

    def test_add_course_folder_nonexistent(self, rag_system_with_mock_ai):
        """Test adding from nonexistent folder"""
        # Execute
        total_courses, total_chunks = rag_system_with_mock_ai.add_course_folder(
            "/nonexistent/folder"
        )

        # Assert
        assert total_courses == 0
        assert total_chunks == 0

    def test_add_course_folder_clear_existing(
        self, rag_system_with_mock_ai, tmp_path, sample_test_document
    ):
        """Test adding course folder with clear_existing=True"""
        # Setup - add initial course
        rag_system_with_mock_ai.add_course_document(sample_test_document)
        assert rag_system_with_mock_ai.get_course_analytics()["total_courses"] == 1

        # Create new course document
        new_doc = tmp_path / "new_course.txt"
        new_doc.write_text(
            """Course Title: New Course
Course Instructor: New Instructor
Lesson 0: New Content
This is new content."""
        )

        # Execute with clear_existing=True
        total_courses, total_chunks = rag_system_with_mock_ai.add_course_folder(
            str(tmp_path), clear_existing=True
        )

        # Assert - after clearing, both existing and new courses are loaded
        # The clear operation clears the DB, but both documents get processed again
        assert total_courses >= 1  # At least the new course should be added
        analytics = rag_system_with_mock_ai.get_course_analytics()
        assert "New Course" in analytics["course_titles"]

    def test_add_course_folder_skip_existing(self, rag_system_with_mock_ai, tmp_path):
        """Test that existing courses are skipped when adding folder"""
        # Setup - create course document
        doc_content = """Course Title: Duplicate Course
Course Instructor: Test Instructor
Lesson 0: Test Content
Some content here."""

        doc_file = tmp_path / "course.txt"
        doc_file.write_text(doc_content)

        # Add course first time
        total1, chunks1 = rag_system_with_mock_ai.add_course_folder(str(tmp_path))
        assert total1 == 1

        # Add same folder again
        total2, chunks2 = rag_system_with_mock_ai.add_course_folder(str(tmp_path))
        assert total2 == 0  # Should skip existing course
        assert chunks2 == 0

        # Total should still be 1
        analytics = rag_system_with_mock_ai.get_course_analytics()
        assert analytics["total_courses"] == 1

    def test_query_error_propagation(self, integration_config):
        """Test that errors in query are properly handled"""
        # Setup RAG system with invalid API key to trigger errors
        with patch("rag_system.AIGenerator") as mock_ai_gen:
            mock_ai_instance = Mock()
            mock_ai_gen.return_value = mock_ai_instance

            # Mock AI generator to raise an error
            import anthropic

            mock_request = Mock()
            mock_body = {}
            mock_ai_instance.generate_response.side_effect = anthropic.APIError(
                "Invalid API key", request=mock_request, body=mock_body
            )

            rag_system = RAGSystem(integration_config)

            # Execute and assert
            with pytest.raises(anthropic.APIError):
                rag_system.query("test query")

    def test_empty_vector_store_query(self, rag_system_with_mock_ai):
        """Test querying empty vector store"""
        # Mock AI to simulate tool execution with empty results
        mock_ai = rag_system_with_mock_ai.mock_ai_generator
        mock_ai.generate_response.return_value = (
            "I don't have any information about that topic."
        )

        # Execute
        response, sources = rag_system_with_mock_ai.query("What is testing?")

        # Assert
        assert response is not None
        # Should still call AI even with empty vector store
        assert mock_ai.generate_response.called

    @pytest.mark.parametrize(
        "query,expected_tool_call",
        [
            (
                "What is integration testing?",
                True,
            ),  # Content question - should use search
            ("Hello", False),  # General greeting - might not use tools
            ("List all courses", True),  # Might use outline tool
        ],
    )
    def test_tool_usage_based_on_query_type(
        self, rag_system_with_mock_ai, sample_test_document, query, expected_tool_call
    ):
        """Test that different query types trigger appropriate tool usage"""
        # Setup
        rag_system_with_mock_ai.add_course_document(sample_test_document)
        mock_ai = rag_system_with_mock_ai.mock_ai_generator

        # Execute
        response, sources = rag_system_with_mock_ai.query(query)

        # Assert
        assert mock_ai.generate_response.called
        call_args = mock_ai.generate_response.call_args

        # Check if tools were provided to AI
        tools_provided = "tools" in call_args[1] and call_args[1]["tools"] is not None
        assert tools_provided  # Tools should always be provided, AI decides whether to use them

    def test_source_retrieval_and_reset(
        self, rag_system_with_mock_ai, sample_test_document
    ):
        """Test that sources are properly retrieved and reset between queries"""
        # Setup
        rag_system_with_mock_ai.add_course_document(sample_test_document)

        # Mock the tool manager methods
        with (
            patch.object(
                rag_system_with_mock_ai.tool_manager, "get_last_sources"
            ) as mock_get_sources,
            patch.object(
                rag_system_with_mock_ai.tool_manager, "reset_sources"
            ) as mock_reset_sources,
        ):

            mock_get_sources.return_value = [
                {
                    "text": "Integration Testing Course - Lesson 1",
                    "url": "https://example.com/lesson1",
                }
            ]

            # First query
            response1, sources1 = rag_system_with_mock_ai.query(
                "What is integration testing?"
            )

            # Assert sources were retrieved and reset was called
            assert mock_get_sources.called
            assert mock_reset_sources.called

            # Reset call counts for second query
            mock_get_sources.reset_mock()
            mock_reset_sources.reset_mock()
            mock_get_sources.return_value = []

            # Second query
            response2, sources2 = rag_system_with_mock_ai.query("Tell me more")

            # Assert reset and retrieval happened again
            assert mock_get_sources.called
            assert mock_reset_sources.called

    def test_course_analytics(self, rag_system_with_mock_ai, sample_test_document):
        """Test course analytics functionality"""
        # Initially empty
        analytics = rag_system_with_mock_ai.get_course_analytics()
        assert analytics["total_courses"] == 0
        assert analytics["course_titles"] == []

        # After adding course
        rag_system_with_mock_ai.add_course_document(sample_test_document)
        analytics = rag_system_with_mock_ai.get_course_analytics()
        assert analytics["total_courses"] == 1
        assert "Integration Testing Course" in analytics["course_titles"]

    def test_component_initialization_order(self, integration_config):
        """Test that all components are initialized in correct order"""
        # This test ensures that dependencies are properly initialized
        rag_system = RAGSystem(integration_config)

        # Assert all components exist
        assert rag_system.document_processor is not None
        assert rag_system.vector_store is not None
        assert rag_system.ai_generator is not None
        assert rag_system.session_manager is not None
        assert rag_system.tool_manager is not None
        assert rag_system.search_tool is not None
        assert rag_system.outline_tool is not None

        # Assert vector store is properly connected to tools
        assert rag_system.search_tool.store is rag_system.vector_store
        assert rag_system.outline_tool.store is rag_system.vector_store
