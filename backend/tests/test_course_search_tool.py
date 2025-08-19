import pytest
from unittest.mock import Mock, patch
from search_tools import CourseSearchTool
from vector_store import SearchResults

class TestCourseSearchTool:
    """Test suite for CourseSearchTool.execute() method"""
    
    def test_execute_successful_query(self, course_search_tool_with_mock, mock_search_results_success):
        """Test successful query execution with results"""
        # Setup
        tool = course_search_tool_with_mock
        tool.store.search.return_value = mock_search_results_success
        tool.store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        # Execute
        result = tool.execute(query="test query")
        
        # Assert
        assert "[Test Course: Introduction to Testing - Lesson 1]" in result
        assert "This is lesson 1 content about basic testing concepts." in result
        assert tool.store.search.called
        assert len(tool.last_sources) == 2
        
        # Check sources are properly formatted
        assert tool.last_sources[0]["text"] == "Test Course: Introduction to Testing - Lesson 1"
        assert tool.last_sources[0]["url"] == "https://example.com/lesson1"
    
    def test_execute_empty_results(self, course_search_tool_with_mock, mock_search_results_empty):
        """Test handling of empty search results"""
        # Setup  
        tool = course_search_tool_with_mock
        tool.store.search.return_value = mock_search_results_empty
        
        # Execute
        result = tool.execute(query="nonexistent query")
        
        # Assert
        assert result == "No relevant content found."
        assert tool.store.search.called
        assert len(tool.last_sources) == 0
    
    def test_execute_empty_results_with_course_filter(self, course_search_tool_with_mock, mock_search_results_empty):
        """Test empty results with course name filter shows appropriate message"""
        # Setup
        tool = course_search_tool_with_mock  
        tool.store.search.return_value = mock_search_results_empty
        
        # Execute
        result = tool.execute(query="test query", course_name="Nonexistent Course")
        
        # Assert
        assert result == "No relevant content found in course 'Nonexistent Course'."
        assert tool.store.search.called
    
    def test_execute_empty_results_with_lesson_filter(self, course_search_tool_with_mock, mock_search_results_empty):
        """Test empty results with lesson number filter"""
        # Setup
        tool = course_search_tool_with_mock
        tool.store.search.return_value = mock_search_results_empty
        
        # Execute  
        result = tool.execute(query="test query", lesson_number=99)
        
        # Assert
        assert result == "No relevant content found in lesson 99."
        assert tool.store.search.called
    
    def test_execute_empty_results_with_both_filters(self, course_search_tool_with_mock, mock_search_results_empty):
        """Test empty results with both course and lesson filters"""
        # Setup
        tool = course_search_tool_with_mock
        tool.store.search.return_value = mock_search_results_empty
        
        # Execute
        result = tool.execute(query="test query", course_name="Test Course", lesson_number=5)
        
        # Assert
        assert result == "No relevant content found in course 'Test Course' in lesson 5."
        assert tool.store.search.called
    
    def test_execute_with_vector_store_error(self, course_search_tool_with_mock, mock_search_results_error):
        """Test handling of vector store errors"""
        # Setup
        tool = course_search_tool_with_mock
        tool.store.search.return_value = mock_search_results_error
        
        # Execute
        result = tool.execute(query="test query")
        
        # Assert
        assert result == "Vector store connection error"
        assert tool.store.search.called
        assert len(tool.last_sources) == 0
    
    def test_execute_with_course_name_parameter(self, course_search_tool_with_mock, mock_search_results_success):
        """Test query execution with course name parameter"""
        # Setup
        tool = course_search_tool_with_mock
        tool.store.search.return_value = mock_search_results_success
        
        # Execute
        result = tool.execute(query="test query", course_name="Test Course")
        
        # Assert
        tool.store.search.assert_called_once_with(
            query="test query",
            course_name="Test Course", 
            lesson_number=None
        )
        assert "This is lesson 1 content about basic testing concepts." in result
    
    def test_execute_with_lesson_number_parameter(self, course_search_tool_with_mock, mock_search_results_success):
        """Test query execution with lesson number parameter"""
        # Setup
        tool = course_search_tool_with_mock
        tool.store.search.return_value = mock_search_results_success
        
        # Execute  
        result = tool.execute(query="test query", lesson_number=1)
        
        # Assert
        tool.store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=1
        )
        assert "This is lesson 1 content about basic testing concepts." in result
    
    def test_execute_with_both_parameters(self, course_search_tool_with_mock, mock_search_results_success):
        """Test query execution with both course name and lesson number"""
        # Setup
        tool = course_search_tool_with_mock
        tool.store.search.return_value = mock_search_results_success
        
        # Execute
        result = tool.execute(query="test query", course_name="Test Course", lesson_number=1)
        
        # Assert
        tool.store.search.assert_called_once_with(
            query="test query", 
            course_name="Test Course",
            lesson_number=1
        )
        assert "This is lesson 1 content about basic testing concepts." in result
    
    def test_format_results_with_lesson_links(self, course_search_tool_with_mock, mock_search_results_success):
        """Test that lesson links are properly retrieved and stored in sources"""
        # Setup
        tool = course_search_tool_with_mock
        tool.store.get_lesson_link.side_effect = lambda course, lesson: f"https://example.com/lesson{lesson}"
        
        # Execute format_results directly
        result = tool._format_results(mock_search_results_success)
        
        # Assert
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["url"] == "https://example.com/lesson1"
        assert tool.last_sources[1]["url"] == "https://example.com/lesson2"
        
        # Verify get_lesson_link was called correctly
        tool.store.get_lesson_link.assert_any_call("Test Course: Introduction to Testing", 1)
        tool.store.get_lesson_link.assert_any_call("Test Course: Introduction to Testing", 2)
    
    def test_format_results_without_lesson_numbers(self, course_search_tool_with_mock):
        """Test formatting results that don't have lesson numbers"""
        # Setup
        tool = course_search_tool_with_mock
        results = SearchResults(
            documents=["Course overview content"],
            metadata=[{"course_title": "Test Course: Introduction to Testing"}],
            distances=[0.1]
        )
        
        # Execute
        result = tool._format_results(results)
        
        # Assert
        assert "[Test Course: Introduction to Testing]" in result
        assert "Course overview content" in result
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Test Course: Introduction to Testing"
        assert tool.last_sources[0]["url"] is None
    
    def test_format_results_unknown_course(self, course_search_tool_with_mock):
        """Test formatting results with unknown course title"""
        # Setup
        tool = course_search_tool_with_mock
        results = SearchResults(
            documents=["Some content"],
            metadata=[{"lesson_number": 1}], # Missing course_title
            distances=[0.1]
        )
        
        # Execute
        result = tool._format_results(results)
        
        # Assert
        assert "[unknown - Lesson 1]" in result
        assert "Some content" in result
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "unknown - Lesson 1"
    
    def test_get_tool_definition_structure(self, course_search_tool_with_mock):
        """Test that tool definition has correct structure"""
        # Setup
        tool = course_search_tool_with_mock
        
        # Execute
        definition = tool.get_tool_definition()
        
        # Assert
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"] 
        assert "lesson_number" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]
    
    def test_execute_exception_handling(self, course_search_tool_with_mock):
        """Test handling of unexpected exceptions during execution"""
        # Setup
        tool = course_search_tool_with_mock
        tool.store.search.side_effect = Exception("Unexpected error")
        
        # Execute
        result = tool.execute(query="test query")
        
        # Assert
        # With our new error handling, exceptions are caught and returned as error messages
        assert "Search failed:" in result
        assert "Unexpected error" in result
    
    def test_source_tracking_reset_between_calls(self, course_search_tool_with_mock, mock_search_results_success):
        """Test that sources are properly tracked and can be reset"""
        # Setup
        tool = course_search_tool_with_mock
        tool.store.search.return_value = mock_search_results_success
        
        # Execute first query
        result1 = tool.execute(query="first query")
        first_sources = tool.last_sources.copy()
        
        # Execute second query with empty results
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        tool.store.search.return_value = empty_results
        result2 = tool.execute(query="second query")
        
        # Assert
        assert len(first_sources) == 2  # First query had results
        assert len(tool.last_sources) == 0  # Second query was empty, sources reset
    
    @pytest.mark.parametrize("query,expected_called", [
        ("", True),  # Empty query should still call search
        ("   ", True),  # Whitespace query should still call search  
        ("test query", True),  # Normal query
        ("query with special chars !@#$%", True),  # Special characters
    ])
    def test_execute_with_various_query_inputs(self, course_search_tool_with_mock, mock_search_results_success, query, expected_called):
        """Test execute with various query input formats"""
        # Setup
        tool = course_search_tool_with_mock
        tool.store.search.return_value = mock_search_results_success
        
        # Execute
        result = tool.execute(query=query)
        
        # Assert
        if expected_called:
            tool.store.search.assert_called_once()
        else:
            tool.store.search.assert_not_called()