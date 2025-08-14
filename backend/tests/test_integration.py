#!/usr/bin/env python
"""Integration test to verify sequential tool calling works end-to-end"""

import sys
import os
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator, ToolCallState
from search_tools import ToolManager


def test_sequential_tool_execution():
    """Test that the system can handle sequential tool calls properly"""
    
    # Create a mock tool manager
    tool_manager = ToolManager()
    
    # Mock tool 1 - get course outline
    mock_outline_tool = Mock()
    mock_outline_tool.get_tool_definition.return_value = {
        "name": "get_course_outline",
        "description": "Get course outline",
        "input_schema": {
            "type": "object",
            "properties": {
                "course_title": {"type": "string"}
            },
            "required": ["course_title"]
        }
    }
    mock_outline_tool.execute.return_value = """
    Course Title: Python Basics
    Lessons:
      Lesson 1: Introduction
      Lesson 2: Variables and Types
      Lesson 3: Control Flow
      Lesson 4: Object-Oriented Programming
      Lesson 5: Advanced Topics
    """
    
    # Mock tool 2 - search content
    mock_search_tool = Mock()
    mock_search_tool.get_tool_definition.return_value = {
        "name": "search_course_content",
        "description": "Search course content",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    }
    mock_search_tool.execute.return_value = """
    Found 3 courses discussing Object-Oriented Programming:
    - Java Advanced: Full OOP implementation
    - C++ Fundamentals: Classes and inheritance
    - Ruby Design Patterns: OOP best practices
    """
    
    # Register mock tools
    tool_manager.tools = {
        "get_course_outline": mock_outline_tool,
        "search_course_content": mock_search_tool
    }
    
    # Create AI generator with mock API key
    generator = AIGenerator("test-key", "claude-3-opus-20240229")
    
    # Mock the Anthropic client
    mock_client = Mock()
    generator.client = mock_client
    
    # Mock response 1: Get outline
    mock_response1 = Mock()
    mock_response1.stop_reason = "tool_use"
    tool_call1 = Mock()
    tool_call1.type = "tool_use"
    tool_call1.name = "get_course_outline"
    tool_call1.input = {"course_title": "Python Basics"}
    tool_call1.id = "tool_1"
    mock_response1.content = [tool_call1]
    
    # Mock response 2: Search based on lesson 4
    mock_response2 = Mock()
    mock_response2.stop_reason = "tool_use"
    tool_call2 = Mock()
    tool_call2.type = "tool_use"
    tool_call2.name = "search_course_content"
    tool_call2.input = {"query": "Object-Oriented Programming"}
    tool_call2.id = "tool_2"
    mock_response2.content = [tool_call2]
    
    # Mock final response
    mock_final = Mock()
    mock_final.stop_reason = "end_turn"
    mock_final.content = [Mock(text="Based on lesson 4 of Python Basics (Object-Oriented Programming), I found similar content in Java Advanced, C++ Fundamentals, and Ruby Design Patterns courses.")]
    
    # Set up the mock sequence
    mock_client.messages.create.side_effect = [mock_response1, mock_response2, mock_final]
    
    # Test query that requires sequential tool calls
    query = "Find courses that cover similar topics to lesson 4 of Python Basics"
    
    # Execute
    result = generator.generate_response(
        query=query,
        tools=tool_manager.get_tool_definitions(),
        tool_manager=tool_manager,
        max_tool_rounds=2
    )
    
    # Verify the result
    print("Query:", query)
    print("Result:", result)
    
    # Verify both tools were called
    assert mock_outline_tool.execute.called, "Outline tool should have been called"
    assert mock_search_tool.execute.called, "Search tool should have been called"
    
    # Verify the sequence
    assert mock_outline_tool.execute.call_args[1]["course_title"] == "Python Basics"
    assert mock_search_tool.execute.call_args[1]["query"] == "Object-Oriented Programming"
    
    # Verify we made 3 API calls total
    assert mock_client.messages.create.call_count == 3, f"Expected 3 API calls, got {mock_client.messages.create.call_count}"
    
    print("\n[PASS] Integration test passed! Sequential tool calling is working correctly.")
    print(f"   - Made {mock_client.messages.create.call_count} API calls")
    print(f"   - Executed {mock_outline_tool.execute.call_count + mock_search_tool.execute.call_count} tool calls")
    
    return True


def test_single_tool_still_works():
    """Ensure backward compatibility - single tool calls still work"""
    
    # Create a mock tool manager
    tool_manager = ToolManager()
    
    # Mock a single search tool
    mock_search_tool = Mock()
    mock_search_tool.get_tool_definition.return_value = {
        "name": "search_course_content",
        "description": "Search course content",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    }
    mock_search_tool.execute.return_value = "Found 5 courses about Python"
    
    tool_manager.tools = {"search_course_content": mock_search_tool}
    
    # Create AI generator
    generator = AIGenerator("test-key", "claude-3-opus-20240229")
    mock_client = Mock()
    generator.client = mock_client
    
    # Mock single tool use then done
    mock_response1 = Mock()
    mock_response1.stop_reason = "tool_use"
    tool_call = Mock()
    tool_call.type = "tool_use"
    tool_call.name = "search_course_content"
    tool_call.input = {"query": "Python"}
    tool_call.id = "tool_1"
    mock_response1.content = [tool_call]
    
    mock_response2 = Mock()
    mock_response2.stop_reason = "end_turn"
    mock_response2.content = [Mock(text="I found 5 courses about Python.")]
    
    mock_client.messages.create.side_effect = [mock_response1, mock_response2]
    
    # Execute
    result = generator.generate_response(
        query="Search for Python courses",
        tools=tool_manager.get_tool_definitions(),
        tool_manager=tool_manager
    )
    
    print("\n[PASS] Backward compatibility test passed! Single tool calls still work.")
    print(f"   Result: {result}")
    
    return True


if __name__ == "__main__":
    print("Running integration tests for sequential tool calling...\n")
    
    try:
        test_sequential_tool_execution()
        test_single_tool_still_works()
        print("\n[SUCCESS] All integration tests passed!")
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)