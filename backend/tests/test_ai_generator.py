import unittest
from unittest.mock import Mock, MagicMock, patch, call
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator, ToolCallState


class TestToolCallState(unittest.TestCase):
    """Test the ToolCallState class"""
    
    def test_initial_state(self):
        """Test initial state of ToolCallState"""
        state = ToolCallState(max_rounds=2)
        self.assertEqual(state.max_rounds, 2)
        self.assertEqual(state.current_round, 0)
        self.assertTrue(state.can_make_more_calls())
        self.assertEqual(len(state.tool_calls_made), 0)
    
    def test_increment_round(self):
        """Test round incrementing"""
        state = ToolCallState(max_rounds=2)
        state.increment_round()
        self.assertEqual(state.current_round, 1)
        self.assertTrue(state.can_make_more_calls())
        
        state.increment_round()
        self.assertEqual(state.current_round, 2)
        self.assertFalse(state.can_make_more_calls())
    
    def test_add_tool_call(self):
        """Test adding tool call records"""
        state = ToolCallState()
        state.add_tool_call("search_tool", {"query": "test"}, "result text")
        
        self.assertEqual(len(state.tool_calls_made), 1)
        self.assertEqual(state.tool_calls_made[0]['tool'], "search_tool")
        self.assertEqual(state.tool_calls_made[0]['params'], {"query": "test"})
        self.assertEqual(state.tool_calls_made[0]['result_length'], 11)


class TestAIGenerator(unittest.TestCase):
    """Test the AIGenerator class with sequential tool calling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.api_key = "test-api-key"
        self.model = "claude-3-opus-20240229"
        
        # Create generator with mocked client
        with patch('ai_generator.anthropic.Anthropic'):
            self.generator = AIGenerator(self.api_key, self.model)
            self.mock_client = Mock()
            self.generator.client = self.mock_client
    
    def test_no_tools_needed(self):
        """Test direct response when no tools are needed"""
        # Mock response without tool use
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Direct answer to question")]
        
        self.mock_client.messages.create.return_value = mock_response
        
        # Call generate_response
        result = self.generator.generate_response(
            query="What is Python?",
            tools=None,
            tool_manager=None
        )
        
        # Verify result
        self.assertEqual(result, "Direct answer to question")
        
        # Verify only one API call was made
        self.assertEqual(self.mock_client.messages.create.call_count, 1)
    
    def test_single_tool_call(self):
        """Test backward compatibility with single tool call"""
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result content"
        
        # Mock tools
        mock_tools = [{"name": "search_tool", "description": "Search tool"}]
        
        # Mock first response with tool use
        mock_response1 = Mock()
        mock_response1.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_tool"
        mock_tool_block.input = {"query": "test query"}
        mock_tool_block.id = "tool_123"
        mock_response1.content = [mock_tool_block]
        
        # Mock second response after tool execution (no more tools needed)
        mock_response2 = Mock()
        mock_response2.stop_reason = "end_turn"
        mock_response2.content = [Mock(text="Final answer based on tool results")]
        
        # Mock third response (would be final synthesis if Claude used another tool)
        mock_response3 = Mock()
        mock_response3.stop_reason = "end_turn"
        mock_response3.content = [Mock(text="Should not reach here")]
        
        # Set up mock to return different responses
        self.mock_client.messages.create.side_effect = [mock_response1, mock_response2, mock_response3]
        
        # Call generate_response
        result = self.generator.generate_response(
            query="Search for Python tutorials",
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify result
        self.assertEqual(result, "Final answer based on tool results")
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_tool", query="test query"
        )
        
        # Verify only two API calls were made (not three)
        self.assertEqual(self.mock_client.messages.create.call_count, 2)
        
        # Verify second call DOES include tools (can still make another tool call)
        second_call_args = self.mock_client.messages.create.call_args_list[1][1]
        self.assertIn("tools", second_call_args)
        
        # Verify Claude chose not to use more tools (stop_reason was end_turn)
    
    def test_sequential_tool_calls(self):
        """Test sequential tool calling with two rounds"""
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "First tool result",
            "Second tool result"
        ]
        
        # Mock tools
        mock_tools = [
            {"name": "get_outline", "description": "Get course outline"},
            {"name": "search_content", "description": "Search content"}
        ]
        
        # Mock first response - first tool call
        mock_response1 = Mock()
        mock_response1.stop_reason = "tool_use"
        mock_tool1 = Mock()
        mock_tool1.type = "tool_use"
        mock_tool1.name = "get_outline"
        mock_tool1.input = {"course": "Python"}
        mock_tool1.id = "tool_1"
        mock_response1.content = [mock_tool1]
        
        # Mock second response - second tool call based on first results
        mock_response2 = Mock()
        mock_response2.stop_reason = "tool_use"
        mock_tool2 = Mock()
        mock_tool2.type = "tool_use"
        mock_tool2.name = "search_content"
        mock_tool2.input = {"query": "lesson 4 topic"}
        mock_tool2.id = "tool_2"
        mock_response2.content = [mock_tool2]
        
        # Mock final response after all tools
        mock_response3 = Mock()
        mock_response3.stop_reason = "end_turn"
        mock_response3.content = [Mock(text="Final comprehensive answer")]
        
        # Set up mock to return different responses
        self.mock_client.messages.create.side_effect = [
            mock_response1, mock_response2, mock_response3
        ]
        
        # Call generate_response
        result = self.generator.generate_response(
            query="Find courses similar to lesson 4 of Python course",
            tools=mock_tools,
            tool_manager=mock_tool_manager,
            max_tool_rounds=2
        )
        
        # Verify result
        self.assertEqual(result, "Final comprehensive answer")
        
        # Verify both tools were executed in sequence
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        mock_tool_manager.execute_tool.assert_any_call("get_outline", course="Python")
        mock_tool_manager.execute_tool.assert_any_call("search_content", query="lesson 4 topic")
        
        # Verify three API calls were made
        self.assertEqual(self.mock_client.messages.create.call_count, 3)
        
        # Verify first two calls included tools
        first_call_args = self.mock_client.messages.create.call_args_list[0][1]
        self.assertIn("tools", first_call_args)
        
        second_call_args = self.mock_client.messages.create.call_args_list[1][1]
        self.assertIn("tools", second_call_args)
        
        # Verify final call did NOT include tools
        third_call_args = self.mock_client.messages.create.call_args_list[2][1]
        self.assertNotIn("tools", third_call_args)
    
    def test_max_rounds_reached(self):
        """Test that tool calling stops after max rounds"""
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        # Mock tools
        mock_tools = [{"name": "search_tool", "description": "Search tool"}]
        
        # Create responses that always want to use tools
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool = Mock()
        mock_tool.type = "tool_use"
        mock_tool.name = "search_tool"
        mock_tool.input = {"query": "test"}
        mock_tool.id = "tool_id"
        mock_tool_response.content = [mock_tool]
        
        # Final response without tools
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [Mock(text="Final answer after max rounds")]
        
        # Set up mock to return tool responses then final
        self.mock_client.messages.create.side_effect = [
            mock_tool_response,  # Round 1
            mock_tool_response,  # Round 2
            mock_final_response  # Final synthesis
        ]
        
        # Call with max_tool_rounds=2
        result = self.generator.generate_response(
            query="Complex query",
            tools=mock_tools,
            tool_manager=mock_tool_manager,
            max_tool_rounds=2
        )
        
        # Verify result
        self.assertEqual(result, "Final answer after max rounds")
        
        # Verify exactly 2 tool executions
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        
        # Verify exactly 3 API calls (2 with tools, 1 without)
        self.assertEqual(self.mock_client.messages.create.call_count, 3)
    
    def test_message_accumulation(self):
        """Test that messages accumulate correctly across rounds"""
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        # Mock tools
        mock_tools = [{"name": "test_tool", "description": "Test tool"}]
        
        # Mock responses
        mock_response1 = Mock()
        mock_response1.stop_reason = "tool_use"
        mock_tool = Mock()
        mock_tool.type = "tool_use"
        mock_tool.name = "test_tool"
        mock_tool.input = {}
        mock_tool.id = "tool_1"
        mock_response1.content = [mock_tool]
        
        mock_response2 = Mock()
        mock_response2.stop_reason = "end_turn"
        mock_response2.content = [Mock(text="Final")]
        
        self.mock_client.messages.create.side_effect = [mock_response1, mock_response2]
        
        # Call generate_response
        result = self.generator.generate_response(
            query="Test query",
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        # Check that second API call has accumulated messages
        second_call_args = self.mock_client.messages.create.call_args_list[1][1]
        messages = second_call_args["messages"]
        
        # Should have: user query, assistant tool call, user tool result
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[2]["role"], "user")
        
        # Verify tool result is in the messages
        self.assertIsInstance(messages[2]["content"], list)
        self.assertEqual(messages[2]["content"][0]["type"], "tool_result")
    
    def test_error_handling_no_tools_executed(self):
        """Test handling when tool execution fails"""
        # Mock tool manager that raises an exception
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool error")
        
        # Mock tools
        mock_tools = [{"name": "failing_tool", "description": "Failing tool"}]
        
        # Mock response with tool use
        mock_response1 = Mock()
        mock_response1.stop_reason = "tool_use"
        mock_tool = Mock()
        mock_tool.type = "tool_use"
        mock_tool.name = "failing_tool"
        mock_tool.input = {}
        mock_tool.id = "tool_1"
        mock_response1.content = [mock_tool]
        
        # Mock final response
        mock_response2 = Mock()
        mock_response2.stop_reason = "end_turn"
        mock_response2.content = [Mock(text="Fallback response")]
        
        self.mock_client.messages.create.side_effect = [mock_response1, mock_response2]
        
        # Call should handle the error gracefully
        with self.assertRaises(Exception):
            result = self.generator.generate_response(
                query="Test query",
                tools=mock_tools,
                tool_manager=mock_tool_manager
            )
    
    def test_conversation_history_preserved(self):
        """Test that conversation history is included in all API calls"""
        history = "User: Previous question\nAssistant: Previous answer"
        
        # Mock response without tools
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Answer")]
        
        self.mock_client.messages.create.return_value = mock_response
        
        # Call with conversation history
        result = self.generator.generate_response(
            query="New question",
            conversation_history=history
        )
        
        # Verify system prompt includes history
        call_args = self.mock_client.messages.create.call_args[1]
        self.assertIn(history, call_args["system"])


class TestComplexScenarios(unittest.TestCase):
    """Test complex real-world scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('ai_generator.anthropic.Anthropic'):
            self.generator = AIGenerator("test-key", "test-model")
            self.mock_client = Mock()
            self.generator.client = self.mock_client
    
    def test_course_comparison_scenario(self):
        """Test comparing two courses using sequential tool calls"""
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "MCP Course Outline: Lesson 1: Intro, Lesson 2: Basics...",
            "Computer Use Course Outline: Lesson 1: Setup, Lesson 2: Navigation..."
        ]
        
        # Mock tools
        mock_tools = [{"name": "get_course_outline", "description": "Get course outline"}]
        
        # Mock responses for two sequential outline calls
        mock_response1 = Mock()
        mock_response1.stop_reason = "tool_use"
        tool1 = Mock()
        tool1.type = "tool_use"
        tool1.name = "get_course_outline"
        tool1.input = {"course_title": "MCP"}
        tool1.id = "tool_1"
        mock_response1.content = [tool1]
        
        mock_response2 = Mock()
        mock_response2.stop_reason = "tool_use"
        tool2 = Mock()
        tool2.type = "tool_use"
        tool2.name = "get_course_outline"
        tool2.input = {"course_title": "Computer Use"}
        tool2.id = "tool_2"
        mock_response2.content = [tool2]
        
        mock_final = Mock()
        mock_final.stop_reason = "end_turn"
        mock_final.content = [Mock(text="Both courses cover similar introductory topics...")]
        
        self.mock_client.messages.create.side_effect = [
            mock_response1, mock_response2, mock_final
        ]
        
        # Execute the comparison
        result = self.generator.generate_response(
            query="How does the MCP introduction compare to the Computer Use course structure?",
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify both outlines were retrieved
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        
        # Verify final synthesis
        self.assertIn("similar introductory topics", result)
    
    def test_find_specific_then_search_scenario(self):
        """Test finding specific lesson then searching for related content"""
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course: Python Basics\nLesson 4: Object-Oriented Programming",
            "Found 3 courses discussing OOP: Java Advanced, C++ Fundamentals, Ruby Design"
        ]
        
        # Mock tools
        mock_tools = [
            {"name": "get_course_outline", "description": "Get outline"},
            {"name": "search_course_content", "description": "Search content"}
        ]
        
        # First call gets outline
        mock_response1 = Mock()
        mock_response1.stop_reason = "tool_use"
        tool1 = Mock()
        tool1.type = "tool_use"
        tool1.name = "get_course_outline"
        tool1.input = {"course_title": "Python Basics"}
        tool1.id = "tool_1"
        mock_response1.content = [tool1]
        
        # Second call searches based on lesson 4 topic
        mock_response2 = Mock()
        mock_response2.stop_reason = "tool_use"
        tool2 = Mock()
        tool2.type = "tool_use"
        tool2.name = "search_course_content"
        tool2.input = {"query": "Object-Oriented Programming"}
        tool2.id = "tool_2"
        mock_response2.content = [tool2]
        
        # Final synthesis
        mock_final = Mock()
        mock_final.stop_reason = "end_turn"
        mock_final.content = [Mock(text="Courses covering similar OOP topics: Java Advanced, C++ Fundamentals, Ruby Design")]
        
        self.mock_client.messages.create.side_effect = [
            mock_response1, mock_response2, mock_final
        ]
        
        # Execute the complex query
        result = self.generator.generate_response(
            query="Find courses that discuss the same topic as lesson 4 of Python Basics",
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify the sequence
        calls = mock_tool_manager.execute_tool.call_args_list
        self.assertEqual(calls[0][0][0], "get_course_outline")
        self.assertEqual(calls[1][0][0], "search_course_content")
        
        # Verify final result mentions the found courses
        self.assertIn("Java Advanced", result)


if __name__ == "__main__":
    unittest.main()