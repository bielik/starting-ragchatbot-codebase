import anthropic
from typing import List, Optional, Dict, Any, Tuple

class ToolCallState:
    """Tracks the state of tool calls across multiple rounds"""
    
    def __init__(self, max_rounds: int = 2):
        self.max_rounds = max_rounds
        self.current_round = 0
        self.tool_calls_made = []
    
    def can_make_more_calls(self) -> bool:
        """Check if more tool calls can be made"""
        return self.current_round < self.max_rounds
    
    def increment_round(self):
        """Increment the current round counter"""
        self.current_round += 1
    
    def add_tool_call(self, tool_name: str, params: Dict[str, Any], result: str):
        """Record a tool call that was made"""
        self.tool_calls_made.append({
            'round': self.current_round,
            'tool': tool_name,
            'params': params,
            'result_length': len(result) if result else 0
        })

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Available Tools:
1. **get_course_outline**: Retrieves complete course structure with title, link, and all lessons
   - Use for: course outlines, syllabus queries, lesson lists, course structure questions
   - Returns: Course title, course link, and numbered lesson list

2. **search_course_content**: Searches within course materials for specific content
   - Use for: detailed content questions, specific topics, lesson details
   - Returns: Relevant content excerpts from course materials

Tool Usage Guidelines:
- **Course outline/structure questions**: Use get_course_outline tool
- **Specific content questions**: Use search_course_content tool
- **Sequential tool usage**: You may use tools up to 2 times in sequence to gather comprehensive information
- **First tool call**: Use to get initial information (e.g., course outline, basic search)
- **Second tool call**: Use to refine search based on first results or explore related topics
- **Complex queries**: Break down multi-part questions using sequential tool calls
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Examples of multi-step tool usage:
- To find courses with similar topics to a specific lesson: First get the course outline to identify the lesson, then search for that topic
- To compare course structures: Get outline of first course, then get outline of second course
- To find detailed content after overview: First search broadly, then search for specific details

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Course-specific questions**: Use appropriate tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the outline tool"

When presenting course outlines:
- Display the course title prominently
- Include the course link if available
- List all lessons with their numbers and titles
- Keep formatting clean and readable

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         max_tool_rounds: int = 2) -> str:
        """
        Generate AI response with optional sequential tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_tool_rounds: Maximum number of sequential tool calls allowed
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Initialize message history and tool state
        messages = [{"role": "user", "content": query}]
        tool_state = ToolCallState(max_rounds=max_tool_rounds)
        
        # Process tool calls iteratively
        while tool_state.can_make_more_calls():
            # Prepare API call parameters with tools available
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content
            }
            
            # Add tools if available and we can still make tool calls
            if tools and tool_manager:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}
            
            # Get response from Claude
            response = self.client.messages.create(**api_params)
            
            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use":
                # Execute tools and update messages
                messages, tools_executed = self._execute_tool_round(
                    response, messages, tool_manager, tool_state
                )
                
                # If no tools were executed (error case), break
                if not tools_executed:
                    break
                    
                # Increment round counter
                tool_state.increment_round()
            else:
                # Claude doesn't want to use tools, return the response
                return response.content[0].text
        
        # Max rounds reached or no more tool calls needed - make final call without tools
        return self._make_final_response(messages, system_content)
    
    def _execute_tool_round(self, response, messages: List[Dict], tool_manager, tool_state: ToolCallState) -> Tuple[List[Dict], bool]:
        """
        Execute a round of tool calls and update message history.
        
        Args:
            response: The response containing tool use requests
            messages: Current message history
            tool_manager: Manager to execute tools
            tool_state: State tracker for tool calls
            
        Returns:
            Tuple of (updated messages, whether any tools were executed)
        """
        # Create a copy of messages to avoid mutation
        updated_messages = messages.copy()
        
        # Add AI's tool use response
        updated_messages.append({"role": "assistant", "content": response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        tools_executed = False
        
        for content_block in response.content:
            if content_block.type == "tool_use":
                # Execute the tool
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                # Track the tool call
                tool_state.add_tool_call(
                    content_block.name,
                    content_block.input,
                    tool_result
                )
                
                # Add to results
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
                
                tools_executed = True
        
        # Add tool results as single message
        if tool_results:
            updated_messages.append({"role": "user", "content": tool_results})
        
        return updated_messages, tools_executed
    
    def _make_final_response(self, messages: List[Dict], system_content: str) -> str:
        """
        Make a final API call without tools to generate the synthesis response.
        
        Args:
            messages: Complete message history including tool results
            system_content: System prompt content
            
        Returns:
            Final response text
        """
        # Prepare final API call WITHOUT tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text