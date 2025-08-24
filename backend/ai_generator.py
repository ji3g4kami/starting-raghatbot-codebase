from typing import Any

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Tool Usage Guidelines:
- **Course content questions**: Use search_course_content tool for specific lesson materials or detailed educational content
- **Course outline/structure questions**: Use get_course_outline tool for course titles, links, lesson lists, and course structure information
- **Sequential tool usage**: You can make up to 2 sequential tool calls if needed to fully answer the user's question
- **Efficiency**: Only use tools when necessary - don't make unnecessary tool calls
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course-specific questions**: Use appropriate tool(s) first, then answer
- **Course outline requests**: Always use get_course_outline tool to provide complete course title, link, and lesson information
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        # Validate API key is provided
        if not api_key or api_key.strip() == "":
            raise ValueError(
                "Anthropic API key is required but not provided. Please set ANTHROPIC_API_KEY in your .env file."
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: str | None = None,
        tools: list | None = None,
        tool_manager=None,
        max_tool_rounds: int = 2,
    ) -> str:
        """
        Generate AI response with support for sequential tool usage (up to max_tool_rounds).

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_tool_rounds: Maximum number of sequential tool calling rounds (default: 2)

        Returns:
            Generated response as string
        """

        # Build initial system content and messages
        system_content = self._build_system_content(conversation_history)
        messages = [{"role": "user", "content": query}]

        # Main sequential tool execution loop
        for _round_num in range(max_tool_rounds):
            # Prepare API parameters with tools for each round
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content,
            }

            # Add tools if available
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            # Get response from Claude
            try:
                response = self.client.messages.create(**api_params)
            except Exception as api_error:
                # API errors terminate immediately
                raise api_error

            # Check termination conditions
            if response.stop_reason != "tool_use":
                # No tool use - return response
                return response.content[0].text

            if not tool_manager:
                # No tool manager available - return response
                return response.content[0].text

            # Handle tool execution for this round
            try:
                messages = self._execute_tools_and_update_messages(
                    response, messages, tool_manager
                )
            except Exception as tool_error:
                # Tool execution failed - return error response
                return f"I encountered an error while searching: {str(tool_error)}"

        # Maximum rounds reached - make final call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }

        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text

    def _build_system_content(self, conversation_history: str | None) -> str:
        """
        Build system content with conversation history.

        Args:
            conversation_history: Previous conversation context

        Returns:
            Complete system content string
        """
        return (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

    def _execute_tools_and_update_messages(
        self, response, messages: list, tool_manager
    ) -> list:
        """
        Execute tools and update message history for next round.

        Args:
            response: The API response containing tool use requests
            messages: Current message history
            tool_manager: Manager to execute tools

        Returns:
            Updated messages list with tool results
        """
        # Copy messages to avoid mutations
        updated_messages = messages.copy()

        # Add AI's response to messages
        updated_messages.append({"role": "assistant", "content": response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, **content_block.input
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result,
                    }
                )

        # Add tool results to messages
        if tool_results:
            updated_messages.append({"role": "user", "content": tool_results})

        return updated_messages

    def _handle_tool_execution(
        self, initial_response, base_params: dict[str, Any], tool_manager
    ):
        """
        DEPRECATED: Legacy method for single-round tool execution.
        Maintained for backward compatibility. Use generate_response() instead.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Convert to new format and execute single round
        messages = base_params["messages"].copy()
        messages = self._execute_tools_and_update_messages(
            initial_response, messages, tool_manager
        )

        # Make final call without tools (legacy behavior)
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
        }

        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text
