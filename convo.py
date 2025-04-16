import os
import re
import logging
import json
import requests
import google.generativeai as genai
from textwrap import dedent
from datetime import datetime, timedelta
import agno
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.sql import SQLTools
import psycopg2
from psycopg2 import sql
from analysis import analyze_message, classify_message_type, classify_booking_message
from memory import store_memory, get_user_preferences, extract_and_store_preferences, get_conversation_history, get_ranked_conversation_history, select_relevant_history_with_agno
from booking import handle_booking_with_agno, handle_booking_query_with_agno
from agent_init import get_agno_agent_with_retry, get_agno_agent, rotate_gemini_key, rotate_gemini_key_and_model
from config import MEMORY_TYPES, serper_api_key, gemini_api_keys
from date_filter import extract_entities_from_text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Process user message
def process_message(user_id, message):
    """
    Process a user message and generate a response

    Args:
        user_id (str): User ID
        message (str): User message

    Returns:
        str: Response message
    """
    try:
        # First, classify the message type
        message_type = classify_message_type(message)
        logger.info(f"Message classified as: {message_type}")
        
        # For context-dependent messages, use the user's conversation history
        if message_type == "context_dependent":
            logger.info("Handling as context-dependent query with conversation history")
            return handle_context_dependent_query(user_id, message)
        
        # For standard flow messages, proceed with the existing logic
        # Analyze the message for intents and entities
        analysis = analyze_message(message)
        intents = analysis.get("intents", [])

        # Check if this is a booking-related intent
        if "booking" in intents or "booking_query" in intents or any(word in message.lower() for word in ["book", "appointment", "schedule", "reserve", "my bookings"]):
            # Use Gemini with Agno to classify the booking message type
            booking_type = classify_booking_message(message)
            logger.info(f"Booking message classified as: {booking_type}")
            
            # Extract entities
            service_entities = analysis.get("service_entities", [])
            date_entities = analysis.get("date_entities", [])
            time_entities = analysis.get("time_entities", [])
            
            if booking_type == "booking_retrieval":
                # Determine the type of booking query
                message_lower = message.lower()
                has_previous_indicators = "previous_bookings" in intents or any(word in message_lower for word in ["previous", "past", "earlier", "history"])
                has_future_indicators = "future_bookings" in intents or any(word in message_lower for word in ["future", "upcoming", "next", "scheduled", "coming"])
                
                # Check for explicit "previous" or "past" indicators
                if has_previous_indicators:
                    query_type = "previous"
                    logger.info("Query classified as 'previous' due to explicit previous/past indicators")
                # Check for words indicating future/upcoming bookings
                elif has_future_indicators:
                    query_type = "future"
                    logger.info("Query classified as 'future' due to explicit future/upcoming indicators")
                # If "before" is used with a date but without past-tense indicators, assume future bookings
                elif "before" in message_lower and not has_previous_indicators:
                    query_type = "future"
                    logger.info("Query classified as 'future' due to 'before' without past indicators")
                # If "after" is used with "previous" or past-tense indicators, ensure it's treated as previous bookings
                elif "after" in message_lower and has_previous_indicators:
                    query_type = "previous"
                    logger.info("Query classified as 'previous' due to 'after' with past indicators")
                else:
                    query_type = "all"
                    logger.info("Query classified as 'all' due to no specific time indicators")
                
                # Use the user_id as the customer name to filter bookings
                # This ensures users can only see their own bookings
                customer_name = user_id
                
                # Handle the booking query with Agno
                return handle_booking_query_with_agno(user_id, message, query_type, customer_name)
            else:  # booking_type == "create_booking"
                # Handle the booking creation with Agno
                return handle_booking_with_agno(user_id, message, intents, service_entities, date_entities, time_entities)
        
        # For other intents, use Agno with SQLTools and web search
        else:
            return handle_general_query_with_agno(user_id, message)

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return "I'm sorry, but I encountered an error processing your request. Please try again or contact our staff for assistance."



# Handle context-dependent queries with Agno using user's conversation history
def handle_context_dependent_query(user_id, message):
    """
    Handle context-dependent queries using Agno agent with the user's conversation history
    
    Args:
        user_id (str): User ID
        message (str): User message
        
    Returns:
        str: Response message from Agno agent
    """
    # Use the new smart context selection approach
    return answer_with_smart_context(user_id, message)

# Answer questions using context-aware approach with Gemini and Agno
def answer_with_smart_context(user_id, message, custom_instructions=None, additional_tools=None, model_name=None):
    """
    Answer questions using smart context selection with Gemini and Agno
    
    Args:
        user_id (str): User ID
        message (str): User message/question
        custom_instructions (str): Optional custom instructions for the agent
        additional_tools (list): Optional additional tools for the agent
        model_name (str): Optional specific Gemini model to use
        
    Returns:
        str: Response message from Agno agent
    """
    try:
        # Create database connection string for SQL tools
        db_url = f"postgresql+psycopg://{os.getenv('user')}:{os.getenv('password')}@{os.getenv('host')}:{os.getenv('port')}/{os.getenv('dbname')}"
        
        # Create SQL tools for Agno
        sql_tools = SQLTools(db_url=db_url)
        
        # Get the user's conversation history using adaptive sliding window
        history_items = get_ranked_conversation_history(
            user_id, 
            limit=15, 
            format_as_text=False,
            current_message=message  # Pass the current message for relevance ranking
        )
        
        # If no history, handle as a new conversation
        if not history_items:
            logger.info("No conversation history found, handling as new conversation")
            return handle_general_query_with_agno(user_id, message)
        
        # Check if this is a short follow-up question
        is_short_followup = len(message.split()) <= 5
        
        # For very short messages with pronouns, they're almost certainly follow-ups
        has_pronouns = any(pronoun in message.lower() for pronoun in ['it', 'this', 'that', 'they', 'them', 'these', 'those'])
        has_after_words = any(word in message.lower() for word in ['after', 'before', 'during', 'following', 'between'])
        is_definite_followup = is_short_followup and (has_pronouns or has_after_words)
        
        # Special handling for "can I do it after X" type queries
        message_lower = message.lower()
        is_after_before_query = (
            ('after' in message_lower or 'before' in message_lower) and 
            ('can' in message_lower or 'could' in message_lower or 'possible' in message_lower) and
            has_pronouns
        )
        
        if is_after_before_query:
            logger.info(f"Detected special 'after/before' query with pronouns: '{message}'")
            # This is a special case that needs explicit handling
        
        # Use Gemini to select the most relevant history items
        # For short follow-ups, we'll get more context items to ensure we don't miss anything
        max_items = 7 if is_short_followup else 5
        
        # Get relevant history items using Agno for timestamp and conversation flow analysis
        relevant_items = select_relevant_history_with_agno(user_id, message, history_items, max_items=max_items)
        
        # For definite follow-ups, log that we detected it and increase the number of context items
        if is_definite_followup and history_items:
            logger.info(f"Detected definite follow-up question with pronouns or time relations: '{message}'")
            # For definite follow-ups, get more context items to ensure we don't miss anything
            max_items = 10  # Increase from 7 to 10 for definite follow-ups
            # Get relevant history items again with more context
            relevant_items = select_relevant_history_with_agno(user_id, message, history_items, max_items=max_items)
            logger.info(f"Increased context to {len(relevant_items)} items for definite follow-up")
        
        # Extract potential entities from the conversation history
        entities = extract_entities_from_history(relevant_items)
        
        # Format the selected history items as context
        context_history = ""
        for item in relevant_items:
            metadata = item.get('metadata', {})
            user_msg = metadata.get('message', '')
            bot_msg = metadata.get('response', '')
            
            if user_msg and bot_msg:
                context_history += f"User: {user_msg}\nAssistant: {bot_msg}\n\n"
        
        logger.info(f"Using {len(relevant_items)} relevant conversation history items as context")
        
        # Create tools list
        tools = [sql_tools]
        if additional_tools:
            tools.extend(additional_tools)
            
        # Add web search capability if Serper API key is available
        web_search_instructions = ""
        if serper_api_key:
            web_search_instructions = dedent("""
                HANDLING MISSING INFORMATION:
                1. If information is not found in the database, EXPLICITLY state "I need to search for: [search query]" 
                2. Alternatively, you can indicate that information is not in the database by saying phrases like:
                   - "I don't have specific information about that in our database"
                   - "That information is not available in our database"
                   - "I couldn't find information about that in our system"
                3. When you use these phrases, I will automatically search the web for the information
                4. DO NOT make up information if it's not in the database - either request a search or indicate the information is missing
            """)
        
        # Create default context-dependent query instructions if not provided
        # Add special instructions for after/before queries
        special_after_before_instructions = ""
        if is_after_before_query and history_items:
            # Get the most recent conversation
            most_recent = history_items[0]
            most_recent_user_msg = most_recent.get('metadata', {}).get('message', '')
            most_recent_bot_msg = most_recent.get('metadata', {}).get('response', '')
            
            special_after_before_instructions = dedent(f"""
                SPECIAL INSTRUCTIONS FOR THIS QUERY:
                The user has asked: "{message}"
                
                This is a special "after/before" query with pronouns. In this type of query:
                1. The pronoun "it" ALWAYS refers to a service or treatment mentioned in the most recent conversation
                2. The user is asking if they can get that service after/before another treatment
                
                Most recent conversation:
                User: {most_recent_user_msg}
                Assistant: {most_recent_bot_msg}
                
                You MUST start your response by explicitly identifying what "it" refers to from this conversation.
                Example: "Regarding the [specific service] we discussed earlier, ..."
            """)
            
            logger.info(f"Added special instructions for after/before query")
        
        default_instructions = dedent(f"""
            You are an assistant for Red Trends Spa & Wellness Center.
            Your task is to help users with their questions, maintaining context from previous conversations.
            
            {special_after_before_instructions}
            
            IMPORTANT INSTRUCTIONS:
            1. The conversation history provided has been SPECIFICALLY SELECTED using advanced analysis of:
               - Timestamps (temporal relevance)
               - Conversation flows (topic continuity)
               - Semantic relevance to the current question
               - Reference resolution needs (pronouns, implicit references)
            
            2. When the user refers to something mentioned earlier, use the history to determine what they're referring to
            3. ALWAYS check the database for information about spa services, prices, etc.
            4. For service-related questions, query the services table
            5. For booking-related questions, direct the user to make a booking
            6. Be conversational and maintain continuity with previous interactions
            
            {web_search_instructions}
            
            CONTEXT HANDLING GUIDELINES:
            1. For short messages like "can i do it after detan" or "what about this one", you MUST explicitly identify what "it" or "this one" refers to
            2. If the user mentions a pronoun (it, this, that, they, etc.), you MUST determine what it refers to from the conversation history
               and EXPLICITLY state this in your response: "I understand that by 'it' you're referring to [specific service]..."
            3. If the user's message is very short (1-5 words), it's likely a follow-up to a previous question
            4. When in doubt about what a user is referring to, use the most recently discussed service or topic
            5. IMPORTANT: For follow-up questions, ALWAYS check the MOST RECENT conversation first
            6. CRITICAL: When a user asks about doing something "after X" or "before X" (e.g., "can I do it after microneedle therapy"), 
               they are asking if they can get a service (mentioned in previous messages) after getting the treatment X.
               In this case, "it" refers to the previously discussed service, and you should explain if it's advisable to get that service after X.
            7. Pay attention to the TIMESTAMPS of messages to identify conversation sessions and topic continuity
            8. For questions about service compatibility (e.g., "can I do X after Y?"), check the database for both services and provide advice
            9. NEVER respond to a question containing pronouns without first clarifying what those pronouns refer to
            10. If you cannot confidently determine what a pronoun refers to, ASK THE USER to clarify
            
            CONVERSATION FLOW AWARENESS:
            1. The selected history represents the most coherent conversation flow related to the current message
            2. Messages close in time are likely part of the same conversation thread
            3. When a user refers to "this" or "it", they're almost always referring to something in the most recent exchange
            4. If the conversation history shows a clear topic progression, maintain that flow in your response
            
            SERVICE COMPATIBILITY GUIDELINES:
            1. For questions like "can I do it after microneedle therapy?", the user is asking about the compatibility
               of a previously discussed service with microneedle therapy
            2. CRITICAL: When answering such questions:
               - FIRST, explicitly identify what "it" refers to from the conversation history (usually a spa service)
               - ALWAYS start your response by clarifying what "it" refers to: "Regarding [specific service] you asked about earlier..."
               - If you're unsure what "it" refers to, ASK THE USER to clarify which service they're referring to
               - NEVER respond without first identifying what "it" refers to
               - Then provide specific advice about whether that service is compatible after/before the mentioned treatment
               - Include any recommended waiting periods between treatments
               - Explain the reasoning behind your recommendations (skin sensitivity, healing time, etc.)
            
            ENTITIES MENTIONED IN CONVERSATION HISTORY:
            {entities}
            
            DATABASE SCHEMA (EXACT NAMES - DO NOT MODIFY):
            - Table name: public.services
              Columns: "Service ID", "Service Name", "Description", "Price (INR)", "Category"
              Note: Column names include spaces and must be quoted in SQL queries
            
            - Table name: public.bookings
              Columns: "Booking ID", "Customer Name", "Service ID", "Booking Date", "Time Slot (HH:MM)", "Price (INR)"
              Note: The "Customer Name" column contains the user ID of the customer
            
            EXAMPLE SQL QUERIES:
            - To find information about a specific service:
              ```sql
              SELECT * FROM public.services WHERE "Service Name" ILIKE '%facial%'
              ```
            
            - To get all services in a category:
              ```sql
              SELECT * FROM public.services WHERE "Category" ILIKE '%massage%'
              ```
            
            RESPONSE GUIDELINES:
            1. For service questions, provide details about the service, price, and description
            2. For general spa questions, provide helpful information based on the database
            3. Always be professional, friendly, and concise
            4. If suggesting a service, mention how the user can book it
            5. Maintain continuity with previous conversations
            6. CRITICAL: For follow-up questions, ALWAYS start by explicitly acknowledging what the user is referring to
               Example: "Regarding the Deep Tissue Massage you asked about earlier, yes it is available..."
            7. For questions about service compatibility (e.g., "can I do X after Y?"):
               - ALWAYS begin by clearly identifying which services the user is asking about
               - Start your response with: "I understand you're asking if you can get [specific service] after [treatment]..."
               - Provide specific advice about the timing and compatibility of these services
               - Example: "Regarding the Deep Tissue Massage you asked about earlier, it's generally recommended to wait at least 
                 24-48 hours after microneedle therapy before getting this massage. This allows your skin to recover..."
            8. For questions containing pronouns like "it", "this", or "that":
               - ALWAYS begin your response by clarifying what the pronoun refers to
               - Format: "I understand that by 'it' you're referring to [specific service/topic]..."
               - If you cannot determine what the pronoun refers to, ask the user to clarify
               - NEVER proceed with your response without first resolving pronoun references
            9. When listing all services or multiple services, format them in a clean markdown format:
               - Use underline for categories (with markdown __underline__)
               - Use bold for service names and ### for categories
               - Use regular text for price and description
               - Example:
                 ```
                 ### __Massage Therapy__
                 
                 **Deep Tissue Massage**
                 Price: INR 2500
                 Description: A therapeutic massage targeting deeper muscle layers
                 
                 **Aromatherapy Massage**
                 Price: INR 2200
                 Description: Relaxing massage with essential oils
                 
                 **Facial Treatments**
                 ---
                 
                 **Glowing Facial**
                 Price: INR 1800
                 Description: A customized treatment for radiant skin
                 ```
            
            MOST RELEVANT CONVERSATION HISTORY:
            {context_history}
        """)
        
        # Use custom instructions if provided, otherwise use default
        context_query_instructions = custom_instructions if custom_instructions else default_instructions
        
        # Create the Agno agent with key rotation on failure
        try:
            context_agent = get_agno_agent_with_retry(
                instructions=context_query_instructions, 
                tools=tools,
                model_name=model_name
            )
        except Exception as e:
            logger.error(f"Failed to create context-dependent agent after retries: {str(e)}")
            return "I'm sorry, but I'm having trouble accessing our system right now. Could you please try again with a more specific question?"
        
        # Prepare the prompt for the context agent
        # For short follow-ups, explicitly ask the agent to identify what the user is referring to
        if is_short_followup:
            # Even more explicit instructions for definite follow-ups with pronouns
            if is_definite_followup:
                context_prompt = dedent(f"""
                    User's current message: "{message}"
                    
                    CRITICAL INSTRUCTION: This is a short follow-up question with pronouns. The user is ALWAYS referring to the MAIN TOPIC/SERVICE in the MOST RECENT message exchange (ITEM 1), not earlier conversations.
                    
                    The conversation history has been carefully selected using advanced analysis of:
                    - Timestamps (temporal relevance)
                    - Conversation flows (topic continuity)
                    - Semantic relevance to the current question
                    - Reference resolution needs (pronouns, implicit references)
                    
                    MOST IMPORTANT RULE: When a user message contains pronouns like "it", "this", or "that", these ALWAYS refer to the MAIN SERVICE, PRODUCT, or PRIMARY TOPIC from the MOST RECENT conversation exchange (ITEM 1).
                    
                    Before answering:
                    1. FIRST, identify what specific service, treatment, or topic the pronoun(s) in the user's message refer to
                       - For example, if they say "can i go for it if i have oily skin?", "it" ALWAYS refers to the MAIN SERVICE/TOPIC in the MOST RECENT conversation
                       - If the most recent conversation (ITEM 1) mentions multiple services/topics, the pronoun refers to the PRIMARY one that was the main focus
                       - The MOST RECENT conversation is ITEM 1 in the history - this is the ONLY item you should check for pronoun resolution
                       - NEVER look at older conversations (ITEM 2, 3, etc.) for pronoun resolution unless explicitly directed by the user
                    
                    2. EXPLICITLY state what you understand the user is referring to at the beginning of your response
                       - Example: "Regarding the Aromatherapy Massage you just asked about, yes you can..."
                       - Be VERY SPECIFIC about which service/topic you're referring to
                    
                    3. The user's previous message (ITEM 1) mentioned a specific service/topic - THAT is what they're referring to now
                    
                    4. NEVER assume they're referring to a service from earlier in the conversation (ITEM 2+) if a more recent service was mentioned
                    
                    5. Provide a complete answer that clearly addresses their question about the identified service/topic
                    
                    Please respond to this message using the MOST RECENT conversation exchange as the primary context.
                """)
            else:
                context_prompt = dedent(f"""
                    User's current message: "{message}"
                    
                    This appears to be a short follow-up question. Before answering:
                    1. Identify what specific service, treatment, or topic the user is referring to
                       - ALWAYS prioritize the MOST RECENT conversation (ITEM 1) for context
                       - If the user's previous message mentioned a specific service, that is most likely what they're referring to
                       - If multiple services/topics are mentioned in ITEM 1, focus on the PRIMARY one that was the main focus
                    
                    2. Check the conversation history to understand the context
                       - The MOST RECENT conversation is the most important for short follow-up questions
                       - Only consider older conversations if the recent ones don't provide sufficient context
                    
                    3. If the user uses pronouns like "it", "this", or "that", determine what they refer to
                       - These pronouns ALMOST ALWAYS refer to the MAIN TOPIC/SERVICE mentioned in the MOST RECENT conversation
                       - Be VERY SPECIFIC about which service/topic you believe the pronoun refers to
                    
                    4. Provide a complete answer that clearly states what you understand the user is asking about
                       - Begin your response by explicitly stating what you believe the user is referring to
                       - Example: "Regarding the Aromatherapy Massage you asked about, ..."
                    
                    Please respond to this message using the MOST RECENT conversation as the primary context.
                """)
        else:
            context_prompt = f"User's current message: {message}\n\nPlease respond to this message using the MOST RELEVANT conversation history for context."
        
        # Get response from Agno agent with retry
        max_retries = min(2, len(gemini_api_keys))
        for retry in range(max_retries):
            try:
                # If this is a retry, get a new agent with a fresh API key
                if retry > 0:
                    logger.info(f"Retrying context agent with new API key (attempt {retry+1})")
                    rotate_gemini_key()
                    context_agent = get_agno_agent(
                        instructions=context_query_instructions, 
                        tools=tools,
                        model_name=model_name
                    )
                
                # Try to get a response
                response = context_agent.run(context_prompt)
                agent_response = response.content
                
                # Check if the agent wants to search the web
                # Look for both explicit search requests and implicit indicators that info is not in DB
                search_needed = False
                search_query = None
                
                # Check for explicit search request
                if serper_api_key and "I need to search for:" in agent_response:
                    # Extract the search query
                    search_query_match = re.search(r'I need to search for:\s*(.+?)(?:\n|$)', agent_response)
                    if search_query_match:
                        search_query = search_query_match.group(1).strip()
                        search_needed = True
                
                # Check for implicit indicators that info is not in DB
                if serper_api_key and not search_needed:
                    no_info_indicators = [
                        "don't have information",
                        "don't have specific information", 
                        "no information available",
                        "not available in our database",
                        "not found in our database",
                        "not in our database",
                        "no data available",
                        "I don't have details",
                        "couldn't find information",
                        "no specific information",
                        "not in our system"
                    ]
                    
                    if any(indicator in agent_response.lower() for indicator in no_info_indicators):
                        logger.info(f"Detected implicit need for web search in context-dependent query: info not in database")
                        # Create a search query based on the user's message and context
                        search_query = message
                        search_needed = True
                
                if search_needed and search_query:
                    # Get the most recent conversation for context
                    most_recent_msg = ""
                    if history_items:
                        most_recent = history_items[0]
                        most_recent_msg = most_recent.get('metadata', {}).get('message', '')
                    
                    # Enhance search query with previous message context if available
                    enhanced_query = search_query
                    if most_recent_msg:
                        # Combine current and previous messages for context-aware search
                        combined_context = f"{most_recent_msg} {message}"
                        # Use the original search query but enhance it with context
                        enhanced_query = f"{search_query} {combined_context}"
                        # Limit query length to avoid issues with search API
                        if len(enhanced_query) > 300:
                            enhanced_query = enhanced_query[:300]
                        
                        logger.info(f"Enhanced search query with conversation context")
                    
                    logger.info(f"Performing web search for context-dependent query: {enhanced_query}")
                    
                    # Import the search function from serper.py
                    from serper import search_web
                    
                    # Perform the search with the enhanced query
                    search_results_text = search_web(enhanced_query, num_results=3)
                    
                    # Format the search results
                    formatted_results = "### WEB SEARCH RESULTS ###\n\n"
                    
                    if not search_results_text:
                        formatted_results += "No relevant search results found."
                    else:
                        formatted_results += search_results_text
                        
                    formatted_results += "\n\n### END OF SEARCH RESULTS ###\n\n"
                    
                    # Send the search results back to the agent for processing
                    context_info = ""
                    if most_recent_msg:
                        context_info = f"The user's previous message was: '{most_recent_msg}'. "
                    
                    follow_up_prompt = f"""
Based on the user's question: '{message}', {context_info}I performed a web search and found the following information:

{formatted_results}

IMPORTANT INSTRUCTIONS:
1. You MUST use the web search results above to answer the user's question. 
2. Do not claim you cannot access the search results or external websites.
3. If the search results contain relevant information, summarize it and provide a complete answer.
4. DO NOT mention that you searched the web or that the information comes from a web search.
5. Present the information as if it's your own knowledge.
6. If the search results don't directly answer the question, provide the best information you can based on what was found.
7. Focus on spa and wellness-related information that is most relevant to the user's question.
8. If the search results contain conflicting information, use your judgment to provide the most accurate answer.
9. If the user asked about compatibility of treatments (e.g., "can I do X after Y?"), provide specific advice based on the search results.
10. REMEMBER: If the user's question contains pronouns like "it", "this", or "that", you MUST explicitly identify what they refer to based on the conversation history.

Please provide a helpful, complete response that directly addresses the user's question.
"""
                    
                    # Get the final response with the search results
                    follow_up_response = context_agent.run(follow_up_prompt)
                    agent_response = follow_up_response.content
                
                # Store the interaction in memory
                store_memory(user_id, message, agent_response, MEMORY_TYPES["INTERACTION"])
                
                return agent_response
            except Exception as e:
                logger.error(f"Error getting response from context agent (attempt {retry+1}): {str(e)}")
                if retry < max_retries - 1:
                    continue
                else:
                    # All retries failed, try a fallback approach
                    return handle_general_query_with_agno(user_id, message)
    except Exception as e:
        logger.error(f"Error in answer_with_smart_context: {str(e)}")
        # Fall back to general query handler
        return handle_general_query_with_agno(user_id, message)
    

    # Handle general queries with Agno
def handle_general_query_with_agno(user_id, message):
    """
    Handle general queries using Agno agent with SQLTools and web search

    Args:
        user_id (str): User ID
        message (str): User message

    Returns:
        str: Response message from Agno agent
    """
    try:
        # Create database connection string
        db_url = f"postgresql+psycopg://{os.getenv('user')}:{os.getenv('password')}@{os.getenv('host')}:{os.getenv('port')}/{os.getenv('dbname')}"

        # Create SQL tools for Agno
        sql_tools = SQLTools(db_url=db_url)

        # Get the previous message from conversation history
        previous_message = ""
        conversation_history = get_conversation_history(user_id, limit=1)
        if conversation_history:
            previous_message = conversation_history[0].get('metadata', {}).get('message', '')
            logger.info(f"Previous message found: {previous_message[:50]}...")

        # Create web search tool if Serper API key is available
        tools = [sql_tools]

        # Create general query agent with specific instructions
        general_query_instructions = dedent("""
            You are an assistant for Red Trends Spa & Wellness Center.
            Your task is to help users with their questions about spa services and related topics.

            IMPORTANT INSTRUCTIONS:
            1. ALWAYS check the database FIRST for information about spa services, prices, etc.
            2. Only use web search if the question cannot be answered from the database
            3. For service-related questions, query the services table
            4. For booking-related questions, direct the user to make a booking
            5. Only search the web for topics related to spa services, wellness, or beauty treatments
            6. When searching the web, consider both the current and previous user messages for context - current message is question if question seems incomplete like maybe it has words like this or it use previous for context about what user is referring to
            7. When searching the web, dont mention searching in web and provide generic answer if not found
            
            DATABASE SCHEMA (EXACT NAMES - DO NOT MODIFY):
            - Table name: public.services
              Columns: "Service ID", "Service Name", "Description", "Price (INR)", "Category"
              Note: Column names include spaces and must be quoted in SQL queries

            - Table name: public.bookings
              Columns: "Booking ID", "Customer Name", "Service ID", "Booking Date", "Time Slot (HH:MM)", "Price (INR)"
              Note: The "Customer Name" column contains the user ID of the customer

            EXAMPLE SQL QUERIES:
            - To find information about a specific service:
              ```sql
              SELECT * FROM public.services WHERE "Service Name" ILIKE '%facial%'
              ```

            - To get all services in a category:
              ```sql
              SELECT * FROM public.services WHERE "Category" ILIKE '%massage%'
              ```

            - To get all available services:
              ```sql
              SELECT * FROM public.services ORDER BY "Category", "Service Name"
              ```

            RESPONSE GUIDELINES:
            1. For service questions, provide details about the service, price, and description
            2. For general spa questions, provide helpful information based on the database or web search
            3. Always be professional, friendly, and concise
            4. If suggesting a service, mention how the user can book it
            5. When listing all services or multiple services, format them in a clean markdown format:
               - Use underline for categories (with markdown __underline__)
               - Use bold for service names
               - Use regular text for price and description
               - Example:
                 ```
                 __Massage Therapy__
                 
                 **Deep Tissue Massage**
                 Price: INR 2500
                 Description: A therapeutic massage targeting deeper muscle layers
                 
                 **Aromatherapy Massage**
                 Price: INR 2200
                 Description: Relaxing massage with essential oils
                 
                 **Facial Treatments**
                 ---
                 
                 **Glowing Facial**
                 Price: INR 1800
                 Description: A customized treatment for radiant skin
                 ```

            HANDLING MISSING INFORMATION:
            1. If information is not found in the database, EXPLICITLY state "I need to search for: [search query]" 
            2. Alternatively, you can indicate that information is not in the database by saying phrases like:
               - "I don't have specific information about that in our database"
               - "That information is not available in our database"
               - "I couldn't find information about that in our system"
            3. When you use these phrases, I will automatically search the web for the information
            4. DO NOT make up information if it's not in the database - either request a search or indicate the information is missing

            IMPORTANT: Only search the web if the question is related to spa services, wellness, or beauty treatments AND cannot be answered from the database.
        """)

        # Add Serper web search capability if API key is available
        if serper_api_key:
            # Create a custom function to search with Serper API
            def search_with_serper(query):
                """
                Search the web using Serper API

                Args:
                    query (str): Search query

                Returns:
                    dict: Search results
                """
                try:
                    headers = {
                        'X-API-KEY': serper_api_key,
                        'Content-Type': 'application/json'
                    }
                    payload = json.dumps({
                        "q": query,
                        "num": 5  # Get top 5 results
                    })
                    response = requests.post('https://google.serper.dev/search', headers=headers, data=payload)
                    return response.json()
                except Exception as e:
                    logger.error(f"Error searching with Serper: {str(e)}")
                    return {"error": str(e)}

            # Add the search function to the instructions
            general_query_instructions += dedent("""

                WEB SEARCH GUIDELINES:
                1. Only use web search if the database doesn't have the information
                2. Focus searches on spa, wellness, and beauty-related topics
                3. Prioritize reputable sources like wellness websites, medical sources, or spa industry publications
                4. Clearly indicate when information comes from web search vs. the database
                5. Do not search for unrelated topics or sensitive information
                6. When search results are provided, you MUST use them to answer the question
                7. Never claim you cannot access search results or external websites

                To search the web, use this format:
                ```
                I need to search for: [your search query]
                ```

                I will then perform the search and provide the results, which you MUST use to answer the user's question.
            """)
        else:
            search_with_serper = None

        # Create the Agno agent with key and model rotation on failure
        try:
            general_query_agent = get_agno_agent_with_retry(
                instructions=general_query_instructions,
                tools=tools
            )

            # If we have a Serper API key, we need to handle the search manually
            # since we're not using a dedicated Agno tool for web search
            if serper_api_key:
                logger.info("Serper API key available - will handle web searches manually")
        except Exception as e:
            logger.error(f"Failed to create general query agent after retries: {str(e)}")
            return "I'm sorry, but I'm having trouble accessing our information system right now. Please try again later or contact us directly at (555) 123-4567 for assistance."

        # Prepare the prompt for the general query agent
        general_query_prompt = f"User has asked: '{message}'. "
        
        # Include previous message for context if available
        if previous_message:
            general_query_prompt += f"Their previous message was: '{previous_message}'. "
        
        general_query_prompt += "First check our database for relevant information about our services. "

        if serper_api_key:
            general_query_prompt += "If the database doesn't have the answer and the question is related to spa services, wellness, or beauty treatments, you have two options: "
            general_query_prompt += "1. Explicitly indicate that you need to search the web by saying 'I need to search for: [your search query]' OR "
            general_query_prompt += "2. Use phrases like 'I don't have specific information about that in our database' or 'That information is not available in our system' "
            general_query_prompt += "When you use either approach, I will automatically search the web for the information. "
            general_query_prompt += "When I provide web search results, you MUST use them to answer the question. Never claim you cannot access search results or external websites. "
            general_query_prompt += "DO NOT make up information if it's not in the database - either request a search or indicate the information is missing. "

        general_query_prompt += "Provide a helpful, accurate response based on all available information."

        # Get response from Agno agent with retry
        max_retries = min(3, len(gemini_api_keys) * 2)
        for retry in range(max_retries):
            try:
                # If this is a retry, get a new agent with a fresh API key and model
                if retry > 0:
                    logger.info(f"Retrying general query agent with new API key/model (attempt {retry+1})")
                    rotate_gemini_key_and_model()
                    general_query_agent = get_agno_agent(instructions=general_query_instructions, tools=tools)

                # Try to get a response
                response = general_query_agent.run(general_query_prompt)
                agent_response = response.content

                # Check if the agent wants to search the web
                # Look for both explicit search requests and implicit indicators that info is not in DB
                search_needed = False
                search_query = None
                
                # Check for explicit search request
                if serper_api_key and "I need to search for:" in agent_response:
                    # Extract the search query
                    search_query_match = re.search(r'I need to search for:\s*(.+?)(?:\n|$)', agent_response)
                    if search_query_match:
                        search_query = search_query_match.group(1).strip()
                        search_needed = True
                
                # Check for implicit indicators that info is not in DB
                if serper_api_key and not search_needed:
                    no_info_indicators = [
                        "don't have information",
                        "don't have specific information", 
                        "no information available",
                        "not available in our database",
                        "not found in our database",
                        "not in our database",
                        "no data available",
                        "I don't have details",
                        "couldn't find information",
                        "no specific information",
                        "not in our system"
                    ]
                    
                    if any(indicator in agent_response.lower() for indicator in no_info_indicators):
                        logger.info(f"Detected implicit need for web search: info not in database")
                        # Create a search query based on the user's message
                        search_query = message
                        search_needed = True
                
                if search_needed and search_query:
                    # Enhance search query with previous message context if available
                    enhanced_query = search_query
                    if previous_message:
                        # Combine current and previous messages for context-aware search
                        # Extract key terms from both messages to create a more focused query
                        combined_context = f"{previous_message} {message}"
                        # Use the original search query but enhance it with context
                        enhanced_query = f"{search_query} {combined_context}"
                        # Limit query length to avoid issues with search API
                        if len(enhanced_query) > 300:
                            enhanced_query = enhanced_query[:300]
                        
                        logger.info(f"Enhanced search query with previous message context")
                    
                    logger.info(f"Performing web search for: {enhanced_query}")

                    # Perform the search with the enhanced query
                    search_results = search_with_serper(enhanced_query)

                    # Format the search results
                    formatted_results = "### WEB SEARCH RESULTS ###\n\n"

                    if "error" in search_results:
                        formatted_results += f"Error performing search: {search_results['error']}"
                    else:
                        # Add organic results
                        if "organic" in search_results:
                            formatted_results += "ORGANIC SEARCH RESULTS:\n"
                            for i, result in enumerate(search_results["organic"][:3], 1):
                                title = result.get("title", "No title")
                                link = result.get("link", "No link")
                                snippet = result.get("snippet", "No description")
                                formatted_results += f"RESULT {i}:\nTitle: {title}\nContent: {snippet}\nSource: {link}\n\n"

                        # Add knowledge graph if available
                        if "knowledgeGraph" in search_results:
                            kg = search_results["knowledgeGraph"]
                            if "title" in kg:
                                formatted_results += "KNOWLEDGE GRAPH INFORMATION:\n"
                                formatted_results += f"Title: {kg.get('title')}\n"
                                if "description" in kg:
                                    formatted_results += f"Description: {kg.get('description')}\n\n"
                                    
                    formatted_results += "### END OF SEARCH RESULTS ###\n\n"

                    # Send the search results back to the agent for processing
                    context_info = ""
                    if previous_message:
                        context_info = f"The user's previous message was: '{previous_message}'. "
                    
                    follow_up_prompt = f"""
Based on the user's question: '{message}', {context_info}I performed a web search for '{enhanced_query}' and found the following information:

{formatted_results}

IMPORTANT INSTRUCTIONS:
1. You MUST use the web search results above to answer the user's question. 
2. Do not claim you cannot access the search results or external websites.
3. If the search results contain relevant information, summarize it and provide a complete answer.
4. DO NOT mention that you searched the web or that the information comes from a web search.
5. Present the information as if it's your own knowledge.
6. If the search results don't directly answer the question, provide the best information you can based on what was found.
7. Focus on spa and wellness-related information that is most relevant to the user's question.
8. If the search results contain conflicting information, use your judgment to provide the most accurate answer.
9. If the user asked about compatibility of treatments (e.g., "can I do X after Y?"), provide specific advice based on the search results.

Please provide a helpful, complete response that directly addresses the user's question.
"""

                    # Get the final response with the search results
                    follow_up_response = general_query_agent.run(follow_up_prompt)
                    agent_response = follow_up_response.content

                # Make sure we have a valid response
                if agent_response is None:
                    raise ValueError("Agno agent returned None response")

                # Log the general query interaction
                logger.info(f"General query interaction - User: {user_id}, Query: {message[:50]}..., Response: {agent_response[:100]}...")

                # Store the interaction in memory
                store_memory(user_id, message, agent_response)

                return agent_response
            except Exception as inner_e:
                logger.error(f"Error getting response from Agno agent (attempt {retry+1}/{max_retries}): {str(inner_e)}")
                if retry < max_retries - 1:
                    # Will retry with a new key and model
                    continue
                else:
                    # All retries failed, return a fallback response
                    return "I'm sorry, but I'm having trouble processing your question right now. Please try asking in a different way or contact our staff directly at (555) 123-4567 for assistance."

    except Exception as e:
        logger.error(f"Error handling general query with Agno: {str(e)}")
        return "I'm sorry, but I encountered an error while trying to answer your question. Please try again or contact our staff for assistance."


# Extract entities (services, treatments, etc.) from conversation history
def extract_entities_from_history(history_items):
    """
    Extract potential entities (services, treatments, etc.) from conversation history
    
    Args:
        history_items (list): List of conversation history items
        
    Returns:
        str: Formatted string of extracted entities
    """
    try:
        if not history_items:
            return "No entities found in conversation history."
            
        # Keywords to look for in the conversation
        service_keywords = [
            "facial", "massage", "treatment", "therapy", "service", "package", 
            "spa", "body", "scrub", "wrap", "manicure", "pedicure", "hair", 
            "makeup", "exfoliation", "detan", "glowing"
        ]
        
        # Extract potential service names and other entities
        entities = set()
        
        # Special handling for the most recent item - it's likely the most relevant
        if history_items:
            most_recent = history_items[0]
            metadata = most_recent.get('metadata', {})
            recent_user_msg = metadata.get('message', '').lower()
            recent_bot_msg = metadata.get('response', '').lower()
            
            # Prioritize entities from the most recent message
            recent_entities = extract_entities_from_text(recent_user_msg + " " + recent_bot_msg, service_keywords)
            entities.update(recent_entities)
            
            # If we found entities in the most recent message, mark them as such
            if recent_entities:
                entities.add("MOST_RECENT: " + ", ".join(recent_entities))
        
        # Process all history items
        for item in history_items:
            metadata = item.get('metadata', {})
            user_msg = metadata.get('message', '').lower()
            bot_msg = metadata.get('response', '').lower()
            
            combined_text = user_msg + " " + bot_msg
            found_entities = extract_entities_from_text(combined_text, service_keywords)
            entities.update(found_entities)
        
        # Format the entities
        if entities:
            # Sort entities but put the MOST_RECENT ones at the top
            sorted_entities = sorted([e for e in entities if not e.startswith("MOST_RECENT:")])
            most_recent_entities = [e for e in entities if e.startswith("MOST_RECENT:")]
            
            all_entities = most_recent_entities + sorted_entities
            entity_list = "- " + "\n- ".join(all_entities)
            
            return f"The following entities have been mentioned in the conversation:\n{entity_list}"
        else:
            return "No specific entities identified in the conversation history."
            
    except Exception as e:
        logger.error(f"Error extracting entities from history: {str(e)}")
        return "Error extracting entities from conversation history."