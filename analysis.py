import re
import logging
import google.generativeai as genai
from datetime import datetime, timedelta
from textwrap import dedent
from config import get_gemini_model, gemini_api_keys
from date_filter import format_date_for_postgresql
from db_connect import get_services_from_db
from agent_init import get_agno_agent, get_agno_agent_with_retry, rotate_gemini_key

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Analyze message for intents and entities
def analyze_message(message):
    """
    Analyze a message for intents and entities

    Args:
        message (str): User message

    Returns:
        dict: Dictionary with intents and entities
    """
    # Initialize results
    results = {
        "intents": [],
        "service_entities": [],
        "date_entities": [],
        "time_entities": [],
        "customer_entities": []
    }

    # Convert message to lowercase for easier matching
    message_lower = message.lower()

    # Check for booking-related intents
    if any(word in message_lower for word in ["book", "appointment", "schedule", "reserve"]):
        results["intents"].append("booking")

    # Check for booking query intents
    if any(phrase in message_lower for phrase in ["my bookings", "my appointments", "check bookings",
                                                 "view bookings", "show bookings", "see my bookings"]):
        results["intents"].append("booking_query")

        # Check for specific booking query types
        if any(word in message_lower for word in ["previous", "past", "earlier", "before", "history"]):
            results["intents"].append("previous_bookings")
        if any(word in message_lower for word in ["future", "upcoming", "next", "scheduled", "coming"]):
            results["intents"].append("future_bookings")

    # Check for service-related intents
    if any(word in message_lower for word in ["service", "treatment", "massage", "facial", "spa", "price", "cost", "how much"]):
        results["intents"].append("service_query")

    # Check for general information intents
    if any(phrase in message_lower for phrase in ["what is", "tell me about", "explain", "how does", "benefits of"]):
        results["intents"].append("information_query")

    # Extract service entities (simplified - in a real app, would use NER)
    # Get services from database
    services = get_services_from_db()
    for service in services:
        service_name = service.get("name", "").lower()
        if service_name and service_name in message_lower:
            results["service_entities"].append(service.get("name"))

    # Extract date entities (simplified)
    date_patterns = [
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY
        r'\b\d{4}-\d{1,2}-\d{1,2}\b',    # YYYY-MM-DD
        r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?\b',  # Month Day
        r'\b(?:tomorrow|today|next week|next month)\b'  # Relative dates
    ]

    date_entities = []
    for pattern in date_patterns:
        matches = re.findall(pattern, message_lower)
        for match in matches:
            # Format the date for PostgreSQL
            formatted_date = format_date_for_postgresql(match)
            date_entities.append(formatted_date)
    
    # Add the formatted dates to results
    results["date_entities"] = date_entities

    # Extract time entities (simplified)
    time_patterns = [
        r'\b\d{1,2}:\d{2}\s*(?:am|pm)?\b',  # HH:MM am/pm
        r'\b\d{1,2}\s*(?:am|pm)\b',         # HH am/pm
        r'\b(?:morning|afternoon|evening)\b'  # Time of day
    ]

    for pattern in time_patterns:
        matches = re.findall(pattern, message_lower)
        results["time_entities"].extend(matches)

    # Extract customer names (simplified)
    # Look for patterns like "for John Doe" or "customer John Doe"
    customer_patterns = [
        r'for\s+([A-Za-z\s]+?)(?:\s+on|\s+at|\s+with|\.|,|\s+$)',
        r'customer\s+([A-Za-z\s]+?)(?:\s+on|\s+at|\s+with|\.|,|\s+$)',
        r'client\s+([A-Za-z\s]+?)(?:\s+on|\s+at|\s+with|\.|,|\s+$)'
    ]

    for pattern in customer_patterns:
        matches = re.findall(pattern, message)
        for match in matches:
            if match.strip() and len(match.strip()) > 2:  # Avoid single letters
                results["customer_entities"].append(match.strip())

    return results


# Classify message type using Agno and Gemini
def classify_message_type(message):
    """
    Classify a message as either "standard_flow" or "context_dependent" using Gemini with Agno
    
    Standard flow: booking retrieval, create booking, service info, web search
    Context dependent: messages like "who am I", "what service were we talking about", etc.

    Args:
        message (str): User message to classify

    Returns:
        str: Classification result - either "standard_flow" or "context_dependent"
    """
    try:
        # Create instructions for the classification agent
        classification_instructions = dedent("""
            You are a message classification assistant for Red Trends Spa & Wellness Center.
            Your task is to analyze user messages and classify them into one of two categories:
            
            1. "standard_flow" - Messages that are about:
               - booking retrieval (checking existing bookings)
               - create booking (making new appointments)
               - service info (asking about spa services, prices, etc.)
               - web search (general questions about spa treatments, wellness, etc.)
               
            2. "context_dependent" - Messages that require previous conversation context to understand, such as:
               - "Who am I?" (asking about user identity)
               - "What service were we talking about just now?" (referring to previous conversation)
               - "What's the price of that item we were just talking about?" (referring to previous context)
               - "Can you tell me more about it?" (where "it" refers to something mentioned earlier)
               - "Is that available tomorrow?" (where "that" refers to something mentioned earlier)
               - Any message with pronouns like "it", "that", "this", "those" without clear referents
               - Any message that seems incomplete without previous context
            
            CLASSIFICATION GUIDELINES:
            - "standard_flow" includes: direct questions about services, explicit booking requests, 
              clear inquiries about spa information that don't reference previous conversation.
              
            - "context_dependent" includes: questions with pronouns that lack clear referents,
              requests for clarification about previous topics, questions that seem incomplete
              without knowing what was discussed before.
              
            RESPONSE FORMAT:
            Respond with ONLY ONE of these two values:
            - "standard_flow"
            - "context_dependent"
            
            Do not include any other text, explanation, or formatting in your response.
        """)
        
        # Create the Agno agent with key rotation on failure
        try:
            classification_agent = get_agno_agent_with_retry(instructions=classification_instructions)
        except Exception as e:
            logger.error(f"Failed to create classification agent after retries: {str(e)}")
            # Default to standard_flow if we can't classify
            return "standard_flow"
        
        # Prepare the prompt for the classification agent
        classification_prompt = f"Classify this message as either 'standard_flow' or 'context_dependent': '{message}'"
        
        # Get response from Agno agent with retry
        max_retries = min(2, len(gemini_api_keys))
        for retry in range(max_retries):
            try:
                # If this is a retry, get a new agent with a fresh API key
                if retry > 0:
                    logger.info(f"Retrying classification agent with new API key (attempt {retry+1})")
                    rotate_gemini_key()
                    classification_agent = get_agno_agent(instructions=classification_instructions)
                
                # Try to get a response
                response = classification_agent.run(classification_prompt)
                agent_response = response.content.strip().lower()
                
                # Make sure we have a valid response
                if agent_response is None:
                    raise ValueError("Agno agent returned None response")
                
                # Normalize the response to ensure it's one of our expected values
                if "context_dependent" in agent_response:
                    return "context_dependent"
                elif "standard_flow" in agent_response:
                    return "standard_flow"
                else:
                    # If the response doesn't clearly match either category, default to standard_flow
                    logger.warning(f"Unclear classification response: {agent_response}. Defaulting to standard_flow.")
                    return "standard_flow"
                
            except Exception as inner_e:
                logger.error(f"Error getting response from classification agent (attempt {retry+1}/{max_retries}): {str(inner_e)}")
                if retry < max_retries - 1:
                    # Will retry with a new key
                    continue
                else:
                    # All retries failed, default to standard_flow
                    logger.error("All classification attempts failed. Defaulting to standard_flow.")
                    return "standard_flow"
    
    except Exception as e:
        logger.error(f"Error classifying message type: {str(e)}")
        # Default to standard_flow if we encounter an error
        return "standard_flow"
    
    
def classify_booking_message(message):
    """
    Classify a message as either "booking_retrieval" or "create_booking" using Gemini with Agno

    Args:
        message (str): User message to classify

    Returns:
        str: Classification result - either "booking_retrieval" or "create_booking"
    """
    try:
        # Create instructions for the classification agent
        classification_instructions = dedent("""
            You are a message classification assistant for Red Trends Spa & Wellness Center.
            Your task is to analyze user messages and classify them into one of two categories:
            
            1. "booking_retrieval" - Messages asking about existing bookings, checking booking status, 
               or requesting information about past or upcoming appointments.
               
            2. "create_booking" - Messages expressing intent to make a new booking, schedule an appointment,
               or reserve a service.
            
            CLASSIFICATION GUIDELINES:
            - "booking_retrieval" includes: checking appointment status, viewing upcoming bookings, 
              asking about past bookings, requesting booking details, etc.
              
            - "create_booking" includes: making new appointments, scheduling services, requesting available 
              time slots with intent to book, etc.
              
            RESPONSE FORMAT:
            Respond with ONLY ONE of these two values:
            - "booking_retrieval"
            - "create_booking"
            
            Do not include any other text, explanation, or formatting in your response.
        """)
        
        # Create the Agno agent with key rotation on failure
        try:
            classification_agent = get_agno_agent_with_retry(instructions=classification_instructions)
        except Exception as e:
            logger.error(f"Failed to create classification agent after retries: {str(e)}")
            # Default to create_booking if we can't classify
            return "create_booking"
        
        # Prepare the prompt for the classification agent
        classification_prompt = f"Classify this message as either 'booking_retrieval' or 'create_booking': '{message}'"
        
        # Get response from Agno agent with retry
        max_retries = min(2, len(gemini_api_keys))
        for retry in range(max_retries):
            try:
                # If this is a retry, get a new agent with a fresh API key
                if retry > 0:
                    logger.info(f"Retrying classification agent with new API key (attempt {retry+1})")
                    rotate_gemini_key()
                    classification_agent = get_agno_agent(instructions=classification_instructions)
                
                # Try to get a response
                response = classification_agent.run(classification_prompt)
                agent_response = response.content.strip().lower()
                
                # Make sure we have a valid response
                if agent_response is None:
                    raise ValueError("Agno agent returned None response")
                
                # Normalize the response to ensure it's one of our expected values
                if "retrieval" in agent_response:
                    return "booking_retrieval"
                elif "create" in agent_response:
                    return "create_booking"
                else:
                    # If the response doesn't clearly match either category, default to create_booking
                    logger.warning(f"Unclear classification response: {agent_response}. Defaulting to create_booking.")
                    return "create_booking"
                
            except Exception as inner_e:
                logger.error(f"Error getting response from classification agent (attempt {retry+1}/{max_retries}): {str(inner_e)}")
                if retry < max_retries - 1:
                    # Will retry with a new key
                    continue
                else:
                    # All retries failed, default to create_booking
                    logger.error("All classification attempts failed. Defaulting to create_booking.")
                    return "create_booking"
    
    except Exception as e:
        logger.error(f"Error classifying booking message: {str(e)}")
        # Default to create_booking if we encounter an error
        return "create_booking"