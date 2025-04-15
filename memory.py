import uuid
import time
import re
import json
import logging
from datetime import datetime
from textwrap import dedent
from mem.emb import get_embedding, cosine_similarity
from mem.pine_client import index
from config import MEMORY_TYPES
from date_filter import extract_entities_from_text
from agent_init import get_agno_agent, get_agno_agent_with_retry, rotate_gemini_key

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define adjust_window_size function here to avoid circular imports
def adjust_window_size(current_message, history_items):
    """
    Dynamically adjust the window size based on the characteristics of the current message.
    
    Args:
        current_message (str): The current user message
        history_items (list): List of conversation history items
        
    Returns:
        int: Adjusted window size
    """
    # Base window size
    base_size = 5
    
    # Check if this is a short message (likely a follow-up)
    is_short_message = len(current_message.split()) <= 5
    
    # Check for pronouns that indicate a follow-up
    has_pronouns = any(pronoun in current_message.lower() for pronoun in 
                      ['it', 'this', 'that', 'they', 'them', 'these', 'those'])
    
    # Check for explicit references to previous conversation
    has_reference = any(ref in current_message.lower() for ref in 
                       ['earlier', 'before', 'previous', 'last time', 'you said', 'you mentioned'])
    
    # Adjust window size based on message characteristics
    if is_short_message and has_pronouns:
        # For definite follow-ups, use a smaller window focused on recent context
        window_size = 3
    elif has_reference:
        # For explicit references to earlier conversation, use a larger window
        window_size = min(10, len(history_items))
    elif is_short_message:
        # For short messages without pronouns, use a medium window
        window_size = 4
    else:
        # For longer, more complex queries, use the base window size
        window_size = base_size
    
    # Ensure we don't exceed the available history
    return min(window_size, len(history_items))

# Store memory with classification
def store_memory(user_id, message, response, memory_type=MEMORY_TYPES["INTERACTION"], additional_metadata=None):
    # Ensure response is not None
    if response is None:
        response = "No response generated"

    combined = f"User: {message}\nBot: {response}"
    vector = get_embedding(combined)
    timestamp = int(time.time())
    memory_id = f"{user_id}-{uuid.uuid4()}"

    metadata = {
        "user_id": user_id,
        "message": message,
        "response": response,
        "timestamp": timestamp,
        "type": memory_type
    }

    if additional_metadata:
 
        sanitized_metadata = {}
        for key, value in additional_metadata.items():
            if value is None:
                sanitized_metadata[key] = "None"
            elif isinstance(value, (str, int, float, bool)) or (isinstance(value, list) and all(isinstance(item, str) for item in value)):
                sanitized_metadata[key] = value
            else:
                sanitized_metadata[key] = str(value)

        metadata.update(sanitized_metadata)

    index.upsert([
        (memory_id, vector, metadata)
    ])

   
    if memory_type == MEMORY_TYPES["FAQ"]:
        question = extract_question(message)
        if question:
            faq_vector = get_embedding(question)
            faq_id = f"{user_id}-faq-{uuid.uuid4()}"
            index.upsert([
                (faq_id, faq_vector, {
                    "user_id": user_id,
                    "question": question,
                    "answer": response,
                    "timestamp": timestamp,
                    "type": "faq_entry"
                })
            ])


def extract_question(text):
    """Extract a question from text if present"""
    question_patterns = [
        r'(?:^|\s)(?:what|how|when|where|why|who|can|could|would|will|is|are|do|does|did|should)(?:\s+\w+){2,}[?]',
        r'(?:^|\s)(?:tell me about|explain|describe)(?:\s+\w+){2,}[?]?',
        r'[^.!?]*[?]'
    ]

    for pattern in question_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[0].strip()

    return None


# Store user preference
def store_user_preference(user_id, key, value):
    # Check if preference already exists
    res = index.query(
        vector=[0] * 768,  # Gemini dimension
        top_k=1,
        include_metadata=True,
        filter={
            "user_id": {"$eq": user_id},
            "type": {"$eq": MEMORY_TYPES["PREFERENCE"]},
            "preference_key": {"$eq": key}
        }
    )

    matches = res.get("matches", [])
    timestamp = int(time.time())

    if matches:
        # Update existing preference
        memory_id = matches[0]["id"]
        vector = get_embedding(f"{key}: {value}")

        index.upsert([
            (memory_id, vector, {
                "user_id": user_id,
                "preference_key": key,
                "preference_value": value,
                "timestamp": timestamp,
                "type": MEMORY_TYPES["PREFERENCE"]
            })
        ])
    else:
        # Create new preference
        memory_id = f"{user_id}-pref-{uuid.uuid4()}"
        vector = get_embedding(f"{key}: {value}")

        index.upsert([
            (memory_id, vector, {
                "user_id": user_id,
                "preference_key": key,
                "preference_value": value,
                "timestamp": timestamp,
                "type": MEMORY_TYPES["PREFERENCE"]
            })
        ])

# Format conversation history into a readable context
def format_conversation_context(history):
    # Reverse to get chronological order (oldest first)
    history = list(reversed(history))
    context = []
    for item in history:
        metadata = item['metadata']
        context.append(f"User: {metadata['message']}")
        context.append(f"Assistant: {metadata['response']}")

    return "\n".join(context)



def recall_semantic_memory(user_id, query, top_k=3, memory_types=None):
    query_vec = get_embedding(query)

    filter_dict = {"user_id": {"$eq": user_id}}

    # Add memory type filter if specified
    if memory_types:
        if isinstance(memory_types, list):
            filter_dict["type"] = {"$in": memory_types}
        else:
            filter_dict["type"] = {"$eq": memory_types}

    res = index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict
    )

    return res.get("matches", [])



# Get recent conversation history for context
def get_conversation_history(user_id, limit=5):
    """
    Get the most recent conversation history for a user.
    This is the original implementation, kept for backward compatibility.
    
    Args:
        user_id (str): User ID
        limit (int): Maximum number of history items to retrieve
        
    Returns:
        list: Sorted list of conversation history items (newest first)
    """
    # Query the most recent interactions for this user
    res = index.query(
        vector=[0] * 768,  # Dummy vector, we're just using filters (Gemini dimension)
        top_k=limit * 2,  # Getting more than needed to filter
        include_metadata=True,
        filter={
            "user_id": {"$eq": user_id},
            "type": {"$eq": MEMORY_TYPES["INTERACTION"]}
        }
    )

    # Sort by timestamp (newest first)
    matches = res.get("matches", [])
    sorted_matches = sorted(matches, key=lambda x: x['metadata'].get('timestamp', 0), reverse=True)

    return sorted_matches[:limit]

def get_adaptive_conversation_history(user_id, current_message, initial_limit=15):
    """
    Get conversation history using an adaptive sliding window approach.
    Retrieves recent messages and then filters them by semantic relevance.
    
    Args:
        user_id (str): User ID
        current_message (str): The current user message
        initial_limit (int): Initial number of history items to retrieve
        
    Returns:
        list: List of conversation history items sorted by relevance
    """
    try:
        # Step 1: Get a larger initial set of recent messages
        res = index.query(
            vector=[0] * 768,  # Dummy vector, we're just using filters
            top_k=initial_limit,  # Get more than we'll eventually use
            include_metadata=True,
            filter={
                "user_id": {"$eq": user_id},
                "type": {"$eq": MEMORY_TYPES["INTERACTION"]}
            }
        )
        
        matches = res.get("matches", [])
        if not matches:
            logger.info(f"No conversation history found for user {user_id}")
            return []
            
        # Step 2: Sort by timestamp (newest first) to create the initial sliding window
        sorted_by_time = sorted(matches, key=lambda x: x.get('metadata', {}).get('timestamp', 0), reverse=True)
        
        # Step 3: Determine the appropriate window size based on the current message
        window_size = adjust_window_size(current_message, sorted_by_time)
        logger.info(f"Using adaptive window size of {window_size} for message: '{current_message[:50]}...'")
        
        # Step 4: Rank the history items by relevance to the current message
        ranked_items = rank_history_by_relevance(current_message, sorted_by_time, window_size)
        
        return ranked_items
        
    except Exception as e:
        logger.error(f"Error in get_adaptive_conversation_history: {str(e)}")
        # Fall back to the standard method if there's an error
        return get_conversation_history(user_id, limit=5)

# Get FAQs for this user
def get_user_faqs(user_id, limit=5):
    res = index.query(
        vector=[0] * 768,  # Dummy vector, we're just using filters (Gemini dimension)
        top_k=limit,
        include_metadata=True,
        filter={
            "user_id": {"$eq": user_id},
            "type": {"$eq": "faq_entry"}
        }
    )

    return res.get("matches", [])

# Get user preferences
def get_user_preferences(user_id):
    res = index.query(
        vector=[0] * 768,  # Dummy vector (Gemini dimension)
        top_k=10,
        include_metadata=True,
        filter={
            "user_id": {"$eq": user_id},
            "type": {"$eq": MEMORY_TYPES["PREFERENCE"]}
        }
    )

    preferences = {}
    for match in res.get("matches", []):
        metadata = match.get("metadata", {})
        if "preference_key" in metadata and "preference_value" in metadata:
            preferences[metadata["preference_key"]] = metadata["preference_value"]

    return preferences

def extract_and_store_preferences(user_id, message, response):
    """Extract and store user preferences from the conversation"""
    # Look for service preferences from database
    try:
        services = get_services_from_db()
        if services:
            for service in services:
                # Convert to string to ensure we can call lower() and handle None values
                service_name = str(service["name"]).lower() if service["name"] is not None else ""
                if service_name and service_name in message.lower() and any(word in message.lower() for word in ["like", "prefer", "favorite", "enjoy"]):
                    store_user_preference(user_id, "preferred_service", str(service["name"]))
    except Exception as e:
        logger.error(f"Error extracting preferences: {str(e)}")

    # Look for time preferences
    time_patterns = [
        (r'prefer\s+(\w+day)', "preferred_day"),
        (r'prefer\s+(morning|afternoon|evening)', "preferred_time"),
        (r'(morning|afternoon|evening)\s+is better', "preferred_time")
    ]

    for pattern, key in time_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            store_user_preference(user_id, key, match.group(1))


def adjust_window_size(current_message, history_items):
    """
    Dynamically adjust the window size based on the characteristics of the current message.
    
    Args:
        current_message (str): The current user message
        history_items (list): List of conversation history items
        
    Returns:
        int: Adjusted window size
    """
    # Base window size
    base_size = 5
    
    # Check if this is a short message (likely a follow-up)
    is_short_message = len(current_message.split()) <= 5
    
    # Check for pronouns that indicate a follow-up
    has_pronouns = any(pronoun in current_message.lower() for pronoun in 
                      ['it', 'this', 'that', 'they', 'them', 'these', 'those'])
    
    # Check for explicit references to previous conversation
    has_reference = any(ref in current_message.lower() for ref in 
                       ['earlier', 'before', 'previous', 'last time', 'you said', 'you mentioned'])
    
    # Adjust window size based on message characteristics
    if is_short_message and has_pronouns:
        # For definite follow-ups, use a smaller window focused on recent context
        window_size = 3
    elif has_reference:
        # For explicit references to earlier conversation, use a larger window
        window_size = min(10, len(history_items))
    elif is_short_message:
        # For short messages without pronouns, use a medium window
        window_size = 4
    else:
        # For longer, more complex queries, use the base window size
        window_size = base_size
    
    # Ensure we don't exceed the available history
    return min(window_size, len(history_items))

def rank_history_by_relevance(current_message, history_items, max_items=5):
    """
    Rank conversation history items by a combination of recency and semantic similarity,
    with special handling for pronoun resolution when multiple potential referents are present.
    
    Args:
        current_message (str): The current user message
        history_items (list): List of conversation history items
        max_items (int): Maximum number of items to return
        
    Returns:
        list: Ranked list of conversation history items
    """
    try:
        if not history_items:
            return []
        
        # For very short messages with pronouns, they're almost certainly follow-ups
        # to the most recent conversation, so prioritize recency
        is_short_followup = len(current_message.split()) <= 5
        has_pronouns = any(pronoun in current_message.lower() for pronoun in 
                          ['it', 'this', 'that', 'they', 'them', 'these', 'those'])
        
        # CRITICAL: For definite follow-ups with pronouns, ALWAYS prioritize the most recent item
        # This is the most important rule for pronoun resolution
        if is_short_followup and has_pronouns and history_items:
            logger.info(f"Definite follow-up with pronouns detected: '{current_message}'. STRONGLY prioritizing most recent context.")
            
            # Always include the most recent item as the primary context
            # This is critical for correct pronoun resolution
            most_recent = [history_items[0]]
            
            # If there's only one item, just return it
            if len(history_items) == 1:
                return most_recent
                
            # Check if the second item is part of the same conversation session
            if len(history_items) >= 2:
                # Get timestamps
                most_recent_timestamp = history_items[0].get('metadata', {}).get('timestamp', 0)
                second_item_timestamp = history_items[1].get('metadata', {}).get('timestamp', 0)
                
                # Calculate time difference
                time_diff = abs(most_recent_timestamp - second_item_timestamp)
                
                # If the second item is within 5 minutes of the most recent, include it as part of the same conversation
                if time_diff <= 300:  # 5 minutes = 300 seconds
                    most_recent.append(history_items[1])
                    logger.info(f"Including second most recent item as part of the same conversation session (time diff: {time_diff}s)")
            
            # For the remaining items, compute semantic similarity
            remaining_items = [item for item in history_items if item not in most_recent]
        else:
            # For other types of messages, still include the most recent item
            # but with less priority
            most_recent = [history_items[0]] if history_items else []
            
            # If there's only one item, just return it
            if len(history_items) <= 1:
                return most_recent
                
            # For the remaining items, compute semantic similarity
            remaining_items = history_items[1:]
        
        # Get embedding for the current message
        try:
            query_embedding = get_embedding(current_message)
        except Exception as e:
            logger.error(f"Error getting embedding for current message: {str(e)}")
            # If embedding fails, fall back to recency-based ranking
            return history_items[:max_items]
        
        # Compute similarity scores for each history item
        scored_items = []
        for item in remaining_items:
            try:
                # Combine user message and bot response for context
                metadata = item.get('metadata', {})
                user_msg = metadata.get('message', '')
                bot_msg = metadata.get('response', '')
                combined_text = f"{user_msg} {bot_msg}"
                
                # Get embedding for the combined text
                item_embedding = get_embedding(combined_text)
                
                # Compute cosine similarity
                similarity = cosine_similarity(query_embedding, item_embedding)
                
                # Get timestamp for recency factor
                timestamp = metadata.get('timestamp', 0)
                
                # Compute a combined score that balances recency and relevance
                # More recent items get a boost, but similarity is the primary factor
                recency_factor = 1.0
                if timestamp > 0:
                    # Calculate how recent the item is (normalized to 0-1)
                    # Assuming the most recent item has the highest timestamp
                    max_timestamp = history_items[0].get('metadata', {}).get('timestamp', timestamp)
                    if max_timestamp > timestamp:  # Avoid division by zero
                        recency = (timestamp / max_timestamp) * 0.5  # Scale recency to 0-0.5
                        recency_factor = 1.0 + recency  # Recency boost between 1.0-1.5
                
                # Special handling for pronoun resolution
                pronoun_boost = 1.0
                if has_pronouns:
                    # Check if this item contains potential referents for pronouns
                    # Look for nouns, products, services, etc. that could be referenced by "it", "this", etc.
                    lower_combined = combined_text.lower()
                    
                    # Check for service-related terms that might be referenced by pronouns
                    service_terms = ['massage', 'facial', 'treatment', 'therapy', 'service', 
                                    'polish', 'scrub', 'wrap', 'manicure', 'pedicure', 'spa']
                    
                    if any(term in lower_combined for term in service_terms):
                        # This item contains potential referents for pronouns
                        pronoun_boost = 1.3  # 30% boost for items with potential referents
                        
                        # Additional boost for items that mention the same service terms as in the query
                        query_lower = current_message.lower()
                        matching_terms = [term for term in service_terms if term in query_lower and term in lower_combined]
                        if matching_terms:
                            pronoun_boost = 1.5  # 50% boost for items with matching service terms
                
                # Final score combines similarity, recency, and pronoun resolution factors
                final_score = similarity * recency_factor * pronoun_boost
                
                scored_items.append((item, final_score))
            except Exception as item_e:
                logger.error(f"Error processing history item: {str(item_e)}")
                # Skip this item if there's an error
                continue
        
        # Sort by score in descending order
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # Select top items from the ranked list
        selected_items = [item for item, score in scored_items[:max_items - len(most_recent)]]
        
        # Combine with the most recent items that we always include for follow-ups
        final_selection = most_recent + selected_items
        
        logger.info(f"Ranked {len(history_items)} history items by relevance, selected {len(final_selection)}")
        return final_selection
        
    except Exception as e:
        logger.error(f"Error ranking history by relevance: {str(e)}")
        # Fall back to recency-based ranking if there's an error
        return history_items[:max_items]

def get_ranked_conversation_history(user_id, limit=10, format_as_text=True, current_message=None):
    """
    Get the user's conversation history ranked by relevance to the current message.
    If current_message is provided, uses adaptive sliding window and semantic ranking.
    Otherwise, ranks by recency only (backward compatibility).
    
    Args:
        user_id (str): User ID
        limit (int): Maximum number of history items to retrieve
        format_as_text (bool): Whether to format the history as text
        current_message (str, optional): Current user message for relevance ranking
        
    Returns:
        If format_as_text is True: 
            str: Formatted conversation history text
        If format_as_text is False:
            list: Ranked list of conversation history items
    """
    try:
        # Use adaptive sliding window if current_message is provided
        if current_message:
            conversation_history = get_adaptive_conversation_history(user_id, current_message, initial_limit=limit*2)
            logger.info(f"Using adaptive sliding window for message: '{current_message[:50]}...'")
        else:
            # Otherwise use the original method (for backward compatibility)
            conversation_history = get_conversation_history(user_id, limit=limit)
            logger.info("Using standard recency-based history retrieval")
        
        if not conversation_history:
            if format_as_text:
                return ""
            return []
        
        # If we're not using adaptive window, sort by timestamp
        if not current_message:
            conversation_history = sorted(
                conversation_history, 
                key=lambda x: x.get('metadata', {}).get('timestamp', 0), 
                reverse=True
            )
        
        # Limit to the requested number of items
        conversation_history = conversation_history[:limit]
        
        if not format_as_text:
            return conversation_history
        
        # Format the history as a conversation text
        context_history = ""
        for item in conversation_history:
            metadata = item.get('metadata', {})
            user_msg = metadata.get('message', '')
            bot_msg = metadata.get('response', '')
            timestamp = metadata.get('timestamp', 0)
            
            if user_msg and bot_msg:
                # Add timestamp for debugging if available
                time_str = ""
                if timestamp:
                    time_str = f" [{datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')}]"
                
                # Add the interaction to our context history
                context_history += f"User{time_str}: {user_msg}\nAssistant: {bot_msg}\n\n"
        
        logger.info(f"Retrieved and ranked {len(conversation_history)} conversation history items")
        return context_history
    
    except Exception as e:
        logger.error(f"Error in get_ranked_conversation_history: {str(e)}")
        if format_as_text:
            return ""
        return []

# Select the most relevant conversation history using Gemini
def select_relevant_history_with_agno(user_id, current_message, history_items, max_items=5):
    """
    Use Gemini with Agno to select the most relevant conversation history items based on
    timestamps and conversation flows.
    
    This function now works with the adaptive sliding window approach, which pre-filters
    and ranks history items by semantic relevance before they reach this function.
    
    Args:
        user_id (str): User ID
        current_message (str): Current user message/question
        history_items (list): List of conversation history items (pre-filtered by adaptive window)
        max_items (int): Maximum number of relevant items to select
        
    Returns:
        list: Selected relevant history items
    """
    try:
        if not history_items:
            return []
        
        # For very short messages (likely follow-ups), always include the most recent message
        is_short_followup = len(current_message.split()) <= 5
        
        # Check for pronouns that indicate a follow-up
        has_pronouns = any(pronoun in current_message.lower() for pronoun in ['it', 'this', 'that', 'they', 'them', 'these', 'those'])
        # Check for words that indicate temporal relationships with services
        has_after_words = any(word in current_message.lower() for word in ['after', 'before', 'during', 'following', 'between'])
        is_definite_followup = is_short_followup and (has_pronouns or has_after_words)
        
        # CRITICAL FIX: For definite follow-ups with pronouns, ALWAYS prioritize the MOST RECENT conversation
        # This ensures that when a user says "can I do it after X", the "it" refers to the most recently discussed service
        if is_definite_followup and history_items:
            # Get the most recent item's timestamp
            most_recent_timestamp = history_items[0].get('metadata', {}).get('timestamp', 0)
            
            # For definite follow-ups, ALWAYS use the most recent item as primary context
            most_recent = [history_items[0]]
            logger.info(f"CRITICAL: For definite follow-up with pronouns, ALWAYS using most recent conversation as primary context: '{current_message}'")
            
            # If we have at least 2 items, check if the second item is part of the same conversation session
            # (within a reasonable time window, e.g., 5 minutes)
            if len(history_items) >= 2:
                second_item_timestamp = history_items[1].get('metadata', {}).get('timestamp', 0)
                time_diff = most_recent_timestamp - second_item_timestamp
                
                # If the second item is within 5 minutes of the most recent, include it as part of the same conversation
                if time_diff <= 300:  # 5 minutes = 300 seconds
                    most_recent.append(history_items[1])
                    logger.info(f"Including second most recent item as part of the same conversation session (time diff: {time_diff}s)")
                    
                    if len(history_items) == 2:
                        return most_recent
                    remaining_items = history_items[2:]
                else:
                    logger.info(f"Second item is from a different conversation session (time diff: {time_diff}s)")
                    if len(history_items) == 1:
                        return most_recent
                    remaining_items = history_items[1:]
            else:
                if len(history_items) == 1:
                    return most_recent
                remaining_items = history_items[1:]
        else:
            # For non-follow-ups, we'll still include the most recent item but with less priority
            if history_items:
                most_recent = [history_items[0]]
                if len(history_items) == 1:
                    return most_recent
                remaining_items = history_items[1:]
            else:
                most_recent = []
                remaining_items = []
        
        # If there are no remaining items to rank, just return the most recent
        if not remaining_items:
            return most_recent
        
        # Create Agno agent for context selection
        try:
            # Create instructions for the context selection agent
            context_selection_instructions = dedent("""
                You are a context selection assistant for a conversational AI system.
                Your task is to analyze a user's current message and select the most relevant previous 
                conversation items that provide context for answering their question.
                
                SELECTION CRITERIA (IN ORDER OF PRIORITY):
                1. TEMPORAL RECENCY: The most recent conversation is ALWAYS the most relevant for pronoun resolution
                2. SEMANTIC RELEVANCE: Conversations about similar topics to the current message
                3. REFERENCE RESOLUTION: Conversations that help resolve pronouns or references in the current message
                4. CONVERSATION FLOW: Identify if the current message is continuing a previous conversation thread
                
                CRITICAL GUIDELINES FOR PRONOUN AND REFERENCE RESOLUTION:
                - When a user message contains pronouns like "it", "this", "that", these ALWAYS refer to something 
                  mentioned in the MOST RECENT conversation exchange (ITEM 1), not earlier conversations
                - For questions like "Can I go for it if I have X?" or "Can I do it after Y?", the pronoun "it" ALWAYS 
                  refers to the MAIN TOPIC/SERVICE in ITEM 1, not something from earlier exchanges
                - EXTREMELY IMPORTANT: When a user asks "can I do it after [treatment]" or "can I do it before [treatment]",
                  "it" ALWAYS refers to the main service discussed in ITEM 1, and they're asking about the compatibility
                  or timing of that service in relation to the treatment they mentioned
                - If ITEM 1 mentions multiple services/topics, the pronoun "it" refers to the PRIMARY service/topic 
                  that was the main focus of the conversation in ITEM 1
                - NEVER select older conversations over ITEM 1 for pronoun resolution
                - In your reasoning, EXPLICITLY state what you believe the pronoun is referring to from ITEM 1
                
                IMPORTANT GUIDELINES:
                - For short follow-up questions with pronouns (e.g., "What about this?", "Is it available?"), 
                  the most recent 1-2 messages are ALWAYS the most relevant
                - For questions containing "after" or "before" (e.g., "can I do it after X?"), ALWAYS select
                  the most recent conversation to determine what "it" refers to
                - For questions about specific services or products, find conversations that mention those items
                - For questions about preferences or personal information, find conversations where the user shared that info
                - Look for temporal indicators in the message (e.g., "as I mentioned earlier", "the service we discussed")
                - Pay attention to timestamps to identify conversation sessions (messages close in time)
                
                OUTPUT FORMAT:
                Return a JSON object with:
                1. "selected_items": Array of item numbers to include (e.g., [1, 3, 5])
                2. "reasoning": Brief explanation of why each item was selected
                3. "conversation_flow": Identified flow type ("continuation", "new_topic", "reference_to_earlier", etc.)
                
                Example:
                {
                  "selected_items": [1, 3, 5],
                  "reasoning": "Item 1 contains the most recent context about facial services, Item 3 mentions user preferences, Item 5 has details about pricing",
                  "conversation_flow": "continuation"
                }
            """)
            
            # Create the Agno agent with key rotation on failure
            context_agent = get_agno_agent_with_retry(instructions=context_selection_instructions)
            
            # Format history items with timestamps for better temporal analysis
            formatted_history = ""
            for i, item in enumerate(history_items):
                metadata = item.get('metadata', {})
                user_msg = metadata.get('message', '')
                bot_msg = metadata.get('response', '')
                timestamp = metadata.get('timestamp', 0)
                
                # Convert timestamp to readable format
                if timestamp:
                    time_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    time_str = "Unknown time"
                
                if user_msg and bot_msg:
                    formatted_history += f"ITEM {i+1} [Time: {time_str}]:\nUser: {user_msg}\nAssistant: {bot_msg}\n\n"
            
            # Create the prompt for the context agent
            context_prompt = dedent(f"""
                CURRENT USER MESSAGE: "{current_message}"
                
                CONVERSATION HISTORY (from newest to oldest):
                {formatted_history}
                
                Analyze the current message and select the {max_items - len(most_recent)} most relevant conversation items 
                that would provide the best context for understanding and responding to this message.
                
                CRITICAL INSTRUCTIONS FOR PRONOUN RESOLUTION:
                - For messages with pronouns like "it", "this", "that", these ALWAYS refer to something in ITEM 1 (the most recent conversation)
                - When a user asks "Can I go for it if I have X?" or "Can I do it after Y?", the pronoun "it" ALWAYS refers to the MAIN TOPIC/SERVICE in ITEM 1
                - SPECIFIC EXAMPLE: If the user asks "can i do it after microneedle therapy" and the most recent conversation was about Deep Tissue Massage,
                  then "it" refers to Deep Tissue Massage, and the user is asking if they can get a Deep Tissue Massage after microneedle therapy
                - If ITEM 1 mentions multiple services/topics, the pronoun "it" refers to the PRIMARY service/topic that was the main focus of ITEM 1
                - NEVER select older conversations over ITEM 1 for pronoun resolution
                - In your reasoning, EXPLICITLY state what you believe the pronoun is referring to from ITEM 1
                
                Consider:
                1. Is this a follow-up to a recent conversation? ({"Yes, definitely" if is_definite_followup else "Possibly" if is_short_followup else "Unclear"})
                2. Does it contain pronouns or references that need resolution? ({"Yes" if has_pronouns else "No"})
                3. What specific topics, services, or entities might it be referring to?
                4. Which previous conversations would provide the most helpful context?
                
                Return your analysis as a JSON object with the selected item numbers, reasoning, and identified conversation flow.
            """)
            
            # Get response from Agno agent
            max_retries = 2
            for retry in range(max_retries):
                try:
                    # If this is a retry, get a new agent
                    if retry > 0:
                        logger.info(f"Retrying context selection with new API key (attempt {retry+1})")
                        rotate_gemini_key()
                        context_agent = get_agno_agent(instructions=context_selection_instructions)
                    
                    # Get the response
                    response = context_agent.run(context_prompt)
                    response_text = response.content
                    
                    # Extract JSON from response
                    try:
                        # Try to parse the response as JSON directly
                        selection_data = json.loads(response_text)
                    except json.JSONDecodeError:
                        # If direct parsing fails, try to extract JSON from text
                        match = re.search(r'\{.*\}', response_text, re.DOTALL)
                        if match:
                            selection_data = json.loads(match.group(0))
                        else:
                            raise ValueError("Could not extract JSON from Agno response")
                    
                    # Get the selected items
                    selected_item_numbers = selection_data.get("selected_items", [])
                    conversation_flow = selection_data.get("conversation_flow", "unknown")
                    reasoning = selection_data.get("reasoning", "No reasoning provided")
                    
                    logger.info(f"Identified conversation flow: {conversation_flow}")
                    logger.info(f"Selection reasoning: {reasoning}")
                    
                    # Convert item numbers to actual items
                    selected_items = []
                    for item_num in selected_item_numbers:
                        try:
                            item_idx = int(item_num) - 1  # Convert to 0-based index
                            if 0 <= item_idx < len(history_items):
                                selected_items.append(history_items[item_idx])
                        except (ValueError, IndexError):
                            continue
                    
                    # CRITICAL FIX: For definite follow-ups with pronouns, FORCE the selection to prioritize the most recent conversation
                    if is_definite_followup and has_pronouns:
                        # Get the most recent item's timestamp
                        most_recent_timestamp = history_items[0].get('metadata', {}).get('timestamp', 0)
                        
                        # Filter selected items to only include those from the same conversation session (within 5 minutes)
                        recent_session_items = []
                        for item in selected_items:
                            item_timestamp = item.get('metadata', {}).get('timestamp', 0)
                            time_diff = abs(most_recent_timestamp - item_timestamp)
                            
                            # Only include items from the same conversation session (within 5 minutes)
                            if time_diff <= 300:  # 5 minutes = 300 seconds
                                recent_session_items.append(item)
                            else:
                                logger.info(f"Excluding item from different conversation session (time diff: {time_diff}s)")
                        
                        # Replace selected items with only those from the recent session
                        selected_items = recent_session_items
                        
                        # Log this critical fix
                        logger.info(f"CRITICAL FIX: For definite follow-up with pronouns, filtered selection to only include items from the same conversation session")
                    
                    # If we're dealing with a continuation of recent conversation,
                    # ensure temporal coherence by including items close in time
                    if conversation_flow == "continuation" and selected_items:
                        # Sort selected items by timestamp
                        selected_items.sort(
                            key=lambda x: x.get('metadata', {}).get('timestamp', 0),
                            reverse=True  # Newest first
                        )
                    
                    # Combine with the most recent items that we always include for follow-ups
                    final_selection = most_recent.copy()
                    
                    # Add selected items, avoiding duplicates
                    for item in selected_items:
                        if item not in final_selection and len(final_selection) < max_items:
                            final_selection.append(item)
                    
                    # If this is a definite follow-up with pronouns, ENSURE the most recent item is included
                    # and given highest priority, even if Agno didn't select it
                    if is_definite_followup and has_pronouns and history_items and history_items[0] not in final_selection:
                        # Insert the most recent item at the beginning
                        final_selection.insert(0, history_items[0])
                        logger.info(f"CRITICAL FIX: Forced inclusion of most recent item for pronoun resolution")
                        
                        # If we exceeded the max items, remove the least relevant one
                        if len(final_selection) > max_items:
                            final_selection.pop()
                    
                    # If we still have room and this is a definite follow-up,
                    # add more recent items to ensure context continuity
                    if is_definite_followup and len(final_selection) < max_items and len(history_items) > len(final_selection):
                        # Add more recent items that weren't already selected
                        for item in history_items:
                            if item not in final_selection and len(final_selection) < max_items:
                                final_selection.append(item)
                    
                    logger.info(f"Selected {len(final_selection)} most relevant history items using Agno")
                    return final_selection
                    
                except Exception as e:
                    logger.error(f"Error in Agno context selection (attempt {retry+1}): {str(e)}")
                    if retry < max_retries - 1:
                        continue
            
            # If Agno selection fails, fall back to simpler selection
            logger.warning("Agno context selection failed, falling back to basic selection")
            return select_relevant_history_fallback(current_message, history_items, max_items, most_recent, is_short_followup, has_pronouns)
            
        except Exception as e:
            logger.error(f"Error creating Agno agent for context selection: {str(e)}")
            return select_relevant_history_fallback(current_message, history_items, max_items, most_recent, is_short_followup, has_pronouns)
            
    except Exception as e:
        logger.error(f"Error in select_relevant_history_with_agno: {str(e)}")
        # Return the most recent items as fallback
        return history_items[:min(max_items, len(history_items))]

def select_relevant_history_fallback(current_message, history_items, max_items=5, most_recent=None, is_short_followup=False, has_pronouns=False):
    """
    Fallback method for selecting relevant history when Agno selection fails
    """
    if most_recent is None:
        most_recent = []
    
    # CRITICAL FIX: For definite follow-ups with pronouns, ALWAYS prioritize the most recent item
    if is_short_followup and has_pronouns and history_items:
        logger.info("CRITICAL FALLBACK: For definite follow-up with pronouns, ALWAYS prioritizing the most recent conversation")
        
        # Always include the most recent item as the primary context
        result = [history_items[0]]
        
        # If we have at least 2 items and there's room, check if the second item is part of the same conversation session
        if len(history_items) >= 2 and max_items > 1:
            # Get timestamps
            most_recent_timestamp = history_items[0].get('metadata', {}).get('timestamp', 0)
            second_item_timestamp = history_items[1].get('metadata', {}).get('timestamp', 0)
            
            # Calculate time difference
            time_diff = abs(most_recent_timestamp - second_item_timestamp)
            
            # If the second item is within 5 minutes of the most recent, include it as part of the same conversation
            if time_diff <= 300:  # 5 minutes = 300 seconds
                result.append(history_items[1])
                logger.info(f"Fallback: Including second most recent item as part of the same conversation session (time diff: {time_diff}s)")
                
                # Add more items if there's room, but only from the same conversation session
                if max_items > 2 and len(history_items) > 2:
                    for i in range(2, min(len(history_items), max_items)):
                        item_timestamp = history_items[i].get('metadata', {}).get('timestamp', 0)
                        time_diff = abs(most_recent_timestamp - item_timestamp)
                        
                        # Only include items from the same conversation session
                        if time_diff <= 300:  # 5 minutes = 300 seconds
                            result.append(history_items[i])
            else:
                # If the second item is from a different conversation, only include items from other sessions
                # if they're semantically relevant (which we can't determine in the fallback)
                logger.info(f"Fallback: Second item is from a different conversation session (time diff: {time_diff}s)")
                
                # Just add a few more recent items to provide some context
                result.extend(history_items[1:min(3, len(history_items))])
        
        return result[:max_items]
    
    # For regular short follow-ups, prioritize the most recent item plus a few more
    elif is_short_followup and len(history_items) > 1:
        # Always include the most recent item for short follow-ups
        return [history_items[0]] + history_items[1:min(max_items, len(history_items))]
    else:
        # Otherwise just return the most recent items
        return history_items[:max_items]

# Legacy function name for backward compatibility
def select_relevant_history(user_id, current_message, history_items, max_items=5):
    """
    Legacy function that calls the new Agno-based implementation
    """
    return select_relevant_history_with_agno(user_id, current_message, history_items, max_items)



