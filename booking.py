import os
import re
import json
import logging
from textwrap import dedent
import agno
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.sql import SQLTools
from config import get_gemini_model, gemini_api_keys
from memory import store_memory, MEMORY_TYPES
from agent_init import get_agno_agent_with_retry, get_agno_agent, rotate_gemini_key, rotate_gemini_key_and_model
from date_filter import extract_date_filter, format_date_for_postgresql

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Handle booking with Agno
def handle_booking_with_agno(user_id, message, intents, service_entities, date_entities=None, time_entities=None):
    """
    Handle booking using Agno agent

    Args:
        user_id (str): User ID
        message (str): User message
        intents (list): Detected intents
        service_entities (list): Detected service entities
        date_entities (list, optional): Detected date entities
        time_entities (list, optional): Detected time entities

    Returns:
        str: Response message from Agno agent
    """
    try:
        # Create database connection string
        db_url = f"postgresql+psycopg://{os.getenv('user1')}:{os.getenv('password1')}@{os.getenv('host1')}:{os.getenv('port1')}/{os.getenv('dbname1')}"
        
        # Create SQL tools for Agno
        sql_tools = SQLTools(db_url=db_url)
        
        # Create booking agent with specific instructions
        booking_instructions = dedent("""
            You are a booking assistant for Red Trends Spa & Wellness Center.

            IMPORTANT BOOKING RULES:
            1. Bookings must be made at least 4 hours in advance
            2. Available time slots are from 9:00 AM to 7:00 PM, every hour
            3. Each service has a 1 hour duration and price
            4. The user must provide: service name, date, and time
            5. Check if the requested time slot is available before confirming by checking bookings table for any existing booking of any service in that particular day and time slot
            6. CRITICAL: When working with dates in SQL queries, ALWAYS use the PostgreSQL date format 'YYYY-MM-DD'
               - CORRECT: WHERE "Booking Date" = '2025-05-23'
               - INCORRECT: WHERE "Booking Date" = 'may 23'
            6. If no year mentioned, always use 2025 for all bookings
            DATABASE SCHEMA (EXACT NAMES - DO NOT MODIFY):
            - Table name: public.services (not spa_services)
              Columns: "Service ID", "Service Name", "Description", "Price (INR)", "Category"
              Note: Column names include spaces and must be quoted in SQL queries

            - Table name: public.bookings
              Columns: "Booking ID", "Customer Name", "Service ID", "Booking Date", "Time Slot (HH:MM)", "Price (INR)"
              Note: Column names include spaces and must be quoted in SQL queries

            EXAMPLE CORRECT SQL QUERIES:
            - SELECT * FROM public.services WHERE "Service Name" = 'Glowing Facial'
            - SELECT * FROM public.bookings WHERE "Booking Date" = '2025-04-07'
            - SELECT "Service ID", "Service Name", "Price (INR)" FROM public.services WHERE "Service ID" = 1
            - INSERT INTO public.bookings ("Customer Name", "Service ID", "Booking Date", "Time Slot (HH:MM)", "Price (INR)")
              VALUES ('user123', 1, '2025-04-07', '14:00', 1500.00)

            BOOKING PROCESS:
            1. Ask for any missing information (service, date, time) unless the user provides a direct booking request.
            2. Use the user's ID as the "Customer Name" when creating bookings.
            3. Check if the service exists in the database using the EXACT table and column names then use Service Name to get Service ID.
            4. IMPORTANT: Fetch the price from the services database using the Service ID - use the "Price (INR)" column value.
            5. Check if the requested time slot is available.
            6. Create the booking in the database using the user's ID as the "Customer Name" and include the price fetched from the services table.
            7. Confirm the booking details to the user including the price.
            
            In case of a direct booking request (for example, if the user says "can you book"), process immediately and return a success confirmation message formatted as:
            "Your booking for <Service Name> on <Date> at <Time> has been successfully created. The total price is INR <price>."
            
            Always be helpful, professional, and provide clear instructions.
        """)
        
        # Create the Agno agent with key rotation on failure
        try:
            booking_agent = get_agno_agent_with_retry(instructions=booking_instructions, tools=[sql_tools])
        except Exception as e:
            logger.error(f"Failed to create booking agent after retries: {str(e)}")
            return "I'm sorry, but I'm having trouble with the booking system right now. Please try again later or contact us directly at (555) 123-4567 to make a booking."

        # Prepare the prompt for the booking agent
        booking_prompt = f"User wants to book a spa service. The user's ID is '{user_id}' - use this as the 'Customer Name' when creating the booking. "
        if service_entities:
            service_names = ", ".join(service_entities)
            booking_prompt += f"They mentioned these services: {service_names}. "
        if date_entities:
            dates = ", ".join(date_entities)
            booking_prompt += f"They mentioned these dates: {dates}. "
        if time_entities:
            times = ", ".join(time_entities)
            booking_prompt += f"They mentioned these times: {times}. "
        booking_prompt += f"Here's what they said: '{message}'. Handle this booking request appropriately."

        # Check if the user's message is a direct booking instruction.
        direct_booking = "can you book" in message.lower()

        # If the request is direct and required information is presumed to be provided,
        # then we bypass additional clarifications and directly query the database.
        if direct_booking:
            # For this example, we assume the service, date, and time are exactly as follows.
            # In a real-world scenario, you might extract these details using entity recognition.
            service_name = service_entities[0]
            booking_date = date_entities[0]
            booking_time = time_entities[0]
            # Build SQL query to get the service details (price and Service ID)
            service_query = f"SELECT \"Service ID\", \"Price (INR)\" FROM public.services WHERE \"Service Name\" = '{service_name}'"
            service_result = json.loads(sql_tools.run_sql_query(service_query))  # Use run_sql_query instead of execute

            if not service_result:
                return f"Sorry, the service '{service_name}' is not available."
            
            # For simplicity, assume service_result is a dict-like object or a row with keys "Service ID" and "Price (INR)"
            service_id = service_result[0]["Service ID"]
            price = service_result[0]["Price (INR)"]

            # Build SQL query to check if the time slot is available
            booking_check_query = f"SELECT * FROM public.bookings WHERE \"Booking Date\" = '{booking_date}' AND \"Time Slot (HH:MM)\" = '{booking_time}'"
            booking_check_result = json.loads(sql_tools.run_sql_query(booking_check_query))  # Use run_sql_query instead of execute
            if booking_check_result:
                return f"Sorry, the time slot {booking_time} on {booking_date} is already booked. Please choose another time."

            # Build SQL query to insert the booking
            booking_insert_query = (
                f"INSERT INTO public.bookings (\"Customer Name\", \"Service ID\", \"Booking Date\", \"Time Slot (HH:MM)\", \"Price (INR)\") "
                f"VALUES ('{user_id}', {service_id}, '{booking_date}', '{booking_time}', {price})"
            )
            # For INSERT queries, we still use run_sql_query but need to handle the result differently
            sql_tools.run_sql_query(booking_insert_query)
            # Since INSERT doesn't return rows, we can consider it successful if no exception was raised
            insertion_result = True
            if insertion_result:  # Since we set insertion_result to True if no exception was raised
                return f"Your booking for {service_name} on {booking_date} at {booking_time} has been successfully created. The total price is INR {price}."
            else:
                return "I'm sorry, there was an error processing your booking. Please try again later."

        # If not a direct booking request, use the agent logic with retry.
        max_retries = min(2, len(gemini_api_keys))
        for retry in range(max_retries):
            try:
                # If this is a retry, get a new agent with a fresh API key
                if retry > 0:
                    logger.info(f"Retrying booking agent with new API key (attempt {retry+1})")
                    rotate_gemini_key()
                    booking_agent = get_agno_agent(instructions=booking_instructions, tools=[sql_tools])

                response = booking_agent.run(booking_prompt)
                agent_response = response.content

                if agent_response is None:
                    raise ValueError("Agno agent returned None response")

                logger.info(f"Booking interaction - User: {user_id}, Message: {message}, Response: {agent_response[:100]}...")
                return agent_response
            except Exception as inner_e:
                logger.error(f"Error getting response from Agno agent (attempt {retry+1}/{max_retries}): {str(inner_e)}")
                if retry < max_retries - 1:
                    continue
                else:
                    return "I'd be happy to help you book a spa service. Could you please tell me which service you're interested in, what date, and what time you prefer? Alternatively, you can contact us directly at (555) 123-4567 to make a booking."
    except Exception as e:
        logger.error(f"Error handling booking with Agno: {str(e)}")
        return "I'm sorry, but I'm having trouble with the booking system right now. Please try again later or contact us directly at (555) 123-4567 to make a booking."




def handle_booking_query_with_agno(user_id, message, query_type="all", customer_name=None):
    """
    Handle booking query using Agno agent with SQLTools.
    The agent will analyze the user's natural language request and generate the appropriate SQL query.
    
    Args:
        user_id (str): User ID.
        message (str): User message.
        query_type (str): Type of bookings to query - "previous", "future", or "all".
        customer_name (str): Optional customer name to filter bookings.
    
    Returns:
        str: Response message from Agno agent.
    """
    try:
        # Build the database URL
        db_url = f"postgresql+psycopg://{os.getenv('user')}:{os.getenv('password')}@" \
                 f"{os.getenv('host')}:{os.getenv('port')}/{os.getenv('dbname')}"
        
        # Instantiate SQLTools with the given db_url
        sql_tools = SQLTools(db_url=db_url)
        
        # Create the instructions for the Agno agent
        booking_query_instructions = dedent("""
            You are a booking assistant for Red Trends Spa & Wellness Center.
            Your job is to analyze the user's natural language request and craft an appropriate SQL query
            to retrieve their booking information from the database.
            YOU CAN ONLY SEARCH FOR THE CURRENT USER.
            DATABASE SCHEMA (DO NOT MODIFY):
            - Table: public.services
              Columns: "Service ID", "Service Name", "Description", "Price (INR)", "Category"

            - Table: public.bookings
              Columns: "Booking ID", "Customer Name", "Service ID", "Booking Date", "Time Slot (HH:MM)", "Price (INR)"
              Note: The "Customer Name" column contains the user ID.
              Note: The "Booking Date" column is of DATE type and must be in 'YYYY-MM-DD' format.

            QUERY ANALYSIS GUIDELINES:
            1. Carefully analyze the user's request to determine:
               - If they want past bookings, future bookings, or all bookings
               - If they're asking about a specific time period (year, month, date range)
               - Any other specific filters they might be requesting
               - If user specifies a specific month get all bookings of that month, 2025
                                            
            2. Pay special attention to time-related words:
               - "previous", "past", "earlier", "history" → past bookings (before CURRENT_DATE)
               - "future", "upcoming", "next", "scheduled" → future bookings (on or after CURRENT_DATE)
               - "year 2025", "in 2025", "for 2025" → bookings in that specific year
               - "previous bookings in 2025" → past bookings in that specific year
               - "month", "April", "May", etc. → bookings in that specific month
            
            3. When a user asks for "previous" or "past" bookings for a specific year:
               - They want bookings that are both in that year AND in the past
               - Example: "previous bookings for 2025" means bookings in 2025 that are before CURRENT_DATE
               - Use: EXTRACT(YEAR FROM b."Booking Date") = 2025 AND b."Booking Date" < CURRENT_DATE

            SQL QUERY REQUIREMENTS:
            1. Always join public.bookings and public.services on "Service ID" to obtain the "Service Name".
            2. Always filter by "Customer Name" using the provided user id.
            3. Use appropriate date filters based on the user's request:
               - For past/previous bookings: b."Booking Date" < CURRENT_DATE
               - For future/upcoming bookings: b."Booking Date" >= CURRENT_DATE
               - For a specific year: EXTRACT(YEAR FROM b."Booking Date") = YYYY
               - For a specific month: TO_CHAR(b."Booking Date", 'YYYY-MM') = 'YYYY-MM'
               - For date ranges: appropriate comparison operators
            4. Order results appropriately:
               - Past bookings: ORDER BY b."Booking Date" DESC, b."Time Slot (HH:MM)"
               - Future bookings: ORDER BY b."Booking Date", b."Time Slot (HH:MM)"
            
            RESPONSE FORMAT:
            1. A summary of the bookings retrieved in a numbered list format:
               1. Service: **[Service Name]**, Date: **[Booking Date in Month Day, Year format]**, Time: [Time Slot], Price: [Price]
               2. Service: **[Service Name]**, Date: **[Booking Date in Month Day, Year format]**, Time: [Time Slot], Price: [Price]
               (and so on for each booking)
            2. For future bookings, include a reminder about the cancellation policy.
            3. For previous bookings, ask if the user would like to rebook the same service.
            4. Use friendly, conversational language in your response.
            5. DO NOT use exact database column names in your response - use user-friendly terms instead.
            
            IMPORTANT NOTES:
            - Always convert dates to a user-friendly format (e.g., "May 1st, 2025" instead of "2025-05-01")
            - If no bookings are found, provide a clear message and suggest booking options
            - If the query is ambiguous, err on the side of providing more information rather than less
        """)

        # Create the agent with a key rotation mechanism
        try:
            booking_query_agent = get_agno_agent_with_retry(
                instructions=booking_query_instructions,
                tools=[sql_tools]
            )
        except Exception as e:
            logger.error(f"Failed to create booking query agent after retries: {str(e)}")
            return ("I'm sorry, but I'm having trouble accessing the booking system right now. "
                    "Please try again later or contact us directly at (555) 123-4567 for booking information.")

        # Build the prompt for the agent
        booking_query_prompt = dedent(f"""
            User message: "{message}"
            
            User ID: {customer_name} (use this exact value for the "Customer Name" column in your SQL query)
            
            Query type hint: {query_type} (but analyze the user's message to determine the actual query type)
            
            Please analyze this request carefully, generate an appropriate SQL query, and provide a helpful response
            about the user's bookings. Remember to check for time-related words and apply the correct date filters.
            
            If the user specifically asked for "previous" or "past" bookings, make sure to only include bookings
            before the current date, even if they specified a year or other time period.
        """)

        # Attempt to get a response from the Agno agent
        max_retries = min(3, len(gemini_api_keys) * 2)
        for retry in range(max_retries):
            try:
                if retry > 0:
                    logger.info(f"Retrying booking query agent with new API key/model (attempt {retry+1})")
                    rotate_gemini_key_and_model()
                    booking_query_agent = get_agno_agent(instructions=booking_query_instructions, tools=[sql_tools])
                
                response = booking_query_agent.run(booking_query_prompt)
                agent_response = response.content
                if agent_response is None:
                    raise ValueError("Agno agent returned None response")
                
                logger.info(f"Booking query interaction - User: {user_id}, Query type: {query_type}, "
                            f"Response: {agent_response[:100]}...")
                return agent_response
            except Exception as inner_e:
                logger.error(f"Error from Agno agent (attempt {retry+1}/{max_retries}): {str(inner_e)}")
                if retry < max_retries - 1:
                    continue
                else:
                    return ("I'm sorry, but I'm having trouble retrieving your bookings at the moment. "
                            "Please try again later or contact us directly at (555) 123-4567 for booking information.")

    except Exception as e:
        logger.error(f"Error handling booking query with Agno: {str(e)}")
        return ("I'm sorry, but I'm having trouble accessing the booking system right now. "
                "Please try again later or contact us directly at (555) 123-4567 for booking information.")
