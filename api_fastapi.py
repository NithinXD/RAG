from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import logging
from app import process_message, handle_booking_with_agno, analyze_message, classify_booking_message, handle_booking_query_with_agno

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Tranquility Spa & Wellness Center API",
    description="API for the Tranquility Spa & Wellness Center chatbot and booking system",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models
class ChatRequest(BaseModel):
    user_id: str
    message: str

class BookingRequest(BaseModel):
    user_id: str
    message: str
    service: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    intents: List[str]
    entities: Dict[str, List[str]]

class BookingResponse(BaseModel):
    response: str
    booking_status: str
    booking_details: Dict[str, Optional[str]]

class ErrorResponse(BaseModel):
    error: str

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "An internal server error occurred"}
    )

# Health check endpoint
@app.get("/api/health", tags=["Health"])
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok"}

# Chat endpoint
@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Process a chat message from the user
    """
    try:
        logger.info(f"Received message from user {request.user_id}: {request.message}")

        # Process the message - this will use Agno agent internally when appropriate
        response = process_message(request.user_id, request.message)

        # Get intents and entities for frontend context
        analysis = analyze_message(request.message)

        # The response is already the direct output from Agno agent when available
        # We just need to format it for the API response
        return ChatResponse(
            response=response,
            intents=analysis.get("intents", []),
            entities={
                "services": analysis.get("service_entities", []),
                "dates": analysis.get("date_entities", []),
                "times": analysis.get("time_entities", [])
            }
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

# Booking endpoint
@app.post("/api/booking", response_model=BookingResponse, tags=["Booking"])
async def booking(request: BookingRequest):
    """
    Handle a booking request directly
    """
    try:
        logger.info(f"Received booking request from user {request.user_id}: {request.message}")

        # Analyze the message for intents and entities
        analysis = analyze_message(request.message)

        # Add explicitly provided details if not found in message
        service_entities = analysis.get("service_entities", [])
        if request.service and request.service not in service_entities:
            service_entities.append(request.service)

        date_entities = analysis.get("date_entities", [])
        if request.date and request.date not in date_entities:
            date_entities.append(request.date)

        time_entities = analysis.get("time_entities", [])
        if request.time and request.time not in time_entities:
            time_entities.append(request.time)

        # Classify the booking message type using Gemini with Agno
        booking_type = classify_booking_message(request.message)
        logger.info(f"Booking message classified as: {booking_type}")

        # Process the booking based on classification
        if booking_type == "booking_retrieval":
            # Determine the type of booking query
            intents = analysis.get("intents", [])
            if "previous_bookings" in intents or any(word in request.message.lower() for word in ["previous", "past", "earlier", "before", "history"]):
                query_type = "previous"
            elif "future_bookings" in intents or any(word in request.message.lower() for word in ["future", "upcoming", "next", "scheduled", "coming"]):
                query_type = "future"
            else:
                query_type = "all"
            
            # Handle the booking query with Agno
            booking_response = handle_booking_query_with_agno(
                request.user_id, 
                request.message, 
                query_type, 
                request.user_id
            )
        else:  # booking_type == "create_booking"
            # Process the booking creation with Agno
            booking_response = handle_booking_with_agno(
                request.user_id,
                request.message,
                analysis.get("intents", []),
                service_entities,
                date_entities,
                time_entities
            )

        # Determine booking status from response
        booking_status = "pending"
        if "confirmed" in booking_response.lower() or "booked" in booking_response.lower():
            booking_status = "confirmed"
        elif "unavailable" in booking_response.lower() or "not available" in booking_response.lower():
            booking_status = "unavailable"

        # Extract booking details from entities
        booking_details = {
            "service": service_entities[0] if service_entities else None,
            "date": date_entities[0] if date_entities else None,
            "time": time_entities[0] if time_entities else None
        }

        # The booking_response is already the direct output from Agno agent or Grok
        # We just need to format it for the API response
        return BookingResponse(
            response=booking_response,
            booking_status=booking_status,
            booking_details=booking_details
        )

    except Exception as e:
        logger.error(f"Error processing booking request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing booking request: {str(e)}")

# Run the server
if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 5000))
    
    # Run the FastAPI app with uvicorn
    uvicorn.run("api_fastapi:app", host="0.0.0.0", port=port, reload=True)