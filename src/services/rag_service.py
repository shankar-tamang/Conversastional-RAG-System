# rag_pipeline.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence, Optional, Literal
import operator
from src.pinecone_client import PineconeClient
from src.redis_client import RedisClient
import json
import re
import datetime
from src.services.booking_service import BookingService
from src.database import get_db
from pydantic import BaseModel, Field

# --- Pydantic Models for Structured Output ---
class Intent(BaseModel):
    """Determine the user's intent based on the conversation."""
    intent: Literal["booking", "retrieval", "greeting"] = Field(
        ...,
        description="The user's primary intent. Choose 'booking' if they are trying to schedule, are in the middle of scheduling, or are providing details for an appointment. Choose 'retrieval' for general questions. Choose 'greeting' for simple hellos.",
    )

# --- RAG State Definition ---
class RAGState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    context: Optional[str]
    session_id: str
    booking_details: Optional[dict]

# --- Clients and LLM Initialization ---
client = PineconeClient(index_name="llama-text-embed-v2-index", namespace="default")
redis_client = RedisClient()

def get_llm():
    return ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")

def get_intent_llm():
    """Returns an LLM configured for structured intent detection."""
    return ChatGoogleGenerativeAI(model="models/gemini-2.5-flash").with_structured_output(Intent)

# --- Session Management with Redis ---
def load_session_data(session_id: str) -> dict:
    """Loads session data from Redis, handling both old (list) and new (dict) formats."""
    session_data = redis_client.get(session_id)
    if session_data:
        data = json.loads(session_data)
        
        # Check if data is in the old list format
        if isinstance(data, list):
            # Convert old format to new format
            messages = [HumanMessage(**msg) if msg.get('type') == 'human' else AIMessage(**msg) for msg in data]
            return {"messages": messages, "booking_details": {}}
        
        # Process new dictionary format
        messages = [HumanMessage(**msg) if msg.get('type') == 'human' else AIMessage(**msg) for msg in data.get("messages", [])]
        booking_details = data.get("booking_details", {})
        return {"messages": messages, "booking_details": booking_details}
        
    return {"messages": [], "booking_details": {}}

def save_session_data(session_id: str, messages: Sequence[BaseMessage], booking_details: dict):
    """Saves session data (messages and booking details) to Redis."""
    data = {
        "messages": [msg.dict() for msg in messages],
        "booking_details": booking_details
    }
    redis_client.set(session_id, json.dumps(data))

# --- Graph Nodes ---

def load_session(state: RAGState) -> dict:
    """Node to load session data into the state."""
    session_data = load_session_data(state["session_id"])
    current_messages = list(state["messages"])
    all_messages = session_data["messages"] + current_messages
    return {
        "messages": all_messages,
        "booking_details": session_data["booking_details"]
    }

def greeting(state: RAGState) -> dict:
    response = AIMessage(content="Hi! How can I help you today?")
    save_session_data(state["session_id"], state["messages"] + [response], state.get("booking_details", {}))
    return {"messages": [response]}

def retrieve_context(state: RAGState) -> dict:
    last_msg = state["messages"][-1]
    vector_db = PineconeClient(index_name="llama-text-embed-v2-index", namespace="example-namespace")
    docs = vector_db.query(query=last_msg.content, top_k=3)
    context_str = "\n\n".join([hit['fields']['text'] for hit in docs['result']['hits']])
    return {
        "context": context_str,
        "messages": state["messages"],
        "booking_details": state.get("booking_details")
    }

def generate_response(state: RAGState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use this context:\n{context}"),
        MessagesPlaceholder(variable_name="messages")
    ])
    chain = prompt | get_llm()
    response = chain.invoke({
        "context": state["context"],
        "messages": state["messages"]
    })
    save_session_data(state["session_id"], state["messages"] + [response], state.get("booking_details", {}))
    return {"messages": [response]}

def handle_booking(state: RAGState) -> dict:
    """Handles the multi-turn booking conversation."""
    booking_details = state.get("booking_details") or {}
    last_msg_content = state["messages"][-1].content

    # --- Information Extraction ---
    if not booking_details.get("name"):
        # Simple name extraction (can be improved)
        match = re.search(r"my name is (\w+)", last_msg_content, re.IGNORECASE)
        if match:
            booking_details["name"] = match.group(1)
        else: # Assume the message is the name if we're expecting it
             if not any(kw in last_msg_content.lower() for kw in ["book", "schedule", "appointment", "@"]):
                booking_details["name"] = last_msg_content

    if not booking_details.get("email"):
        match = re.search(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", last_msg_content)
        if match:
            booking_details["email"] = match.group(0)

    if not booking_details.get("date"):
        match = re.search(r"(\d{4}-\d{2}-\d{2})", last_msg_content)
        if match:
            booking_details["date"] = match.group(0)

    if not booking_details.get("time"):
        match = re.search(r"(\d{1,2}:\d{2}\s*(?:am|pm)?)", last_msg_content, re.IGNORECASE)
        if match:
            booking_details["time"] = match.group(0).strip()


    # --- Logic to Ask for Missing Information ---
    if not booking_details.get("name"):
        response = AIMessage(content="Great, I can help with that. What is your full name?")
    elif not booking_details.get("email"):
        response = AIMessage(content=f"Thanks, {booking_details['name']}. What is your email address?")
    elif not booking_details.get("date"):
        response = AIMessage(content="Got it. What date would you like to book? (e.g., YYYY-MM-DD)")
    elif not booking_details.get("time"):
        response = AIMessage(content="Perfect. And what time? (e.g., 2:30pm)")
    else:
        # --- All details collected, create booking ---
        db = next(get_db())
        booking_service = BookingService(db)
        try:
            # Convert time to 24-hour format
            time_str = booking_details["time"]
            time_obj = datetime.datetime.strptime(time_str.upper().replace(" ",""), "%I:%M%p").time()

            date_obj = datetime.datetime.strptime(booking_details["date"], "%Y-%m-%d").date()

            booking_service.create_booking(
                name=booking_details["name"],
                email=booking_details["email"],
                date=date_obj,
                time=time_obj
            )
            response = AIMessage(content="Your booking is confirmed! You will receive a confirmation email shortly.")
            booking_details = {} # Clear details after successful booking
        except Exception as e:
            response = AIMessage(content=f"Sorry, I encountered an error: {e}. Please try again.")
            booking_details = {} # Clear details on error

    save_session_data(state["session_id"], state["messages"] + [response], booking_details)
    return {"messages": [response], "booking_details": booking_details}


# --- Conditional Routing ---

def detect_intent(state: RAGState) -> str:
    """Uses an LLM to detect the user's intent."""
    intent_llm = get_intent_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert routing assistant. Your job is to determine the user's intent based on the conversation history. "
                "The possible intents are: 'booking', 'retrieval', or 'greeting'.\n"
                "If the user is asking to schedule, providing details for an appointment (like their name, email, date, or time), "
                "or is in a conversation that seems to be leading to a booking, choose 'booking'.\n"
                "If the user is asking a general question, choose 'retrieval'.\n"
                "If the user is just saying 'hi' or 'hello', choose 'greeting'.\n"
                "Analyze the last message in the context of the entire conversation."
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | intent_llm
    result = chain.invoke({"messages": state["messages"]})
    print(f"Detected Intent: {result.intent}") # For debugging
    return result.intent

def booking_finished(state: RAGState) -> str:
    """Checks if the booking process is complete."""
    booking_details = state.get("booking_details", {})
    # If the dictionary is empty, the process is finished (either completed or failed).
    if not booking_details:
        return "end"
    return "continue"

# --- Graph Construction ---
def build_graph():
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("load_session", load_session)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("greeting", greeting)
    workflow.add_node("booking", handle_booking)

    # Define entry and routing
    workflow.set_entry_point("load_session")
    workflow.add_conditional_edges(
        "load_session",
        detect_intent,
        {
            "greeting": "greeting",
            "retrieval": "retrieve_context",
            "booking": "booking",
        },
    )

    # Define edges
    workflow.add_edge("retrieve_context", "generate_response")
    workflow.add_edge("generate_response", END)
    workflow.add_edge("greeting", END)

    # Conditional loop for booking
    workflow.add_conditional_edges(
        "booking",
        booking_finished,
        {
            "end": END,
            "continue": END, # End the current run, wait for next user input
        },
    )

    graph = workflow.compile()
    return graph