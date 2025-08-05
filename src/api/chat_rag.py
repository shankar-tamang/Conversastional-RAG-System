
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from src.services.rag_service import build_graph
from langchain_core.messages import HumanMessage

router = APIRouter()

graph = build_graph()

class ChatRequest(BaseModel):
    session_id: str
    query: str

@router.post("/chat")
async def chat(request: ChatRequest):
    # Pass session_id and messages into the graph's initial state
    inputs = {
        "messages": [HumanMessage(content=request.query)],
        "session_id": request.session_id
    }
    config = {"configurable": {"thread_id": request.session_id}}

    final_state = await graph.ainvoke(inputs, config=config)
    return {"response": final_state["messages"][-1].content}

