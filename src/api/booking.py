
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from src.database import get_db
from src.services.booking_service import BookingService
import datetime

router = APIRouter()

class BookingRequest(BaseModel):
    name: str
    email: EmailStr
    date: datetime.date
    time: datetime.time

@router.post("/bookings")
async def create_booking(
    request: BookingRequest,
    db: Session = Depends(get_db)
):
    """
    Schedules an interview, sends a confirmation email, and stores the booking information.
    """
    try:
        booking_service = BookingService(db)
        booking = await booking_service.create_booking(request.name, request.email, request.date, request.time)
        return {"message": "Booking created successfully", "booking_id": booking.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

