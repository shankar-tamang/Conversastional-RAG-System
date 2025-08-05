
from sqlalchemy.orm import Session
from src.models.booking import Booking
from src.email_sender import EmailSender
import datetime

class BookingService:
    def __init__(self, db: Session):
        self.db = db
        self.email_sender = EmailSender()

    def create_booking(self, name: str, email: str, date: datetime.date, time: datetime.time):
        # Create booking in DB
        booking = Booking(name=name, email=email, date=date, time=time.strftime("%H:%M"))
        self.db.add(booking)
        self.db.commit()
        self.db.refresh(booking)

        # Send confirmation email
        subject = "Interview Confirmation"
        body = f"Hi {name},\n\nYour interview is confirmed for {date} at {time}.\n\nThanks,\nThe PalmMind Team"
        self.email_sender.send_email(to_email=email, subject=subject, body=body)

        return booking
