
from sqlalchemy import Column, String, Integer, DateTime
from src.database import Base

class Booking(Base):
    __tablename__ = "bookings"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    date = Column(DateTime)
    time = Column(String)

