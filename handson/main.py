from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

from . import models, schemas, crud
from .database import SessionLocal, engine

# Create tables

models.Base.metadata.create_all(bind=engine)


app = FastAPI()


# Dependency: get DB session per request

def get_db():
    db = SessionLocal()

    try: 
        yield db
    
    finally: 
        db.close()


@app.post("/users/", response_model=schemas.UserRead)
def create_user(user: schemas.UserCreate, db: Session =  Depends(get_db)):
    return crud.create_user(db, user)

    
@app.get("/users/{user_id}", response_model=schemas.UserRead)
def read_user(user_id: int, db: Session = Depends(get_db)):
    return crud.get_usre(db, user_id)


