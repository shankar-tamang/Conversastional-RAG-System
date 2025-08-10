from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, delclarative_base



#SQLAlchemy database URL for SQLITE
SQLALCHEMY_DATABASE_URL = "sqlite:/// ./test.db"

# The database engine

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}


)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Base class to define ORM models
Base = declarative_base()

