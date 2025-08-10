from pydantic import BaseModel


class UserBase(BaseModel):
    name: str
    email: str

class UserCreate(UserBase):
    pass

class UserRead(UserBase):
    id: int

    class config:
        orm_mode = True
    