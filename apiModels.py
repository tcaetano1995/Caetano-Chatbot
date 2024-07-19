from pydantic import BaseModel
from typing import List, Optional, Any


class InputModel(BaseModel):
    input: str

class OutputModel(BaseModel):
    output: str



###openllama api 
class ModelDetails(BaseModel):
    parent_model: str
    format: str
    family: str
    families: List[str]
    parameter_size: str
    quantization_level: str

class ModelInfo(BaseModel):
    name: str
    model: str
    details: ModelDetails
    digest: str
    modified_at: str
    size: int



class ModelsResponse(BaseModel):
    models: List[ModelInfo]

class VersionResponse(BaseModel):
    version: str


class Message(BaseModel):
    role: str
    content: str
    image: Optional[str] = None  # Optional image property


#class ChatRequest(BaseModel):
#    stream: bool
#    model: str
#    messages: conlist(Message)
#    options: Dict[str, Any]

class ChatRequest(BaseModel):
    stream: bool
    model: str
    messages: List[Message]
    options: Optional[Any] = {}


class ChatResponse(BaseModel):
    model: str
    created_at: str
    message: Message
    done: bool
