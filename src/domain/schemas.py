from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional

class ProductRequest(BaseModel):
    """
    Schema for the input request.
    Validates that the input is a proper URL.
    """
    url: HttpUrl = Field(..., description="The target website URL to scrape")

class ProductResponse(BaseModel):
    """
    Schema for the output response.
    Returns the URL processed, the list of found products, and metadata.
    """
    url: str
    products: List[str] = Field(default_factory=list, description="List of extracted product names")
    product_count: int
    processing_time: float = Field(..., description="Time taken in seconds")
    error: Optional[str] = None