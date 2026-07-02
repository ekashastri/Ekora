"""
Service layer interfaces for AI and OCR capabilities.
"""
from typing import Protocol, Optional
import numpy as np

class IOCRProvider(Protocol):
    """Abstraction for Optical Character Recognition services."""
    def extract_text(self, image: np.ndarray) -> str:
        ...

class IAIProvider(Protocol):
    """Abstraction for AI models (e.g. LLMs, Vision models)."""
    def generate_diagram(self, prompt: str, image: Optional[np.ndarray] = None) -> str:
        """Returns Mermaid or vector representation of a generated diagram."""
        ...
