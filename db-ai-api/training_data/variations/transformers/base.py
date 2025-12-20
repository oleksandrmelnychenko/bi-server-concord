"""Base transformer class for query variations."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseTransformer(ABC):
    """Abstract base class for all query transformers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Transformer name for identification."""
        pass

    @abstractmethod
    def transform(self, text: str, language: str = "uk") -> List[str]:
        """
        Transform input text into variations.

        Args:
            text: Input text to transform
            language: Language code ('uk' or 'en')

        Returns:
            List of transformed variations
        """
        pass

    def can_transform(self, text: str, language: str = "uk") -> bool:
        """
        Check if this transformer can meaningfully transform the text.

        Args:
            text: Input text
            language: Language code

        Returns:
            True if transformation would produce meaningful results
        """
        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return transformer metadata."""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
        }
