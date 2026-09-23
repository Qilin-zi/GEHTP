"""Tokenizer loader and chat template application.

Uses the ``tokenizers`` library when available and falls back to
HuggingFace transformers for model-specific tokenizers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

try:
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers import models, pre_tokenizers, trainers
    HAS_TOKENIZERS = True
except Exception:  # pragma: no cover - tokenizers optional fallback
    HAS_TOKENIZERS = False

try:
    from transformers import PreTrainedTokenizerFast
    HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover
    HAS_TRANSFORMERS = False


@dataclass
class TokenizerConfig:
    tokenizer_path: str
    chat_template: Optional[str] = None
    max_length: int = 4096


class TokenizerError(RuntimeError):
    """Raised when tokenizer loading or encoding fails."""


class ChatTokenizer:
    """Load a tokenizer from ``tokenizer.json`` and apply chat templates."""

    def __init__(self, config: TokenizerConfig) -> None:
        self.config = config
        self.tokenizer: Optional[HFTokenizer] = None
        self._loaded = False

    def load(self) -> None:
        """Load the tokenizer from disk."""
        path = Path(self.config.tokenizer_path)
        if not path.is_file():
            raise TokenizerError(f"tokenizer file not found: {path}")

        if HAS_TOKENIZERS:
            try:
                self.tokenizer = HFTokenizer.from_file(str(path))
                self._loaded = True
                return
            except Exception as exc:
                raise TokenizerError(f"failed to load tokenizer: {exc}") from exc
        raise TokenizerError("tokenizers library is required for loading tokenizer.json")

    @property
    def vocab_size(self) -> int:
        if self.tokenizer is not None:
            return self.tokenizer.get_vocab_size()
        return 0

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text into token IDs."""
        if not self._loaded:
            self.load()
        if self.tokenizer is None:
            raise TokenizerError("tokenizer not loaded")
        output = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return output.ids

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode token IDs back to text."""
        if not self._loaded:
            self.load()
        if self.tokenizer is None:
            raise TokenizerError("tokenizer not loaded")
        return self.tokenizer.decode(list(token_ids))

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool = True,
    ) -> str:
        """Apply a simple chat template from the model config.

        Falls back to concatenating message contents when no template
        is configured.
        """
        if self.config.chat_template:
            # Simple template substitution - full Jinja support requires
            # transformers or a Jinja2 runtime.
            template = self.config.chat_template
            for role, content in messages:
                text = content if isinstance(content, str) else str(content)
                template = template.replace("{{role}}", role).replace(
                    "{{content}}", text
                )
            if add_generation_prompt and "<｜end｜of｜messages｜>" in template:
                template += "<｜end｜of｜messages｜>"
            return template
        # Default: concatenate role + content
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        return "\n".join(parts)


def load_tokenizer(model_dir: str) -> ChatTokenizer:
    """Convenience loader that finds tokenizer.json in a model directory."""
    tokenizer_path = str(Path(model_dir) / "tokenizer.json")
    config = TokenizerConfig(tokenizer_path=tokenizer_path)
    tokenizer = ChatTokenizer(config)
    tokenizer.load()
    return tokenizer
