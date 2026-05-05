"""
src/agents/scripting/rag/embeddings.py

Embedding classes for the multimodal RAG pipeline.

  TextEmbedder  — wraps OpenAI text-embedding-3-small via langchain-openai.
                  Used for Text chunks, Table summaries, AND Image captions.
                  This is the ONLY embedder used in the production pipeline.

  CLIPEmbedder  — DEPRECATED. Retained for reference only.
                  The pipeline now uses GPT-4o mini vision to caption images,
                  then embeds captions with TextEmbedder (same vector space
                  as text and tables). CLIPEmbedder is no longer called.
"""


from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# ──────────────────────────── TextEmbedder ────────────────────────────────────


class TextEmbedder:
    """
    Embed text using OpenAI text-embedding-3-small.

    Backed by langchain-openai ``OpenAIEmbeddings``.
    Produces 1536-dim float32 vectors.
    """

    DIMENSION = 1536
    MODEL = "text-embedding-3-small"

    def __init__(self, model: Optional[str] = None) -> None:
        import os
        backend = os.getenv("PROVIDER_BACKEND", "openai").lower()
        
        if backend == "nvidia":
            try:
                from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
            except ImportError as exc:
                raise ImportError(
                    "langchain-nvidia-ai-endpoints is required: pip install langchain-nvidia-ai-endpoints"
                ) from exc
            
            resolved_model = model or os.getenv("EMBEDDING_MODEL") or "nvidia/llama-3.2-nv-embedqa-1b-v2"
            self._client = NVIDIAEmbeddings(
                model=resolved_model,
                nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
                base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
            )
            logger.debug("TextEmbedder initialised with NVIDIA model=%s", resolved_model)
        else:
            try:
                from langchain_openai import OpenAIEmbeddings
            except ImportError as exc:
                raise ImportError(
                    "langchain-openai is required: pip install langchain-openai"
                ) from exc

            resolved_model = model or self.MODEL
            self._client = OpenAIEmbeddings(model=resolved_model)
            logger.debug("TextEmbedder initialised with OpenAI model=%s", resolved_model)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings. Returns one vector per text."""
        if not texts:
            return []
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._client.embed_query(text)


# ──────────────────────────── CLIPEmbedder ────────────────────────────────────


class CLIPEmbedder:
    """
    Embed images (and cross-modal text queries) using OpenAI's CLIP ViT-B/32.

    Requires the ``clip`` package from OpenAI:
        pip install git+https://github.com/openai/CLIP.git

    Produces 512-dim float32 L2-normalised vectors.

    IMPORTANT
    ---------
    Vectors produced here live in CLIP embedding space (512-dim).
    Do NOT mix with TextEmbedder vectors (1536-dim OpenAI space).
    """

    DIMENSION = 512
    DEFAULT_MODEL = "ViT-B/32"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu") -> None:
        try:
            import clip  # openai/CLIP
            import torch
        except ImportError as exc:
            raise ImportError(
                "OpenAI CLIP is required:\n"
                "  pip install git+https://github.com/openai/CLIP.git torch torchvision"
            ) from exc

        self._torch = torch
        self._device = device

        logger.info("CLIPEmbedder: loading model %s on %s …", model_name, device)
        self._model, self._preprocess = clip.load(model_name, device=device)
        self._model.eval()
        self._tokenize = clip.tokenize
        logger.info("CLIPEmbedder: ready.")

    # ── Image embedding ──────────────────────────────────────────────────────

    def embed_image_bytes(self, image_bytes: bytes) -> list[float]:
        """
        Embed raw image bytes (PNG/JPEG) into a 512-dim CLIP vector.

        Returns an empty list if the image cannot be processed.
        """
        try:
            from PIL import Image as PILImage
        except ImportError as exc:
            raise ImportError("Pillow is required: pip install Pillow") from exc

        try:
            pil_img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
            return self._embed_pil(pil_img)
        except Exception as exc:
            logger.warning("CLIPEmbedder.embed_image_bytes failed: %s", exc)
            return []

    def embed_pil_image(self, pil_image) -> list[float]:
        """Embed a PIL Image object."""
        try:
            return self._embed_pil(pil_image)
        except Exception as exc:
            logger.warning("CLIPEmbedder.embed_pil_image failed: %s", exc)
            return []

    def embed_images_batch(self, images_bytes: list[bytes]) -> list[list[float]]:
        """
        Embed a batch of raw image bytes.
        Failed images get an empty list placeholder.
        """
        results: list[list[float]] = []
        for img_bytes in images_bytes:
            results.append(self.embed_image_bytes(img_bytes))
        return results

    # ── Text query embedding (cross-modal) ───────────────────────────────────

    def embed_text_query(self, text: str) -> list[float]:
        """
        Encode a text string into CLIP text space (512-dim).

        This is used for cross-modal retrieval: the text query vector is
        compared against image vectors stored in the CLIP index.
        Must NOT be mixed with TextEmbedder.embed_query() results.
        """
        try:
            tokens = self._tokenize([text], truncate=True).to(self._device)
            with self._torch.no_grad():
                features = self._model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
            return features.squeeze(0).cpu().numpy().astype(np.float32).tolist()
        except Exception as exc:
            logger.warning("CLIPEmbedder.embed_text_query failed: %s", exc)
            return []

    # ── Internal ──────────────────────────────────────────────────────────────

    def _embed_pil(self, pil_image) -> list[float]:
        """Core PIL → CLIP vector conversion."""
        tensor = self._preprocess(pil_image).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            features = self._model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype(np.float32).tolist()
