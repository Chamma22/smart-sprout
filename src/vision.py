"""Identifies flowers from images using a three band confidence pipeline.

  1. High (>= HIGH_CONFIDENCE_THRESHOLD): return immediately.
  2. Mid (CLARIFY_LOWER_THRESHOLD .. HIGH_CONFIDENCE_THRESHOLD): return a clarifying
     question built from the top 2 candidates so the agent can ask the user before
     spending a Gemini call.
  3. Low (< CLARIFY_LOWER_THRESHOLD): Gemini Vision fallback with retry and backoff.
     If Gemini is unavailable, return the local model's best guess with a caveat.

Local model: dima806/oxford_flowers_image_detection (ViT fine tuned on Oxford 102).
"""
import base64
import os
import random
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

from load_env import load_local_env
load_local_env()

from transformers import pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


class GeminiUnavailable(Exception):
    """Raised when Gemini Vision can't be reached after all retries."""


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
LOCAL_MODEL_ID = "dima806/oxford_flowers_image_detection"
FALLBACK_PROCESSOR_ID = "google/vit-base-patch16-224"
HIGH_CONFIDENCE_THRESHOLD = 0.6
CLARIFY_LOWER_THRESHOLD = 0.4
MAX_RETRIES = 3
INITIAL_BACKOFF = 2.0

CLARIFY_TOKEN = "[CLARIFY]"

_classifier = None


def get_classifier():
    """Load the local model once and reuse it for subsequent calls.

    The dima806 repo is missing preprocessor_config.json, so newer transformers
    versions can't auto resolve the image processor. Fall back to the standard
    ViT processor in that case; architecture matches (ViT base 16 224).
    """
    global _classifier
    if _classifier is None:
        try:
            _classifier = pipeline("image-classification", model=LOCAL_MODEL_ID)
        except (ValueError, OSError):
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            processor = AutoImageProcessor.from_pretrained(FALLBACK_PROCESSOR_ID)
            model = AutoModelForImageClassification.from_pretrained(LOCAL_MODEL_ID)
            _classifier = pipeline("image-classification", model=model, image_processor=processor)
    return _classifier


def load_image_as_base64(image_path: str) -> tuple[str, str]:
    """Return (base64_string, mime_type) for a given image path."""
    ext = Path(image_path).suffix.lower()
    mime_type = "image/jpeg" if ext in {".jpg", ".jpeg"} else f"image/{ext.lstrip('.')}"
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime_type


def identify_with_gemini(image_path: str) -> str:
    """Identify a flower using Gemini Vision. Raise GeminiUnavailable on failure."""
    image_b64, mime_type = load_image_as_base64(image_path)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    message = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
        },
        {
            "type": "text",
            "text": (
                "Identify the flower in this image. "
                "Provide: (1) the common name, (2) the scientific name if you are confident, "
                "and (3) two or three key visual features you used to identify it. "
                "If you are not confident, say so and give your best guess."
            ),
        },
    ])

    backoff = INITIAL_BACKOFF
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = llm.invoke([message])
            return response.content
        except Exception as e:
            last_error = e
            if attempt == MAX_RETRIES:
                break
            print(f"  [Gemini Vision unavailable, retrying in {backoff:.0f}s... ({attempt}/{MAX_RETRIES})]")
            time.sleep(backoff + random.uniform(0, 1))
            backoff *= 2
    raise GeminiUnavailable(
        f"Gemini Vision failed after {MAX_RETRIES} retries. Last error: {last_error}"
    )


def identify_flower(image_path: str) -> str:
    """Run the three band flower identification pipeline on a single image."""
    ext = Path(image_path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image format '{ext}'. Use: {SUPPORTED_EXTENSIONS}")

    classifier = get_classifier()
    results = classifier(image_path, top_k=3)
    top = results[0]
    name = top["label"]
    score = top["score"]

    if score >= HIGH_CONFIDENCE_THRESHOLD:
        runner_up = ""
        if len(results) > 1:
            runner_up = f" (also considered: {results[1]['label']} at {results[1]['score']:.0%})"
        return f"This appears to be a {name} (confidence: {score:.0%}){runner_up}."

    if score >= CLARIFY_LOWER_THRESHOLD and len(results) >= 2:
        first, second = results[0], results[1]
        return (
            f"{CLARIFY_TOKEN} I'm not certain about this one, it could be a "
            f"{first['label']} ({first['score']:.0%}) or a {second['label']} ({second['score']:.0%}). "
            f"To help me decide: can you describe one or two distinctive features you see "
            f"(petal shape, how the flowers are arranged on the stem, color details)? "
            f"Image path was: {image_path}"
        )

    print(f"  [Local model confidence low ({score:.0%}), trying Gemini Vision for a better answer...]")
    try:
        return identify_with_gemini(image_path)
    except GeminiUnavailable:
        return (
            f"I'm not very confident about this one. My best guess is {name} ({score:.0%}), "
            f"but it may not be one of the 102 flowers I know well. "
            f"Try a clearer photo or a different angle for a better result."
        )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/vision.py <path_to_image>")
        sys.exit(1)
    print(identify_flower(sys.argv[1]))
