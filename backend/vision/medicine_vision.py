"""
vision/medicine_vision.py — Medicine image analysis
RRR Clinic MedAssist

Uses GPT-4o via GitHub Models (free with student pack).
Extracts: drug name, generic name, brand name, strength, form, expiry date.

Author: Rajaganapathy M — M.Tech AI, SRM University
"""

import os
import base64
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _encode_image_base64(image_path: str) -> tuple[str, str]:
    """Encode image to base64 and detect media type."""
    path     = Path(image_path)
    ext      = path.suffix.lower()
    type_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".gif":  "image/gif",
        ".webp": "image/webp",
    }
    media_type = type_map.get(ext, "image/jpeg")

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return encoded, media_type


def extract_medicine_info(image_path: str) -> dict:
    """
    Analyse a medicine image using GPT-4o via GitHub Models.

    Returns dict with keys:
      drug_name, generic_name, brand_name, strength, form,
      expiry_date, manufacturer, extraction_method, raw_response
    """
    token = os.environ.get("GITHUB_TOKEN", "")

    if not token:
        logger.warning("[VISION] GITHUB_TOKEN not set — returning empty vision result")
        return _empty_result("GITHUB_TOKEN not set")

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=token,
        )

        image_b64, media_type = _encode_image_base64(image_path)

        prompt = (
            "You are a pharmaceutical expert. Analyse this medicine image carefully.\n\n"
            "Extract and return ONLY a JSON object with these exact keys:\n"
            "{\n"
            '  "drug_name": "generic name if visible, else brand name",\n'
            '  "generic_name": "generic/chemical name or Not visible",\n'
            '  "brand_name": "brand/trade name or Not visible",\n'
            '  "strength": "e.g. 500mg or Not visible",\n'
            '  "form": "tablet/capsule/syrup/injection or Not visible",\n'
            '  "expiry_date": "MM/YYYY format or Not visible",\n'
            '  "manufacturer": "company name or Not visible"\n'
            "}\n\n"
            "Respond with JSON only. No markdown, no explanation."
        )

        response = client.chat.completions.create(
            model=os.environ.get("GITHUB_MODEL", "gpt-4o"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type":      "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=300,
            temperature=0.0,
        )

        raw = response.choices[0].message.content.strip()
        logger.info(f"[VISION] GPT-4o raw response: {raw[:120]}")

        result = _parse_vision_response(raw)
        result["extraction_method"] = "gpt4o-github-models"
        result["raw_response"]      = raw
        return result

    except Exception as e:
        logger.error(f"[VISION] GPT-4o extraction failed: {e}")
        return _empty_result(str(e))


def _parse_vision_response(raw: str) -> dict:
    """Parse JSON from GPT-4o response, with fallback to regex extraction."""
    import re, json

    # Strip markdown fences if present
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        data = json.loads(clean)
        return {
            "drug_name":    data.get("drug_name", "Not detected"),
            "generic_name": data.get("generic_name", "Not visible"),
            "brand_name":   data.get("brand_name", "Not visible"),
            "strength":     data.get("strength", "Not visible"),
            "form":         data.get("form", "Not visible"),
            "expiry_date":  data.get("expiry_date", "Not visible"),
            "manufacturer": data.get("manufacturer", "Not visible"),
        }
    except json.JSONDecodeError:
        # Fallback: regex extraction
        def _extract(key):
            m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', clean)
            return m.group(1) if m else "Not detected"

        return {
            "drug_name":    _extract("drug_name"),
            "generic_name": _extract("generic_name"),
            "brand_name":   _extract("brand_name"),
            "strength":     _extract("strength"),
            "form":         _extract("form"),
            "expiry_date":  _extract("expiry_date"),
            "manufacturer": _extract("manufacturer"),
        }


def _empty_result(reason: str) -> dict:
    """Return a safe empty result when vision fails."""
    return {
        "drug_name":          "Not detected",
        "generic_name":       "Not visible",
        "brand_name":         "Not visible",
        "strength":           "Not visible",
        "form":               "Not visible",
        "expiry_date":        "Not visible",
        "manufacturer":       "Not visible",
        "extraction_method":  "failed",
        "raw_response":       reason,
    }
