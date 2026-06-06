"""
engine/buddhi.py — Buddhi: Core Reasoner using Groq (llama3-70b-8192)
RRR Clinic MedAssist — CPU-only Azure deployment.

Buddhi is intellect/discriminative intelligence in Indian philosophy.
Multi-pass reasoning: Tarka → Pramana → Samsaya

MERGED IMPROVEMENTS:
  ✅ Tamil language detection (Unicode + keyword "in tamil")
  ✅ Structured response format (SUMMARY/USES/DOSAGE/SIDE_EFFECTS/WARNINGS/NOTES)
  ✅ Tamil + English system prompt variants for all question types
  ✅ Increased max_tokens to 2048 for complete answers (fixes cut-off issue)
  ✅ ANSWER extracted with multiline support (fixes incomplete answers)
  ✅ Graceful fallback: Groq → GitHub Models

Author: RAJAGANAPATHY M, SRM University | Patent: 202641043947
"""

import os
import re
import logging
import time
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)

GROQ_MODEL   = os.environ.get("GROQ_MODEL",   "llama3-70b-8192")
GITHUB_MODEL = os.environ.get("GITHUB_MODEL", "gpt-4o")
USE_VLLM     = False  # No GPU on Azure. Hardcoded forever.


# ── Language detection ────────────────────────────────────────────────────────

TAMIL_UNICODE_RANGE = re.compile(r'[\u0B80-\u0BFF]')


def detect_language(text: str) -> str:
    """
    Returns 'ta' if Tamil Unicode chars >= 3 OR if 'tamil'/'in tamil' keyword present.
    Returns 'en' otherwise.
    """
    if len(TAMIL_UNICODE_RANGE.findall(text)) >= 3:
        return "ta"
    if re.search(r'\bin tamil\b|\btamil\b|\bதமிழ்\b', text, re.IGNORECASE):
        return "ta"
    return "en"


# ── Engines ───────────────────────────────────────────────────────────────────

class GroqEngine:
    def __init__(self):
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set.")
        self.client = Groq(api_key=api_key)
        self.model  = GROQ_MODEL
        logger.info(f"[BUDDHI] GroqEngine ready — {self.model}")

    def chat(self, system: str, user: str, max_tokens: int = 2048, temperature: float = 0.0) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()


class GitHubModelsEngine:
    def __init__(self):
        from openai import OpenAI
        token = os.environ.get("GITHUB_TOKEN", "")
        if not token:
            raise RuntimeError("GITHUB_TOKEN not set.")
        self.client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=token,
        )
        self.model = GITHUB_MODEL
        logger.info(f"[BUDDHI] GitHubModelsEngine ready — {self.model}")

    def chat(self, system: str, user: str, max_tokens: int = 2048, temperature: float = 0.0) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()


_engine_instance = None


def _get_engine():
    global _engine_instance
    if _engine_instance is not None:
        return _engine_instance
    if os.environ.get("GROQ_API_KEY"):
        try:
            _engine_instance = GroqEngine()
            return _engine_instance
        except Exception as e:
            logger.warning(f"[BUDDHI] Groq failed: {e}")
    if os.environ.get("GITHUB_TOKEN"):
        try:
            _engine_instance = GitHubModelsEngine()
            return _engine_instance
        except Exception as e:
            logger.error(f"[BUDDHI] GitHub Models failed: {e}")
    raise RuntimeError("No LLM engine available. Set GROQ_API_KEY or GITHUB_TOKEN.")


# ── Structured format instructions ───────────────────────────────────────────

STRUCTURED_FORMAT_EN = """
You MUST respond using EXACTLY this format — do not skip any label:

SUMMARY: <one clear sentence describing the medicine or answering the question>
USES: <comma-separated list of medical uses/indications>
DOSAGE: <specific dosage — adults: X mg, children: Y mg, frequency, max daily>
SIDE_EFFECTS: <comma-separated list of common side effects>
WARNINGS: <comma-separated list of warnings and contraindications>
NOTES: <additional important notes, or "None">
ANSWER: <complete, detailed prose answer combining all of the above>

IMPORTANT: Always complete ALL sections. Never leave DOSAGE or ANSWER empty.
"""

STRUCTURED_FORMAT_TA = """
நீங்கள் கட்டாயமாக இந்த வடிவத்தில் பதிலளிக்க வேண்டும் — எந்த லேபிலையும் தவிர்க்காதீர்கள்:

SUMMARY: <மருந்தை ஒரு தெளிவான வாக்கியத்தில் விவரிக்கவும்>
USES: <மருத்துவ பயன்பாடுகள் — கமாவால் பிரிக்கவும்>
DOSAGE: <குறிப்பிட்ட அளவு — பெரியவர்கள்: X mg, குழந்தைகள்: Y mg, எத்தனை முறை, அதிகபட்சம்>
SIDE_EFFECTS: <பொதுவான பக்க விளைவுகள் — கமாவால் பிரிக்கவும்>
WARNINGS: <எச்சரிக்கைகள் மற்றும் தடைகள் — கமாவால் பிரிக்கவும்>
NOTES: <கூடுதல் முக்கியமான குறிப்புகள் அல்லது "இல்லை">
ANSWER: <மேற்கண்ட அனைத்தையும் உள்ளடக்கிய முழுமையான தமிழ் பதில்>

முக்கியம்: அனைத்து பிரிவுகளையும் நிரப்பவும். DOSAGE மற்றும் ANSWER காலியாக இருக்கக்கூடாது.
"""

# ── System prompts ────────────────────────────────────────────────────────────

_BASE_EN = (
    "You are MedAssist, an expert medical information assistant for RRR Clinic, Tamil Nadu. "
    "ALWAYS use your medical knowledge to give complete, helpful answers. "
    "If the context doesn't mention the medicine, use your own knowledge — never say 'not in context'. "
    "Always provide: uses, dosage (with specific amounts), side effects, and warnings.\n\n"
)

_BASE_TA = (
    "நீங்கள் MedAssist, RRR மருத்துவமனை, தமிழ்நாட்டிற்கான நிபுணர் மருத்துவ தகவல் உதவியாளர். "
    "எப்போதும் உங்கள் மருத்துவ அறிவைப் பயன்படுத்தி முழுமையான, உதவியான பதில்களை தரவும். "
    "சூழலில் மருந்து இல்லாவிட்டாலும், உங்கள் அறிவைப் பயன்படுத்தவும் — 'சூழலில் இல்லை' என்று சொல்லாதீர்கள். "
    "எப்போதும் இவற்றை தரவும்: பயன்பாடுகள், அளவு (குறிப்பிட்ட அளவுகளுடன்), பக்க விளைவுகள், எச்சரிக்கைகள்.\n\n"
)

MEDICAL_SYSTEM_EN   = _BASE_EN + STRUCTURED_FORMAT_EN
MEDICAL_SYSTEM_TA   = _BASE_TA + STRUCTURED_FORMAT_TA

DOSAGE_SYSTEM_EN = (
    "You are a medical dosage expert for RRR Clinic. "
    "Always give SPECIFIC dosage amounts — never vague answers. "
    "Include: adult dose, child dose, frequency, maximum daily dose, and when to take.\n\n"
    + STRUCTURED_FORMAT_EN
)

DOSAGE_SYSTEM_TA = (
    "நீங்கள் RRR மருத்துவமனையின் மருந்தளவு நிபுணர். "
    "எப்போதும் குறிப்பிட்ட அளவுகளை தரவும் — தெளிவற்ற பதில்கள் வேண்டாம். "
    "இவற்றை சேர்க்கவும்: பெரியவர் அளவு, குழந்தை அளவு, எத்தனை முறை, அதிகபட்ச தினசரி அளவு.\n\n"
    + STRUCTURED_FORMAT_TA
)

COMPARISON_SYSTEM_EN = (
    "You are a medical comparison expert. Compare the medications clearly.\n\n"
    + STRUCTURED_FORMAT_EN
)

COMPARISON_SYSTEM_TA = (
    "நீங்கள் மருத்துவ ஒப்பீட்டு நிபுணர். மருந்துகளை தெளிவாக ஒப்பிடவும்.\n\n"
    + STRUCTURED_FORMAT_TA
)

VERIFICATION_SYSTEM_EN = (
    "You are a medical fact-checker. Respond ONLY with: SUPPORTED / NOT SUPPORTED / INSUFFICIENT EVIDENCE\n\n"
    "Evidence assessment: <reasoning>\nANSWER: SUPPORTED or NOT SUPPORTED or INSUFFICIENT EVIDENCE"
)

VERIFICATION_SYSTEM_TA = (
    "நீங்கள் மருத்துவ உண்மை சரிபார்ப்பாளர். இதில் மட்டும் பதிலளிக்கவும்: SUPPORTED / NOT SUPPORTED / INSUFFICIENT EVIDENCE\n\n"
    "சான்று மதிப்பீடு: <பகுப்பாய்வு>\nANSWER: SUPPORTED அல்லது NOT SUPPORTED அல்லது INSUFFICIENT EVIDENCE"
)

PRAMANA_SYSTEM = (
    "You are a strict medical fact verifier.\n\n"
    "Format:\nSupported: yes/no\nEvidence: <max 15 words from context>\nRevised answer: <answer>"
)


# ── Structured response parser ────────────────────────────────────────────────

def _build_structured_response(raw: str, draft_answer: str) -> Dict:
    def _field(label: str) -> str:
        m = re.search(rf'{label}\s*:\s*(.+?)(?=\n[A-Z_]{{3,}}\s*:|$)', raw,
                      re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""

    def _list(text: str) -> List[str]:
        if not text:
            return []
        return [i.strip() for i in re.split(r',\s*|\n[-•*]\s*|\n\d+\.\s*', text) if i.strip()]

    # Extract ANSWER field with multiline support
    answer_m = re.search(r'ANSWER\s*:\s*(.+?)$', raw, re.IGNORECASE | re.DOTALL)
    full_answer = answer_m.group(1).strip() if answer_m else draft_answer

    return {
        "summary":      _field("SUMMARY") or draft_answer[:200],
        "uses":         _list(_field("USES")),
        "dosage":       _field("DOSAGE"),
        "side_effects": _list(_field("SIDE_EFFECTS")),
        "warnings":     _list(_field("WARNINGS")),
        "notes":        _field("NOTES") if _field("NOTES").lower() not in ("none", "இல்லை", "") else "",
        "full_answer":  full_answer,  # complete prose answer for display
    }


# ── Buddhi Class ──────────────────────────────────────────────────────────────

class Buddhi:
    """
    Buddhi — The discriminative intellect.
    Tarka (Pass 1) → Pramana (Pass 2) → Samsaya (Pass 3 if needed)
    """

    def __init__(self):
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            self._engine = _get_engine()
        return self._engine

    def reason(
        self,
        question: str,
        context_str: str,
        q_type: str,
        medicine_info: Optional[Dict] = None,
    ) -> dict:

        t0 = time.time()
        detected_language = detect_language(question)
        logger.info(f"[BUDDHI] lang={detected_language} q={question[:60]}")

        # Vision context enrichment
        vision_ctx = ""
        if medicine_info:
            name     = medicine_info.get("generic_name") or medicine_info.get("brand_name", "")
            strength = medicine_info.get("strength", "")
            form     = medicine_info.get("form", "")
            if name and name not in ["Not detected", "Not visible"]:
                vision_ctx = f"[Identified Medicine: {name} {strength} {form}]\n\n"

        full_context = f"{vision_ctx}{context_str}" if vision_ctx else context_str
        system       = self._select_system(q_type, detected_language)
        user_prompt  = f"Medical Context:\n{full_context}\n\nQuestion: {question}"

        # ── Pass 1: Tarka ─────────────────────────────────────────────────────
        pass1_raw    = self.engine.chat(system, user_prompt, max_tokens=2048)
        pass1_answer = self._extract_answer(pass1_raw)
        pass1_steps  = self._extract_reasoning_steps(pass1_raw)

        # ── Pass 2: Pramana ───────────────────────────────────────────────────
        pass2_raw      = ""
        pass2_answer   = pass1_answer
        pass2_verified = True
        pass2_fired    = False

        if full_context.strip() and not self._is_bad_answer(pass1_answer):
            pramana_user = (
                f"Question: {question}\n\n"
                f"Draft answer: {pass1_answer}\n\n"
                f"Medical Context:\n{full_context[:2000]}"
            )
            pass2_raw     = self.engine.chat(PRAMANA_SYSTEM, pramana_user, max_tokens=512)
            pass2_answer, pass2_verified = self._extract_pramana(pass2_raw, pass1_answer)
            pass2_fired   = True

        # ── Pass 3: Samsaya ───────────────────────────────────────────────────
        pass3_fired  = False
        final_answer = pass2_answer

        if self._is_bad_answer(pass2_answer):
            candidates = []
            for _ in range(3):
                alt = self.engine.chat(system, user_prompt, max_tokens=1024, temperature=0.7)
                candidates.append(self._extract_answer(alt))
            candidates = [c for c in candidates if not self._is_bad_answer(c)]
            if candidates:
                from collections import Counter
                best         = Counter(c.lower() for c in candidates).most_common(1)[0][0]
                final_answer = next((c for c in candidates if c.lower() == best), candidates[0])
                pass3_fired  = True

        elapsed = round(time.time() - t0, 3)

        # Build structured response
        structured_response = _build_structured_response(pass1_raw, final_answer)

        # Use full_answer from structured parse if available (fixes cut-off)
        if structured_response.get("full_answer"):
            final_answer = structured_response["full_answer"]

        # Tamil disclaimer
        if detected_language == "ta" and not structured_response.get("notes"):
            structured_response["notes"] = (
                "இந்த தகவல் பொதுவான நோக்கங்களுக்காக மட்டுமே. "
                "மருந்து எடுப்பதற்கு முன் மருத்துவரை கலந்தாலோசிக்கவும்."
            )

        return {
            "reasoning_steps":     pass1_steps,
            "pass1_raw":           pass1_raw,
            "pass1_answer":        pass1_answer,
            "pass2_raw":           pass2_raw if pass2_fired else None,
            "pass2_answer":        pass2_answer,
            "pass2_verified":      pass2_verified,
            "pass2_fired":         pass2_fired,
            "pass3_fired":         pass3_fired,
            "draft_answer":        final_answer,
            "latency_s":           elapsed,
            "model":               getattr(self._engine, "model", "unknown"),
            "detected_language":   detected_language,
            "structured_response": structured_response,
        }

    def _select_system(self, q_type: str, lang: str = "en") -> str:
        from engine.manas import QType
        is_ta = (lang == "ta")
        if is_ta:
            return {
                QType.VERIFICATION: VERIFICATION_SYSTEM_TA,
                QType.COMPARISON:   COMPARISON_SYSTEM_TA,
                QType.DOSAGE:       DOSAGE_SYSTEM_TA,
                QType.MATH:         DOSAGE_SYSTEM_TA,
            }.get(q_type, MEDICAL_SYSTEM_TA)
        return {
            QType.VERIFICATION: VERIFICATION_SYSTEM_EN,
            QType.COMPARISON:   COMPARISON_SYSTEM_EN,
            QType.DOSAGE:       DOSAGE_SYSTEM_EN,
            QType.MATH:         DOSAGE_SYSTEM_EN,
        }.get(q_type, MEDICAL_SYSTEM_EN)

    def _extract_answer(self, raw: str) -> str:
        # Multiline ANSWER extraction — fixes cut-off issue
        m = re.search(r'ANSWER\s*:\s*(.+?)$', raw, re.IGNORECASE | re.DOTALL)
        if m:
            return re.sub(r'[.,;:]+$', '', m.group(1).strip()).strip()
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        return lines[-1] if lines else raw[:500]

    def _extract_reasoning_steps(self, raw: str) -> List[str]:
        m = re.search(r'Reasoning\s*:\s*(.+?)(?:ANSWER|WARNING|SUMMARY|$)', raw,
                      re.IGNORECASE | re.DOTALL)
        if m:
            steps = re.split(r'\n(?=Step\s*\d+|^\d+\.)', m.group(1).strip(), flags=re.MULTILINE)
            return [s.strip() for s in steps if s.strip()]
        return [l.strip() for l in raw.split("\n") if l.strip()][:5]

    def _extract_pramana(self, raw: str, draft: str) -> Tuple[str, bool]:
        supported_m = re.search(r'Supported\s*:\s*(yes|no)', raw, re.IGNORECASE)
        revised_m   = re.search(r'Revised answer\s*:\s*(.+?)(?:\n|$)', raw, re.IGNORECASE)
        if revised_m and supported_m and supported_m.group(1).lower() == "no":
            revised = revised_m.group(1).strip()
            if revised and not self._is_bad_answer(revised):
                return revised, False
        return draft, True

    APOLOGY_PATTERNS = [
        "i don't know", "i cannot", "i am not sure", "no information",
        "sorry", "cannot determine", "unclear", "i'm not certain",
        "not available", "i'm not aware", "not aware of any",
        "தெரியவில்லை", "முடியாது", "தகவல் இல்லை",
    ]

    def _is_bad_answer(self, answer: str) -> bool:
        if not answer or len(answer.strip()) < 10:
            return True
        return any(p in answer.lower() for p in self.APOLOGY_PATTERNS)