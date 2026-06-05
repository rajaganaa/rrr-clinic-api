"""
engine/buddhi.py — Buddhi: Core Reasoner using Groq (llama3-70b-8192)
Antahkarana MedAssist — CPU-only Azure deployment.

Buddhi is intellect/discriminative intelligence in Indian philosophy.
It performs structured multi-pass reasoning (Tarka → Pramana → Samsaya).

FIX 1 — Tamil language support:
  - detect_language() detects Tamil Unicode range (U+0B80–U+0BFF)
  - All system prompts have Tamil variants
  - reason() passes detected language through; final answer is in Tamil if input was Tamil

FIX 2 — Structured AI response format:
  - reason() now returns a 'structured_response' dict with consistent top-level keys:
      summary, uses, dosage, side_effects, warnings, notes
  - _build_structured_response() parses Buddhi's raw output into these keys
  - Falls back gracefully when the LLM doesn't use the expected format

Engine priority (unchanged):
  1. Groq (llama3-70b-8192)  — primary, fastest free LLM
  2. GitHub Models (gpt-4o)  — fallback if GROQ_API_KEY missing/fails

Return dict keys (extended):
  reasoning_steps, pass1_raw, pass1_answer, pass2_raw, pass2_answer,
  pass2_verified, pass2_fired, pass3_fired, draft_answer, latency_s, model,
  detected_language, structured_response   ← NEW

Author: RAJAGANAPATHY M, SRM University
"""

import os
import re
import logging
import time
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)

# ── Model configuration ───────────────────────────────────────────────────────
GROQ_MODEL   = os.environ.get("GROQ_MODEL",   "llama3-70b-8192")
GITHUB_MODEL = os.environ.get("GITHUB_MODEL", "gpt-4o")

# BUG 2 FIX (preserved): USE_VLLM is permanently False. No GPU on Azure.
USE_VLLM = False


# ── FIX 1: Language detection ─────────────────────────────────────────────────

TAMIL_UNICODE_RANGE = re.compile(r'[\u0B80-\u0BFF]')


def detect_language(text: str) -> str:
    tamil_chars = TAMIL_UNICODE_RANGE.findall(text)
    if len(tamil_chars) >= 3:
        return "ta"
    # Also detect if user explicitly requests Tamil
    if re.search(r'\bin tamil\b|\btamil\b|\bதமிழ்\b', text, re.IGNORECASE):
        return "ta"
    return "en"


# ── Groq Engine (PRIMARY) ─────────────────────────────────────────────────────

class GroqEngine:
    """
    Primary reasoning engine — Groq API with llama3-70b-8192.
    Uses the official `groq` Python library (groq>=0.9.0).
    """

    def __init__(self):
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. "
                "Add it to your .env or Azure portal settings."
            )
        self.client = Groq(api_key=api_key)
        self.model  = GROQ_MODEL
        logger.info(f"[BUDDHI] GroqEngine ready — model: {self.model}")

    def chat(self, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        """Call Groq API with clean message dicts."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()


# ── GitHub Models Fallback Engine ─────────────────────────────────────────────

class GitHubModelsEngine:
    """
    Fallback engine — GitHub Models (GPT-4o) via openai library.
    Used only when GROQ_API_KEY is missing or Groq API call fails.
    """

    def __init__(self):
        from openai import OpenAI
        token = os.environ.get("GITHUB_TOKEN", "")
        if not token:
            raise RuntimeError(
                "GITHUB_TOKEN environment variable is not set. "
                "Cannot use GitHub Models fallback engine."
            )
        self.client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=token,
        )
        self.model = GITHUB_MODEL
        logger.info(f"[BUDDHI] GitHubModelsEngine ready — model: {self.model}")

    def chat(self, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()


# ── Engine factory ────────────────────────────────────────────────────────────

_engine_instance = None


def _get_engine():
    """
    Return the best available engine.
    Priority: Groq → GitHub Models.
    vLLM is never attempted (no GPU on Azure).
    """
    global _engine_instance
    if _engine_instance is not None:
        return _engine_instance

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        try:
            _engine_instance = GroqEngine()
            logger.info("[BUDDHI] Using GroqEngine as primary engine")
            return _engine_instance
        except Exception as e:
            logger.warning(f"[BUDDHI] GroqEngine init failed ({e}) — trying GitHub Models")

    github_token = os.environ.get("GITHUB_TOKEN", "")
    if github_token:
        try:
            _engine_instance = GitHubModelsEngine()
            logger.info("[BUDDHI] Using GitHubModelsEngine as fallback engine")
            return _engine_instance
        except Exception as e:
            logger.error(f"[BUDDHI] GitHubModelsEngine init failed: {e}")

    raise RuntimeError(
        "No LLM engine available. Set GROQ_API_KEY (preferred) "
        "or GITHUB_TOKEN in your environment variables."
    )


# ── FIX 2: Structured response format helpers ─────────────────────────────────
#
# The structured_response dict returned by reason() always has these keys:
#   summary       — one-line answer / overview
#   uses          — list of indications
#   dosage        — dosage information string
#   side_effects  — list of side effects
#   warnings      — list of warnings / contraindications
#   notes         — any additional notes (Tamil note appended when lang == 'ta')
#
# The LLM is asked to emit these as labelled sections so _build_structured_response()
# can parse them reliably.  If a section is missing the parser falls back to the
# full draft_answer text in `summary` and empty lists/strings elsewhere.

STRUCTURED_FORMAT_INSTRUCTIONS_EN = """
Respond in this EXACT format (keep the labels):
SUMMARY: <one sentence answer>
USES: <comma-separated list of uses>
DOSAGE: <dosage information>
SIDE_EFFECTS: <comma-separated list>
WARNINGS: <comma-separated list>
NOTES: <any additional notes, or "None">
ANSWER: <your complete answer in plain prose>
"""

STRUCTURED_FORMAT_INSTRUCTIONS_TA = """
கீழ்கண்ட வடிவத்தில் தமிழிலேயே பதிலளிக்கவும் (லேபில்களை அப்படியே வையுங்கள்):
SUMMARY: <ஒரு வாக்கியத்தில் பதில்>
USES: <பயன்பாடுகள் (கமா பிரிக்கப்பட்டவை)>
DOSAGE: <அளவு தகவல்>
SIDE_EFFECTS: <பக்க விளைவுகள் (கமா பிரிக்கப்பட்டவை)>
WARNINGS: <எச்சரிக்கைகள் (கமா பிரிக்கப்பட்டவை)>
NOTES: <கூடுதல் குறிப்புகள் அல்லது "இல்லை">
ANSWER: <முழுமையான பதில் உரைநடையில்>
"""


def _build_structured_response(raw: str, draft_answer: str) -> Dict:
    """
    Parse the LLM's structured output into a consistent dict.

    Looks for labelled sections (SUMMARY:, USES:, DOSAGE:, SIDE_EFFECTS:,
    WARNINGS:, NOTES:).  Falls back gracefully if the LLM doesn't follow format.
    """
    def _extract_field(label: str) -> str:
        m = re.search(rf'{label}\s*:\s*(.+?)(?=\n[A-Z_]{{3,}}\s*:|$)', raw,
                      re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""

    def _to_list(text: str) -> List[str]:
        if not text:
            return []
        items = re.split(r',\s*|\n[-•*]\s*|\n\d+\.\s*', text)
        return [i.strip() for i in items if i.strip()]

    summary      = _extract_field("SUMMARY")      or draft_answer[:200]
    uses_raw     = _extract_field("USES")
    dosage       = _extract_field("DOSAGE")
    se_raw       = _extract_field("SIDE_EFFECTS")
    warn_raw     = _extract_field("WARNINGS")
    notes        = _extract_field("NOTES")

    return {
        "summary":      summary,
        "uses":         _to_list(uses_raw),
        "dosage":       dosage,
        "side_effects": _to_list(se_raw),
        "warnings":     _to_list(warn_raw),
        "notes":        notes if notes.lower() not in ("none", "இல்லை", "") else "",
    }


# ── Prompt Systems (Antahkarana philosophy preserved) ─────────────────────────

# FIX 1: Each system prompt now exists in English (EN) and Tamil (TA) variants.
# The _select_system() method picks the right one based on detected language.

_MEDICAL_BASE = (
    "You are MedAssist, a medical information assistant. "
    "If the provided context has relevant information, use it. "
    "If the context does NOT contain information about the medicine, "
    "use your own general medical knowledge to answer. "
    "Never say 'not in context' — always give a helpful answer. "
    "Always include: uses, dosage, side effects, warnings.\n\n"
)

MEDICAL_SYSTEM_EN = _MEDICAL_BASE + STRUCTURED_FORMAT_INSTRUCTIONS_EN

MEDICAL_SYSTEM_TA = (
    "நீங்கள் MedAssist, ஒரு மருத்துவ தகவல் உதவியாளர். "
    "கொடுக்கப்பட்ட சூழலில் தகவல் இருந்தால் பயன்படுத்தவும். "
    "இல்லையெனில் உங்கள் மருத்துவ அறிவைப் பயன்படுத்தவும். "
    "'சூழலில் இல்லை' என்று சொல்லாதீர்கள் — எப்போதும் உதவியான பதில் தரவும். "
    "எப்போதும் இவற்றை சேர்க்கவும்: பயன்பாடுகள், அளவு, பக்க விளைவுகள், எச்சரிக்கைகள்.\n\n"
    + STRUCTURED_FORMAT_INSTRUCTIONS_TA
)

DOSAGE_SYSTEM_EN = (
    "You are a medical information assistant helping with dosage information. "
    "Provide general information from medical guidelines only. "
    "Always recommend consulting a healthcare provider.\n\n"
    + STRUCTURED_FORMAT_INSTRUCTIONS_EN
    + "\nWARNINGS: Always consult your doctor or pharmacist before taking medications."
)

DOSAGE_SYSTEM_TA = (
    "நீங்கள் அளவு தகவல் வழங்கும் மருத்துவ உதவியாளர். "
    "மருத்துவ வழிகாட்டுதல்களின் பொதுவான தகவல்களை மட்டும் தரவும். "
    "எப்போதும் மருத்துவரை அணுகுமாறு பரிந்துரைக்கவும்.\n\n"
    + STRUCTURED_FORMAT_INSTRUCTIONS_TA
    + "\nWARNINGS: மருந்து எடுப்பதற்கு முன் உங்கள் மருத்துவர் அல்லது மருந்தாளரை ஆலோசிக்கவும்."
)

COMPARISON_SYSTEM_EN = (
    "You are a medical comparison expert. "
    "Compare medications based on the provided context.\n\n"
    + STRUCTURED_FORMAT_INSTRUCTIONS_EN
)

COMPARISON_SYSTEM_TA = (
    "நீங்கள் மருத்துவ ஒப்பீட்டு வல்லுநர். "
    "கொடுக்கப்பட்ட சூழலின் அடிப்படையில் மருந்துகளை ஒப்பிடவும்.\n\n"
    + STRUCTURED_FORMAT_INSTRUCTIONS_TA
)

VERIFICATION_SYSTEM_EN = (
    "You are a medical fact-checker. "
    "Determine if the claim is supported by the medical context.\n\n"
    "You MUST respond with one of: SUPPORTED / NOT SUPPORTED / INSUFFICIENT EVIDENCE\n\n"
    "Evidence assessment: <your reasoning>\n"
    "ANSWER: SUPPORTED or NOT SUPPORTED or INSUFFICIENT EVIDENCE"
)

VERIFICATION_SYSTEM_TA = (
    "நீங்கள் மருத்துவ உண்மை சரிபார்ப்பாளர். "
    "கோரிக்கை மருத்துவ சூழலால் ஆதரிக்கப்படுகிறதா என்று தீர்மானிக்கவும்.\n\n"
    "நீங்கள் இதில் ஒன்றை மட்டும் பதிலளிக்க வேண்டும்: SUPPORTED / NOT SUPPORTED / INSUFFICIENT EVIDENCE\n\n"
    "சான்று மதிப்பீடு: <உங்கள் பகுப்பாய்வு>\n"
    "ANSWER: SUPPORTED அல்லது NOT SUPPORTED அல்லது INSUFFICIENT EVIDENCE"
)

# Pramana — second-pass Tarka verification (language-agnostic; always English internally)
PRAMANA_SYSTEM = (
    "You are a strict medical fact verifier. "
    "Check if the draft answer is supported by the medical context.\n\n"
    "You MUST follow this exact format:\n"
    "Supported: yes\n"
    "Evidence: <quote from context, max 15 words>\n"
    "Revised answer: <repeat the draft answer>\n\n"
    "OR if NOT supported:\n"
    "Supported: no\n"
    "Evidence: <what context actually says, max 15 words>\n"
    "Revised answer: <corrected answer from context>"
)


# ── Buddhi Class ──────────────────────────────────────────────────────────────

class Buddhi:
    """
    Buddhi — The discriminative intellect.

    Multi-pass reasoning pipeline (Tarka → Pramana → Samsaya):
      Pass 1 (Tarka)   — initial LLM reasoning with context
      Pass 2 (Pramana) — verification against retrieved context
      Pass 3 (Samsaya) — self-consistency sampling if answer is bad

    FIX 1: Tamil language is detected from the question; system prompts
            and structured response are returned in the detected language.
    FIX 2: reason() now returns a 'structured_response' dict with
            consistent keys: summary, uses, dosage, side_effects, warnings, notes.
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
        """
        Run Antahkarana multi-pass reasoning.

        Returns dict with exactly these keys:
          reasoning_steps, pass1_raw, pass1_answer, pass2_raw, pass2_answer,
          pass2_verified, pass2_fired, pass3_fired, draft_answer, latency_s, model,
          detected_language, structured_response
        """
        t0 = time.time()

        # ── FIX 1: Detect language from the incoming question ────────────────
        detected_language = detect_language(question)
        logger.info(f"[BUDDHI] Detected language: {detected_language} for question: {question[:60]}")

        # Build vision context prefix (Chitta memory enrichment)
        vision_ctx = ""
        if medicine_info:
            name     = medicine_info.get("generic_name") or medicine_info.get("brand_name", "")
            strength = medicine_info.get("strength", "")
            form     = medicine_info.get("form", "")
            if name:
                vision_ctx = f"[Identified Medicine: {name} {strength} {form}]\n\n"

        full_context = f"{vision_ctx}{context_str}" if vision_ctx else context_str

        system      = self._select_system(q_type, detected_language)
        user_prompt = f"Medical Context:\n{full_context}\n\nQuestion: {question}"

        # ── Pass 1: Tarka (initial reasoning) ────────────────────────────────
        pass1_raw    = self.engine.chat(system, user_prompt, max_tokens=1024)
        pass1_answer = self._extract_answer(pass1_raw)
        pass1_steps  = self._extract_reasoning_steps(pass1_raw)

        # ── Pass 2: Pramana (verification against context) ───────────────────
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
            pass2_raw       = self.engine.chat(PRAMANA_SYSTEM, pramana_user, max_tokens=512)
            pass2_answer, pass2_verified = self._extract_pramana(pass2_raw, pass1_answer)
            pass2_fired     = True

        # ── Pass 3: Samsaya (self-consistency — only if answer still bad) ────
        pass3_fired  = False
        final_answer = pass2_answer

        if self._is_bad_answer(pass2_answer):
            candidates = []
            for _ in range(3):
                alt = self.engine.chat(system, user_prompt, max_tokens=512, temperature=0.7)
                candidates.append(self._extract_answer(alt))
            candidates = [c for c in candidates if not self._is_bad_answer(c)]
            if candidates:
                from collections import Counter
                best         = Counter(c.lower() for c in candidates).most_common(1)[0][0]
                final_answer = next((c for c in candidates if c.lower() == best), candidates[0])
                pass3_fired  = True

        elapsed = round(time.time() - t0, 3)

        # ── FIX 2: Build structured response from the best raw output ────────
        # Use pass1_raw as the source since it has the fullest structured content.
        structured_response = _build_structured_response(pass1_raw, final_answer)

        # Append Tamil disclaimer note when responding in Tamil
        if detected_language == "ta" and not structured_response.get("notes"):
            structured_response["notes"] = (
                "இந்த தகவல் பொதுவான நோக்கங்களுக்காக மட்டுமே. "
                "மருந்து எடுப்பதற்கு முன் மருத்துவரை கலந்தாலோசிக்கவும்."
            )

        return {
            "reasoning_steps":    pass1_steps,
            "pass1_raw":          pass1_raw,
            "pass1_answer":       pass1_answer,
            "pass2_raw":          pass2_raw if pass2_fired else None,
            "pass2_answer":       pass2_answer,
            "pass2_verified":     pass2_verified,
            "pass2_fired":        pass2_fired,
            "pass3_fired":        pass3_fired,
            "draft_answer":       final_answer,
            "latency_s":          elapsed,
            "model":              getattr(self._engine, "model", "unknown"),
            # ── NEW keys ──────────────────────────────────────────────────────
            "detected_language":  detected_language,        # 'ta' or 'en'
            "structured_response": structured_response,     # dict with 6 keys
        }

    def _select_system(self, q_type: str, lang: str = "en") -> str:
        """
        FIX 1: Pick the system prompt for (q_type, language) pair.
        Tamil ('ta') gets Tamil-language system prompts.
        All other languages fall back to English.
        """
        from engine.manas import QType

        is_tamil = (lang == "ta")

        system_map_en = {
            QType.VERIFICATION: VERIFICATION_SYSTEM_EN,
            QType.COMPARISON:   COMPARISON_SYSTEM_EN,
            QType.DOSAGE:       DOSAGE_SYSTEM_EN,
            QType.MATH:         DOSAGE_SYSTEM_EN,
        }
        system_map_ta = {
            QType.VERIFICATION: VERIFICATION_SYSTEM_TA,
            QType.COMPARISON:   COMPARISON_SYSTEM_TA,
            QType.DOSAGE:       DOSAGE_SYSTEM_TA,
            QType.MATH:         DOSAGE_SYSTEM_TA,
        }

        if is_tamil:
            return system_map_ta.get(q_type, MEDICAL_SYSTEM_TA)
        return system_map_en.get(q_type, MEDICAL_SYSTEM_EN)

    def _extract_answer(self, raw: str) -> str:
        m = re.search(r'ANSWER\s*:\s*(.+?)(?:\n|$)', raw, re.IGNORECASE)
        if m:
            ans = re.sub(r'[.,;:]+$', '', m.group(1).strip())
            return ans.strip()
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        return lines[-1] if lines else raw[:200]

    def _extract_reasoning_steps(self, raw: str) -> List[str]:
        steps = []
        m = re.search(r'Reasoning\s*:\s*(.+?)(?:ANSWER|WARNING|SUMMARY|$)', raw,
                      re.IGNORECASE | re.DOTALL)
        if m:
            numbered = re.split(r'\n(?=Step\s*\d+|^\d+\.)', m.group(1).strip(),
                                flags=re.MULTILINE)
            steps    = [s.strip() for s in numbered if s.strip()]
        if not steps:
            steps = [l.strip() for l in raw.split("\n") if l.strip()][:5]
        return steps

    def _extract_pramana(self, raw: str, draft: str) -> Tuple[str, bool]:
        """Parse Pramana verification response."""
        supported_m = re.search(r'Supported\s*:\s*(yes|no)',            raw, re.IGNORECASE)
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
        # Tamil apology patterns
        "தெரியவில்லை", "முடியாது", "தகவல் இல்லை",
    ]

    def _is_bad_answer(self, answer: str) -> bool:
        if not answer or len(answer.strip()) < 2:
            return True
        return any(p in answer.lower() for p in self.APOLOGY_PATTERNS)
