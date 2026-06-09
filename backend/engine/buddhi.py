"""
engine/buddhi.py — Buddhi: Core Reasoner using Groq (llama-3.3-70b-versatile)
Anbu Clinic MedAssist — CPU-only Azure deployment.
Author: RAJAGANAPATHY M, SRM University | Patent: 202641043947
"""

import os, re, logging, time
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)

GROQ_MODEL   = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GITHUB_MODEL = os.environ.get("GITHUB_MODEL", "gpt-4o")
USE_VLLM     = False

TAMIL_UNICODE_RANGE = re.compile(r'[\u0B80-\u0BFF]')

def detect_language(text: str) -> str:
    if len(TAMIL_UNICODE_RANGE.findall(text)) >= 3:
        return "ta"
    if re.search(r'\bin tamil\b|\btamil\b|\bதமிழ்\b', text, re.IGNORECASE):
        return "ta"
    return "en"

class GroqEngine:
    def __init__(self):
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set.")
        self.client = Groq(api_key=api_key)
        self.model  = GROQ_MODEL
        logger.info(f"[BUDDHI] GroqEngine ready — {self.model}")

    def chat(self, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens, temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

class GitHubModelsEngine:
    def __init__(self):
        from openai import OpenAI
        token = os.environ.get("GITHUB_TOKEN", "")
        if not token:
            raise RuntimeError("GITHUB_TOKEN not set.")
        self.client = OpenAI(base_url="https://models.inference.ai.azure.com", api_key=token)
        self.model = GITHUB_MODEL
        logger.info(f"[BUDDHI] GitHubModelsEngine ready — {self.model}")

    def chat(self, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens, temperature=temperature,
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
    raise RuntimeError("No LLM engine available.")

# ── CONCISE structured format — fixes paragraph dump issue ────────────────────
STRUCTURED_FORMAT_EN = """
Respond in EXACTLY this format. Be CONCISE — no long paragraphs:

SUMMARY: <ONE sentence only>
USES: <max 4 items, comma-separated>
DOSAGE: <adults: Xmg every Xhrs, max Xmg/day | children: Xmg/kg>
SIDE_EFFECTS: <max 4 items, comma-separated>
WARNINGS: <max 3 items, comma-separated>
NOTES: None
ANSWER: <2-3 sentences max. Key facts only. No repetition.>
"""

STRUCTURED_FORMAT_TA = """
இந்த வடிவத்தில் மட்டும் பதிலளிக்கவும். சுருக்கமாக இருக்கவும்:

SUMMARY: <ஒரே ஒரு வாக்கியம்>
USES: <அதிகபட்சம் 4 பயன்பாடுகள், கமாவால் பிரிக்கவும்>
DOSAGE: <பெரியவர்: Xmg தினம் X முறை | குழந்தை: Xmg/kg>
SIDE_EFFECTS: <அதிகபட்சம் 4, கமாவால் பிரிக்கவும்>
WARNINGS: <அதிகபட்சம் 3, கமாவால் பிரிக்கவும்>
NOTES: இல்லை
ANSWER: <2-3 வாக்கியங்கள் மட்டும். முக்கிய தகவல்கள். மீண்டும் சொல்லாதீர்கள்.>
"""

_BASE_EN = (
    "You are MedAssist for Anbu Clinic, Tamil Nadu. "
    "Give SHORT, PRECISE medical answers. No long paragraphs. "
    "Use your knowledge if context lacks info. Never say 'not in context'.\n\n"
)

_BASE_TA = (
    "நீங்கள் அன்பு கிளினிக், தமிழ்நாட்டிற்கான MedAssist. "
    "குறுகிய, துல்லியமான மருத்துவ பதில்களை தரவும். நீண்ட பத்திகள் வேண்டாம். "
    "சூழலில் இல்லாவிட்டாலும் உங்கள் அறிவைப் பயன்படுத்தவும்.\n\n"
)

MEDICAL_SYSTEM_EN = _BASE_EN + STRUCTURED_FORMAT_EN
MEDICAL_SYSTEM_TA = _BASE_TA + STRUCTURED_FORMAT_TA

DOSAGE_SYSTEM_EN = (
    "You are a dosage expert for Anbu Clinic. "
    "Give EXACT dosage only. Short and precise.\n\n" + STRUCTURED_FORMAT_EN
)
DOSAGE_SYSTEM_TA = (
    "அன்பு கிளினிக்கின் மருந்தளவு நிபுணர். துல்லியமான அளவு மட்டும்.\n\n" + STRUCTURED_FORMAT_TA
)
COMPARISON_SYSTEM_EN = "Compare medications briefly.\n\n" + STRUCTURED_FORMAT_EN
COMPARISON_SYSTEM_TA = "மருந்துகளை சுருக்கமாக ஒப்பிடவும்.\n\n" + STRUCTURED_FORMAT_TA

VERIFICATION_SYSTEM_EN = (
    "Medical fact-checker. Reply ONLY: SUPPORTED / NOT SUPPORTED / INSUFFICIENT EVIDENCE\n"
    "Evidence: <reasoning>\nANSWER: SUPPORTED or NOT SUPPORTED or INSUFFICIENT EVIDENCE"
)
VERIFICATION_SYSTEM_TA = (
    "மருத்துவ உண்மை சரிபார்ப்பாளர். இதில் மட்டும்: SUPPORTED / NOT SUPPORTED / INSUFFICIENT EVIDENCE\n"
    "சான்று: <பகுப்பாய்வு>\nANSWER: SUPPORTED அல்லது NOT SUPPORTED அல்லது INSUFFICIENT EVIDENCE"
)
PRAMANA_SYSTEM = (
    "Strict medical fact verifier.\n"
    "Supported: yes/no\nEvidence: <max 10 words>\nRevised answer: <concise answer>"
)

def _build_structured_response(raw: str, draft_answer: str) -> Dict:
    def _field(label):
        m = re.search(rf'{label}\s*:\s*(.+?)(?=\n[A-Z_]{{3,}}\s*:|$)', raw, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""

    def _list(text):
        if not text:
            return []
        return [i.strip() for i in re.split(r',\s*|\n[-•*]\s*|\n\d+\.\s*', text) if i.strip()]

    answer_m = re.search(r'ANSWER\s*:\s*(.+?)$', raw, re.IGNORECASE | re.DOTALL)
    full_answer = answer_m.group(1).strip() if answer_m else draft_answer

    notes_raw = _field("NOTES")
    return {
        "summary":      _field("SUMMARY") or draft_answer[:150],
        "uses":         _list(_field("USES"))[:4],
        "dosage":       _field("DOSAGE"),
        "side_effects": _list(_field("SIDE_EFFECTS"))[:4],
        "warnings":     _list(_field("WARNINGS"))[:3],
        "notes":        notes_raw if notes_raw.lower() not in ("none", "இல்லை", "") else "",
        "full_answer":  full_answer,
    }

class Buddhi:
    def __init__(self):
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            self._engine = _get_engine()
        return self._engine

    def reason(self, question: str, context_str: str, q_type: str, medicine_info: Optional[Dict] = None) -> dict:
        t0 = time.time()
        detected_language = detect_language(question)
        logger.info(f"[BUDDHI] lang={detected_language} q={question[:60]}")

        vision_ctx = ""
        if medicine_info:
            name = medicine_info.get("generic_name") or medicine_info.get("brand_name", "")
            strength = medicine_info.get("strength", "")
            form = medicine_info.get("form", "")
            if name and name not in ["Not detected", "Not visible"]:
                vision_ctx = f"[Medicine: {name} {strength} {form}]\n\n"

        full_context = f"{vision_ctx}{context_str}" if vision_ctx else context_str
        system = self._select_system(q_type, detected_language)
        user_prompt = f"Context:\n{full_context[:1500]}\n\nQuestion: {question}"

        # Pass 1: Tarka
        pass1_raw = self.engine.chat(system, user_prompt, max_tokens=1024)
        pass1_answer = self._extract_answer(pass1_raw)
        pass1_steps = self._extract_reasoning_steps(pass1_raw)

        # Pass 2: Pramana
        pass2_raw, pass2_answer, pass2_verified, pass2_fired = "", pass1_answer, True, False
        if full_context.strip() and not self._is_bad_answer(pass1_answer):
            pramana_user = f"Q: {question}\nDraft: {pass1_answer}\nContext: {full_context[:1000]}"
            pass2_raw = self.engine.chat(PRAMANA_SYSTEM, pramana_user, max_tokens=256)
            pass2_answer, pass2_verified = self._extract_pramana(pass2_raw, pass1_answer)
            pass2_fired = True

        # Pass 3: Samsaya (only if bad)
        pass3_fired, final_answer = False, pass2_answer
        if self._is_bad_answer(pass2_answer):
            candidates = []
            for _ in range(2):
                alt = self.engine.chat(system, user_prompt, max_tokens=512, temperature=0.7)
                candidates.append(self._extract_answer(alt))
            candidates = [c for c in candidates if not self._is_bad_answer(c)]
            if candidates:
                final_answer = candidates[0]
                pass3_fired = True

        elapsed = round(time.time() - t0, 3)
        structured_response = _build_structured_response(pass1_raw, final_answer)

        if structured_response.get("full_answer"):
            final_answer = structured_response["full_answer"]

        if detected_language == "ta" and not structured_response.get("notes"):
            structured_response["notes"] = "மருந்து எடுப்பதற்கு முன் மருத்துவரை கலந்தாலோசிக்கவும்."

        return {
            "reasoning_steps": pass1_steps, "pass1_raw": pass1_raw,
            "pass1_answer": pass1_answer, "pass2_raw": pass2_raw if pass2_fired else None,
            "pass2_answer": pass2_answer, "pass2_verified": pass2_verified,
            "pass2_fired": pass2_fired, "pass3_fired": pass3_fired,
            "draft_answer": final_answer, "latency_s": elapsed,
            "model": getattr(self._engine, "model", "unknown"),
            "detected_language": detected_language,
            "structured_response": structured_response,
        }

    def _select_system(self, q_type: str, lang: str = "en") -> str:
        from engine.manas import QType
        is_ta = (lang == "ta")
        if is_ta:
            return {QType.VERIFICATION: VERIFICATION_SYSTEM_TA, QType.COMPARISON: COMPARISON_SYSTEM_TA,
                    QType.DOSAGE: DOSAGE_SYSTEM_TA, QType.MATH: DOSAGE_SYSTEM_TA}.get(q_type, MEDICAL_SYSTEM_TA)
        return {QType.VERIFICATION: VERIFICATION_SYSTEM_EN, QType.COMPARISON: COMPARISON_SYSTEM_EN,
                QType.DOSAGE: DOSAGE_SYSTEM_EN, QType.MATH: DOSAGE_SYSTEM_EN}.get(q_type, MEDICAL_SYSTEM_EN)

    def _extract_answer(self, raw: str) -> str:
        m = re.search(r'ANSWER\s*:\s*(.+?)$', raw, re.IGNORECASE | re.DOTALL)
        if m:
            return re.sub(r'[.,;:]+$', '', m.group(1).strip()).strip()
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        return lines[-1] if lines else raw[:300]

    def _extract_reasoning_steps(self, raw: str) -> List[str]:
        return [l.strip() for l in raw.split("\n") if l.strip()][:4]

    def _extract_pramana(self, raw: str, draft: str) -> Tuple[str, bool]:
        supported_m = re.search(r'Supported\s*:\s*(yes|no)', raw, re.IGNORECASE)
        revised_m = re.search(r'Revised answer\s*:\s*(.+?)(?:\n|$)', raw, re.IGNORECASE)
        if revised_m and supported_m and supported_m.group(1).lower() == "no":
            revised = revised_m.group(1).strip()
            if revised and not self._is_bad_answer(revised):
                return revised, False
        return draft, True

    APOLOGY_PATTERNS = [
        "i don't know", "i cannot", "i am not sure", "no information",
        "sorry", "cannot determine", "unclear", "i'm not certain",
        "not available", "தெரியவில்லை", "முடியாது",
    ]

    def _is_bad_answer(self, answer: str) -> bool:
        if not answer or len(answer.strip()) < 10:
            return True
        return any(p in answer.lower() for p in self.APOLOGY_PATTERNS)
