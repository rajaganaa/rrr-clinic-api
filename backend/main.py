"""
RRR Clinic MedAssist API — main.py
FastAPI backend for Dr. Rajeswari's clinic, Tamil Nadu.

7-step Antahkarana reasoning pipeline:
  Manas    → Question routing + NLP classification
  Chitta   → ChromaDB dense retrieval (drug PDFs)
  Buddhi   → Groq LLM reasoner (llama3-70b, Tamil + English)
  Ahamkara → Confidence scoring
  Sakshi   → Hallucination detection + verified final answer
  Vision   → GPT-4o medicine image analysis (GitHub Models)
  Tools    → Dosage calculator / expiry checker / FDA API

Author: Rajaganapathy M — M.Tech AI, SRM University
Patent: 202641043947
"""

import os
import re
import uuid
import logging
import time
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RRR Clinic MedAssist API",
    description=(
        "AI-powered medical assistant for RRR Clinic, Tamil Nadu. "
        "7-step Antahkarana reasoning: Manas → Chitta → Buddhi (Groq) → "
        "Ahamkara → Sakshi. Vision: GPT-4o via GitHub Models. "
        "Tamil + English. Author: Rajaganapathy M, SRM University."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy-loaded pipeline components ───────────────────────────────────────────
_manas    = None
_chitta   = None
_buddhi   = None
_ahamkara = None
_sakshi   = None

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/tmp/rrr_clinic_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_components():
    global _manas, _chitta, _buddhi, _ahamkara, _sakshi
    if _manas is None:
        logger.info("[INIT] Loading RRR Clinic pipeline components...")
        from engine.manas import Manas
        from engine.chitta import Chitta
        from engine.buddhi import Buddhi
        from engine.ahamkara import Ahamkara
        from engine.sakshi import Sakshi
        _manas    = Manas()
        _chitta   = Chitta()
        _buddhi   = Buddhi()
        _ahamkara = Ahamkara()
        _sakshi   = Sakshi()
        logger.info("[INIT] All pipeline components ready")
    return _manas, _chitta, _buddhi, _ahamkara, _sakshi


# ── W&B: one persistent run per process ───────────────────────────────────────
_wandb_run = None
_request_count = 0  # track total requests in this process


def _get_wandb_run():
    global _wandb_run
    if _wandb_run is not None:
        return _wandb_run
    api_key = os.environ.get("WANDB_API_KEY", "")
    if not api_key:
        return None
    try:
        import wandb
        _wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "rrr-clinic-medassist"),
            entity=os.environ.get("WANDB_ENTITY", "rajaganaa-ai"),
            name="rrr-clinic-production",
            resume="allow",
            config={
                "clinic":   "RRR Clinic, Tamil Nadu",
                "doctor":   "Dr. Rajeswari M.D",
                "author":   "Rajaganapathy M, SRM University",
                "patent":   "202641043947",
                "model":    "llama3-70b-8192 (Groq)",
                "pipeline": "7-step Antahkarana",
                "deploy":   "Azure Container Apps — Central India",
            },
            tags=["production", "healthcare", "tamil", "rag", "groq", "azure"],
        )
        logger.info("[W&B] Run initialized: rrr-clinic-production")
        return _wandb_run
    except Exception as e:
        logger.warning(f"[W&B] Init failed: {e}")
        return None


def _log_wandb(request_id, question, buddhi_result, ahamkara_result, sakshi_result, latency):
    """
    Log each inference to W&B with namespaced metrics for clean dashboard.
    Metrics logged:
      latency/   — total, buddhi, overhead
      quality/   — confidence, verified, corrected, hallucinations
      pipeline/  — pass2, pass3, verification
      usage/     — tamil ratio, question length, model
    """
    global _request_count
    run = _get_wandb_run()
    if run is None:
        return
    try:
        import wandb
        _request_count += 1

        hallucination_flags = sakshi_result.get("hallucination_flags", [])
        hallucination_count = len(hallucination_flags) if isinstance(hallucination_flags, list) else 0

        raw_conf  = ahamkara_result.get("confidence_score", 0)
        conf_float = raw_conf / 100.0 if raw_conf > 1 else raw_conf

        lang     = buddhi_result.get("detected_language", "en")
        is_tamil = 1 if lang == "ta" else 0

        run.log({
            # Performance
            "latency/total_s":    latency,
            "latency/buddhi_s":   buddhi_result.get("latency_s", 0),
            "latency/overhead_s": round(latency - buddhi_result.get("latency_s", 0), 3),

            # Quality
            "quality/confidence":          conf_float,
            "quality/sakshi_verified":     int(sakshi_result.get("verified", True)),
            "quality/sakshi_corrected":    int(sakshi_result.get("corrected", False)),
            "quality/hallucination_count": hallucination_count,

            # Pipeline
            "pipeline/pass2_fired":    int(buddhi_result.get("pass2_fired", False)),
            "pipeline/pass3_fired":    int(buddhi_result.get("pass3_fired", False)),
            "pipeline/pass2_verified": int(buddhi_result.get("pass2_verified", True)),

            # Usage
            "usage/is_tamil":        is_tamil,
            "usage/question_length": len(question),
            "usage/total_requests":  _request_count,

            # Meta
            "meta/confidence_label": ahamkara_result.get("confidence_label", ""),
            "meta/model":            buddhi_result.get("model", "unknown"),
        })
    except Exception as e:
        logger.debug(f"[W&B] Logging skipped: {e}")


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("[STARTUP] RRR Clinic MedAssist API starting...")
    try:
        from rag.medassist_rag import build_index_if_needed
        build_index_if_needed()
        logger.info("[STARTUP] ChromaDB index ready")
    except Exception as e:
        logger.warning(f"[STARTUP] ChromaDB index build skipped: {e}")
    # Pre-init W&B so first request isn't slow
    _get_wandb_run()


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status":   "healthy",
        "service":  "RRR Clinic MedAssist API",
        "clinic":   "RRR Clinic, Tamil Nadu — Dr. Rajeswari M.D",
        "author":   "Rajaganapathy M, SRM University",
        "patent":   "202641043947",
        "version":  "1.0.0",
        "wandb":    "enabled" if os.environ.get("WANDB_API_KEY") else "disabled",
        "pipeline": {
            "manas":    "Question Router",
            "chitta":   "ChromaDB Dense Retrieval",
            "buddhi":   "Groq llama3-70b (Tamil + English)",
            "ahamkara": "Confidence Scorer",
            "sakshi":   "Hallucination Verifier",
            "vision":   "GPT-4o via GitHub Models",
        },
    }


@app.get("/")
async def root():
    return {
        "message": "RRR Clinic MedAssist API — /docs for Swagger, /health for status",
        "clinic":  "RRR Clinic, Tamil Nadu",
        "doctor":  "Dr. Rajeswari M.D",
    }


# ── Main reasoning endpoint ───────────────────────────────────────────────────

@app.post("/api/reason")
async def reason(
    question: str = Form(...),
    image: Optional[UploadFile] = File(None),
):
    request_id = str(uuid.uuid4())[:8]
    t_total    = time.time()
    logger.info(f"[{request_id}] Request: {question[:80]}")

    manas, chitta, buddhi, ahamkara, sakshi = get_components()

    # ── Step 1: Vision ────────────────────────────────────────────────────────
    vision_result = None
    image_path    = None

    if image and image.filename:
        try:
            image_path = UPLOAD_DIR / f"{request_id}_{image.filename}"
            contents   = await image.read()
            with open(image_path, "wb") as f:
                f.write(contents)
            from vision.medicine_vision import extract_medicine_info
            vision_result = extract_medicine_info(str(image_path))
            logger.info(f"[{request_id}] Vision: {vision_result.get('drug_name', 'unknown')}")
            drug_name = (
                vision_result.get("generic_name") or
                vision_result.get("brand_name", "")
            )
            if drug_name and drug_name not in ["Not detected", "Not visible"] and drug_name not in question:
                question = f"[About {drug_name}] {question}"
        except Exception as e:
            logger.warning(f"[{request_id}] Vision failed: {e}")
            vision_result = {"error": str(e), "extraction_method": "failed"}

    # ── Step 2: Manas ─────────────────────────────────────────────────────────
    manas_result = manas.get_routing_info(question)
    q_type       = manas_result["question_type"]
    entities     = manas_result["entities"]
    logger.info(f"[{request_id}] Manas: {q_type} conf={manas_result['confidence']}")

    # ── Step 3: Tool shortcuts ────────────────────────────────────────────────
    tool_result = None
    from engine.manas import QType

    if q_type == QType.DOSAGE:
        tool_result = await _handle_dosage(question, entities, vision_result)
    elif q_type == QType.EXPIRY:
        tool_result = await _handle_expiry(question, vision_result)
    elif q_type == QType.FDA:
        tool_result = await _handle_fda(entities, vision_result)

    # ── Step 4: Chitta ────────────────────────────────────────────────────────
    chitta_result = chitta.retrieve(question, entities, k=5)
    logger.info(f"[{request_id}] Chitta: {chitta_result['num_chunks']} chunks")

    # ── Step 5: Buddhi ────────────────────────────────────────────────────────
    if vision_result:
        drug_hint = vision_result.get("brand_name", "")
        if drug_hint and drug_hint not in ["Not visible", "Not detected"] and drug_hint not in question:
            question = f"[Medicine: {drug_hint}] {question}"

    buddhi_result = buddhi.reason(
        question=question,
        context_str=chitta_result["context_str"],
        q_type=q_type,
        medicine_info=vision_result,
    )
    logger.info(
        f"[{request_id}] Buddhi: lang={buddhi_result['detected_language']} "
        f"pass2={buddhi_result['pass2_fired']} latency={buddhi_result['latency_s']}s"
    )

    # ── Step 6: Ahamkara ──────────────────────────────────────────────────────
    ahamkara_result = ahamkara.score(buddhi_result, chitta_result, question)
    logger.info(
        f"[{request_id}] Ahamkara: {ahamkara_result['confidence_score']} "
        f"({ahamkara_result['confidence_label']})"
    )

    # ── Step 7: Sakshi ────────────────────────────────────────────────────────
    sakshi_result = sakshi.verify(
        question=question,
        draft_answer=buddhi_result["draft_answer"],
        context_str=chitta_result["context_str"],
        sources=chitta_result["sources"],
        buddhi_result=buddhi_result,
        ahamkara_result=ahamkara_result,
    )
    logger.info(f"[{request_id}] Sakshi: verified={sakshi_result['verified']}")

    total_latency = round(time.time() - t_total, 3)

    # ── W&B logging (MLOps Day 4) ─────────────────────────────────────────────
    _log_wandb(request_id, question, buddhi_result, ahamkara_result, sakshi_result, total_latency)

    # ── Assemble response ─────────────────────────────────────────────────────
    response = {
        "request_id":      request_id,
        "question":        question,
        "total_latency_s": total_latency,
        "vision":          vision_result,
        "manas":           manas_result,
        "chitta": {
            "retrieved_chunks": chitta_result["retrieved_chunks"],
            "scores":           [c.get("score", 0) for c in chitta_result["retrieved_chunks"]],
            "num_chunks":       chitta_result["num_chunks"],
            "retrieval_method": chitta_result["retrieval_method"],
        },
        "buddhi": {
            "reasoning_steps":     buddhi_result["reasoning_steps"],
            "draft_answer":        buddhi_result["draft_answer"],
            "pass1_answer":        buddhi_result["pass1_answer"],
            "pass2_fired":         buddhi_result["pass2_fired"],
            "pass2_verified":      buddhi_result["pass2_verified"],
            "pass3_fired":         buddhi_result["pass3_fired"],
            "model":               buddhi_result["model"],
            "latency_s":           buddhi_result["latency_s"],
            "detected_language":   buddhi_result["detected_language"],
            "structured_response": buddhi_result["structured_response"],
        },
        "ahamkara":    ahamkara_result,
        "sakshi": {
            "verified":            sakshi_result["verified"],
            "corrected":           sakshi_result["corrected"],
            "hallucination_flags": sakshi_result["hallucination_flags"],
            "correction_note":     sakshi_result["correction_note"],
            "final_answer":        sakshi_result["final_answer"],
            "sakshi_summary":      sakshi_result.get("sakshi_summary", ""),
            "medical_disclaimer":  sakshi_result["medical_disclaimer"],
        },
        "tool_result":  tool_result,
        "final_answer": sakshi_result["final_answer"],
        "sources":      chitta_result["sources"],
    }

    if image_path and image_path.exists():
        try:
            image_path.unlink()
        except Exception:
            pass

    return JSONResponse(content=response)


# ── Tool handlers ─────────────────────────────────────────────────────────────

_KNOWN_DRUGS = [
    "cetirizine", "paracetamol", "acetaminophen", "ibuprofen", "amoxicillin",
    "dolo", "crocin", "calpol", "tylenol", "advil", "motrin", "brufen",
    "augmentin", "zyrtec", "metformin", "atorvastatin", "omeprazole",
    "pantoprazole", "azithromycin", "ciprofloxacin",
]


def _extract_drug_from_question(question: str, entities: list, vision_result) -> str:
    q_lower = question.lower()
    for drug in _KNOWN_DRUGS:
        if drug in q_lower:
            return drug
    if vision_result:
        d = vision_result.get("generic_name") or vision_result.get("brand_name", "")
        if d and d not in ["Not detected", "Not visible"]:
            return d.lower()
    if entities:
        for entity in entities:
            e_lower = entity.lower()
            if any(drug in e_lower or e_lower in drug for drug in _KNOWN_DRUGS):
                return e_lower
    return "paracetamol"


async def _handle_dosage(question: str, entities: list, vision_result) -> dict:
    try:
        from tools.dosage_calc import calculate_dosage
        weight_m  = re.search(r"(\d+(?:\.\d+)?)\s*(?:kg|kilogram)", question, re.IGNORECASE)
        weight    = float(weight_m.group(1)) if weight_m else 70.0
        q_lower   = question.lower()
        age_group = "adult"
        if any(w in q_lower for w in ["child", "kid", "pediatric", "baby", "infant", "year-old", "years old"]):
            age_group = "child"
        elif any(w in q_lower for w in ["elderly", "old", "senior", "geriatric"]):
            age_group = "elderly"
        drug        = _extract_drug_from_question(question, entities, vision_result)
        result_text = calculate_dosage.invoke({"drug": drug, "weight_kg": weight, "age_group": age_group})
        return {"tool": "dosage_calculator", "drug": drug, "weight_kg": weight, "age_group": age_group, "result": result_text}
    except Exception as e:
        logger.error(f"[DOSAGE] {e}")
        return {"tool": "dosage_calculator", "error": str(e)}


async def _handle_expiry(question: str, vision_result) -> dict:
    try:
        from tools.expiry_check import check_medicine_expiry
        expiry_date = None
        if vision_result:
            expiry_date = vision_result.get("expiry_date")
            if expiry_date in ["Not visible", "Not detected", None]:
                expiry_date = None
        if not expiry_date:
            m = re.search(
                r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}\b"
                r"|\b\d{1,2}[/\-]\d{4}\b|\b\d{4}[/\-]\d{1,2}\b",
                question, re.IGNORECASE,
            )
            expiry_date = m.group(0) if m else "Unknown"
        if expiry_date == "Unknown":
            return {"tool": "expiry_checker", "error": "No expiry date found in image or question"}
        result_text = check_medicine_expiry.invoke(expiry_date)
        return {"tool": "expiry_checker", "expiry_date": expiry_date, "result": result_text}
    except Exception as e:
        logger.error(f"[EXPIRY] {e}")
        return {"tool": "expiry_checker", "error": str(e)}


async def _handle_fda(entities: list, vision_result) -> dict:
    try:
        from tools.fda_api import get_reaction_counts
        drug = None
        if vision_result:
            drug = vision_result.get("generic_name")
            if drug in ["Not detected", "Not visible", None]:
                drug = None
        if not drug and entities:
            drug = entities[0]
        if not drug:
            return {"tool": "fda_api", "error": "No drug name identified"}
        data      = get_reaction_counts(drug, top_n=10)
        reactions = []
        if data and "results" in data:
            reactions = [
                {"reaction": r.get("term", ""), "count": r.get("count", 0)}
                for r in data["results"][:10]
            ]
        return {"tool": "fda_api", "drug": drug, "reactions": reactions}
    except Exception as e:
        logger.error(f"[FDA] {e}")
        return {"tool": "fda_api", "error": str(e)}


# ── Utility endpoints ─────────────────────────────────────────────────────────

@app.get("/api/sources")
async def list_sources():
    data_dir = Path(os.environ.get("MEDASSIST_DATA_DIR", "./data/drug_guides"))
    pdfs     = list(data_dir.glob("*.pdf")) if data_dir.exists() else []
    return {"sources": [p.name for p in pdfs], "count": len(pdfs)}


@app.post("/api/search")
async def search(query: str = Form(...)):
    from rag.medassist_rag import search_drug_database
    chunks = search_drug_database(query, k=5)
    return {"query": query, "results": chunks, "count": len(chunks)}


@app.post("/api/vision")
async def vision_only(image: UploadFile = File(...)):
    image_path = UPLOAD_DIR / f"{uuid.uuid4()}_{image.filename}"
    contents   = await image.read()
    with open(image_path, "wb") as f:
        f.write(contents)
    try:
        from vision.medicine_vision import extract_medicine_info
        return extract_medicine_info(str(image_path))
    finally:
        if image_path.exists():
            image_path.unlink()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, log_level="info")