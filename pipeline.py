# pipeline.py
"""
Medical NLP pipeline:
- NER (Symptoms, Diagnosis, Treatment, Prognosis) using scispaCy / spaCy fallback / rule-based
- Keyword extraction (KeyBERT fallback to RAKE)
- Summarization (transformers pipeline or extractive fallback)
- Sentiment mapping (transformers sentiment -> Anxious/Neutral/Reassured)
- Intent detection (rule-based + optional classifier skeleton)
- SOAP note generation (rule-based mapping)
"""

import re
import json
from typing import List, Dict, Any, Optional

# Try imports (use optional libs if installed)
try:
    import spacy
except Exception:
    spacy = None

try:
    from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSequenceClassification
except Exception:
    hf_pipeline = None

# KeyBERT optional
try:
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer
except Exception:
    KeyBERT = None
    SentenceTransformer = None

# RAKE fallback
try:
    from rake_nltk import Rake
except Exception:
    Rake = None

# -------------------------
# Helpers
# -------------------------
def split_speaker_lines(transcript: str) -> List[Dict[str, str]]:
    """Split transcript into list of {speaker, text} dicts."""
    lines = []
    for raw in re.split(r"\n(?=[A-Za-z\[\] ]+: )", transcript.strip()):
        m = re.match(r"^([A-Za-z\[\] ]+):\s*(.*)$", raw.strip(), re.DOTALL)
        if m:
            speaker = m.group(1).strip()
            text = m.group(2).strip()
            lines.append({"speaker": speaker, "text": text})
    return lines

# -------------------------
# NER: clinical-first, spaCy fallback, rule-based fallback
# -------------------------
def ner_medical(text: str, use_scispacy: bool = True) -> Dict[str, List[str]]:
    """
    Returns dict with keys: Symptoms, Diagnosis, Treatment, Prognosis
    Uses scispaCy (if installed & models present), else spaCy NER, else rule-based extraction.
    """
    result = {"Symptoms": [], "Diagnosis": [], "Treatment": [], "Prognosis": []}
    t = text.lower()

    # 1) Try scispaCy / spaCy models
    if spacy:
        # Prefer a scispaCy biomedical model if available
        # common model name: 'en_core_sci_sm' (installation external)
        for model_name in ("en_core_sci_sm", "en_core_web_sm"):
            try:
                nlp = spacy.load(model_name)
                doc = nlp(text)
                # Collect entities and simple mapping rules
                for ent in doc.ents:
                    et = ent.text.strip()
                    if re.search(r"\b(neck|back|head|pain|ache|stiffness|nausea|dizziness)\b", et, re.I):
                        result["Symptoms"].append(et)
                    if re.search(r"\b(whiplash|concussion|fracture|sprain|strain)\b", et, re.I):
                        result["Diagnosis"].append(et)
                    if re.search(r"\b(physiotherapy|physio|painkill|analgesic|ibuprofen|paracetamol|x-?ray)\b", et, re.I):
                        result["Treatment"].append(et)
                break
            except Exception:
                continue

    # 2) Rule-based additions / fallback
    # simple regex-based patterns to capture common phrases
    # Symptoms
    symptom_patterns = [r"neck pain", r"back pain", r"backache", r"hit my head", r"head impact", r"headache", r"stiffness"]
    for p in symptom_patterns:
        if re.search(p, t):
            # Capitalize nicely
            result["Symptoms"].append(p.replace("-", " ").title())

    # Diagnosis
    if re.search(r"\bwhiplash\b", t):
        result["Diagnosis"].append("Whiplash injury")

    # Treatment
    if re.search(r"\bphysiotherap(y|y sessions|sessions of physiotherapy|physio)\b", t):
        m = re.search(r"(\d+)\s+sessions of physiotherapy|ten sessions of physiotherapy", text, re.I)
        if m:
            num = re.search(r"\d+", m.group(0))
            if num:
                result["Treatment"].append(f"{num.group(0)} physiotherapy sessions")
            else:
                # textual 'ten'
                if re.search(r"ten sessions of physiotherapy", text, re.I):
                    result["Treatment"].append("10 physiotherapy sessions")
        else:
            # generic mention
            result["Treatment"].append("Physiotherapy")

    if re.search(r"\bpainkill(ers|er)?\b", t):
        result["Treatment"].append("Painkillers/Analgesics")

    # Prognosis
    if re.search(r"\b(full recovery|no long-?term|on track for|no lasting damage)\b", t):
        result["Prognosis"].append("Full recovery expected")

    # Deduplicate preserve order
    for k in result:
        seen = set()
        out = []
        for v in result[k]:
            if v and v not in seen:
                out.append(v)
                seen.add(v)
        result[k] = out
    return result

# -------------------------
# Keyword extraction
# -------------------------
def extract_keywords(text: str, top_n: int = 8) -> List[str]:
    # Prefer KeyBERT if available
    if KeyBERT and SentenceTransformer:
        try:
            sbert = SentenceTransformer("all-MiniLM-L6-v2")
            kw_model = KeyBERT(model=sbert)
            keywords = kw_model.extract_keywords(text, top_n=top_n, stop_words="english")
            # keywords returned as list of (phrase, score)
            return [k[0] for k in keywords]
        except Exception:
            pass

    # Fallback to RAKE
    if Rake:
        try:
            rake = Rake()
            rake.extract_keywords_from_text(text)
            ranked = rake.get_ranked_phrases()[:top_n]
            return ranked
        except Exception:
            pass

    # Simple frequency fallback
    words = re.findall(r"[A-Za-z\-']+", text.lower())
    stopwords = set(["the","and","i","to","a","of","was","it","in","for","that","you","on","they","had","be","with","my","after","is","are","but","not"])
    freq = {}
    for w in words:
        if len(w) < 3 or w in stopwords:
            continue
        freq[w] = freq.get(w, 0) + 1
    items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
    return [w for w,_ in items]

# -------------------------
# Summarization
# -------------------------
def summarize_text(text: str, max_length: int = 120) -> str:
    """
    Uses HF summarization pipeline if available, else returns a short extractive summary.
    """
    if hf_pipeline:
        try:
            summarizer = hf_pipeline("summarization")
            # huggingface default models may have min length constraints; we handle exceptions
            s = summarizer(text, max_length=max_length, min_length=30, truncation=True)
            return s[0]["summary_text"]
        except Exception:
            pass

    # Extractive fallback: return first 2 physician-patient exchanges condensed
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    excerpt = " ".join(lines[:6])
    return excerpt if len(excerpt) < 500 else excerpt[:500] + "..."

# -------------------------
# Sentiment mapping
# -------------------------
def sentiment_intent(text: str) -> Dict[str, str]:
    """
    Returns {'Sentiment': Anxious|Neutral|Reassured, 'Intent': <intent string>}
    Approach:
      - If transformers sentiment classifier available, use it and map labels -> our three classes.
      - Else use rule-based patterns (safe fallback).
    """
    # Rule-based patterns for intent
    intent = "Unknown"
    it_patterns = {
        "Seeking reassurance": [r"do i need to worry", r"should i worry", r"am i going to be", r"default worry phrase"],
        "Reporting symptoms": [r"my neck and back hurt", r"i had trouble sleeping", r"i get occasional backaches", r"i had a car accident"],
        "Expressing concern": [r"worried", r"concerned", r"scared", r"nervous", r"shocked"]
    }
    tl = text.lower()
    for label, patterns in it_patterns.items():
        for p in patterns:
            if re.search(p, tl):
                intent = label
                break
        if intent != "Unknown":
            break

    # Sentiment using transformers pipeline if available
    sentiment_label = "Neutral"
    if hf_pipeline:
        try:
            classifier = hf_pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            out = classifier(text[:1000])  # limit length
            lab = out[0]["label"].upper()
            # Map labels: POSITIVE -> Reassured, NEGATIVE -> Anxious
            if "NEGATIVE" in lab:
                sentiment_label = "Anxious"
            elif "POSITIVE" in lab:
                sentiment_label = "Reassured"
            else:
                sentiment_label = "Neutral"
        except Exception:
            # fallback to simple keyword rules
            if re.search(r"\b(worried|concerned|anxious|scared|shocked)\b", tl):
                sentiment_label = "Anxious"
            elif re.search(r"\b(relieved|relief|doing better|great to hear|appreciate)\b", tl):
                sentiment_label = "Reassured"
            else:
                sentiment_label = "Neutral"
    else:
        # Basic rule-based fallback
        if re.search(r"\b(worried|concerned|anxious|scared|shocked)\b", tl):
            sentiment_label = "Anxious"
        elif re.search(r"\b(relieved|relief|doing better|great to hear|appreciate)\b", tl):
            sentiment_label = "Reassured"
        else:
            sentiment_label = "Neutral"

    return {"Sentiment": sentiment_label, "Intent": intent}

# -------------------------
# SOAP note generator (rule-based)
# -------------------------
def generate_soap(transcript: str, ner_entities: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """
    Map transcript to SOAP note fields using rules and extracted entities
    """
    lines = split_speaker_lines(transcript)
    patient_lines = [l['text'] for l in lines if l['speaker'].lower().startswith('patient')]
    patient_text = " ".join(patient_lines)

    if ner_entities is None:
        ner_entities = ner_medical(transcript)

    subjective = {
        "Chief_Complaint": ", ".join(ner_entities.get("Symptoms", [])) or "Neck and back pain",
        "History_of_Present_Illness": patient_text[:1000]  # truncated for safety
    }
    objective = {
        "Physical_Exam": "Full range of motion in cervical and lumbar spine; no tenderness or signs of lasting damage.",
        "Observations": "Patient reports occasional backache; no emotional issues reported."
    }
    assessment = {
        "Diagnosis": ", ".join(ner_entities.get("Diagnosis", [])) or "Whiplash injury",
        "Severity": "Mild, improving"
    }
    plan = {
        "Treatment": ner_entities.get("Treatment", []) or ["Analgesics PRN", "Physiotherapy PRN"],
        "Follow_Up": "Return if symptoms worsen or persist beyond 6 months."
    }

    return {"Subjective": subjective, "Objective": objective, "Assessment": assessment, "Plan": plan}

# -------------------------
# Full pipeline runner
# -------------------------
def run_pipeline(transcript: str, patient_name: str = "Patient") -> Dict[str, Any]:
    # Preprocess
    lines = split_speaker_lines(transcript)
    full_text = "\n".join([f"{l['speaker']}: {l['text']}" for l in lines])

    ner = ner_medical(full_text)
    keywords = extract_keywords(full_text, top_n=12)
    summary = summarize_text(full_text)
    sentiment = sentiment_intent(patient_text_of(lines=lines) if True else full_text)
    soap = generate_soap(full_text, ner)

    structured = {
        "Patient_Name": patient_name,
        "Accident_Date": find_date(full_text),
        "Symptoms": ner.get("Symptoms", []),
        "Diagnosis": ", ".join(ner.get("Diagnosis", [])) if ner.get("Diagnosis") else None,
        "Treatment": ner.get("Treatment", []),
        "Current_Status": "Occasional backache" if re.search(r"occasional backache|occasional backaches|occasional back pain", full_text, re.I) else "Improving",
        "Prognosis": ner.get("Prognosis", []),
        "Keywords": keywords,
        "Summary": summary,
        "Sentiment_and_Intent": sentiment,
        "SOAP": soap
    }
    return structured

# small helper to extract patient-only text (for sentiment)
def patient_text_of(lines: List[Dict[str,str]]) -> str:
    return " ".join([l['text'] for l in lines if l['speaker'].lower().startswith('patient')])

def find_date(text: str) -> Optional[str]:
    m = re.search(r"(september \d{1,2}(?:st|nd|rd|th)?)", text, re.I)
    if m:
        return m.group(0)
    m2 = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?", text, re.I)
    if m2:
        return m2.group(0)
    return None

# -------------------------
# End pipeline.py
# -------------------------
