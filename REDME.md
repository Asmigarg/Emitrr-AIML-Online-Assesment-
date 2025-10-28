# Medical NLP Pipeline — Company Assessment Submission

This repo contains an end-to-end Medical NLP pipeline for:
- Clinical NER (Symptoms, Diagnosis, Treatment, Prognosis)
- Keyword extraction
- Text summarization (transcript -> structured report)
- Sentiment & intent detection
- SOAP note generation (rule-based)

## Files
- `pipeline.py` — main pipeline (NER, keywords, summarization, sentiment, SOAP)
- `main.py` — CLI wrapper to run pipeline on transcript (.txt) and output JSON
- `sample_transcript.txt` — sample transcript to test
- `requirements.txt` — Python dependencies

---

## Run locally in VS Code (recommended)
1. **Open VS Code** and open the project folder (`medical-nlp-pipeline`).

2. **Create a virtual environment** (recommended Python 3.8+):
   - Windows:
     ```
     python -m venv venv
     venv\\Scripts\\activate
     ```
   - macOS / Linux:
     ```
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies**:
pip install --upgrade pip
pip install -r requirements.txt

- If you're short on disk or want a quick run, install minimal:
  ```
  pip install spacy transformers torch datasets
  ```
- Optional (recommended for better results):
  ```
  pip install keybert sentence-transformers rake-nltk scispacy medspacy
  ```

4. **(Optional) Install scispaCy model** (if using scispaCy):
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_sm-0.5.0.tar.gzpip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_sm-0.5.0.tar.gz
Or a larger scispaCy model if you need more coverage.

5. **Run the pipeline**:
python main.py --input sample_transcript.txt --output output.json --patient "Janet Jones"

The script writes `output.json` with the structured JSON.

6. **Open `output.json`** to view the structured summary, keywords, sentiment & intent, and the SOAP note.

---

## Expected (sample) output
An example of `output.json` content (partial):

```json
{
"Patient_Name": "Janet Jones",
"Accident_Date": "September 1st",
"Symptoms": ["Neck Pain", "Back Pain", "Head Impact"],
"Diagnosis": "Whiplash injury",
"Treatment": ["10 physiotherapy sessions", "Painkillers/Analgesics"],
"Current_Status": "Occasional backache",
"Prognosis": ["Full recovery expected"],
"Keywords": ["physiotherapy", "whiplash", "neck pain", "back pain"],
"Summary": "Patient was in a rear-end car accident on September 1st. Immediate head impact with neck and back pain; severe for approximately four weeks; improved after ten physiotherapy sessions and use of painkillers. Currently has occasional backache; full recovery expected within six months.",
"Sentiment_and_Intent": {"Sentiment": "Reassured", "Intent": "Reporting symptoms"},
"SOAP": { "...": "..." }
}
