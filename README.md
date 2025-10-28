# 🩺 Medical NLP Pipeline — Emitrr Case Study

This project implements an **AI-based Medical NLP System** that performs:
1. **Medical Transcription Summarization**
2. **Named Entity Recognition (NER)** for Symptoms, Diagnosis, Treatment, Prognosis
3. **Keyword Extraction**
4. **Sentiment & Intent Analysis**
5. **SOAP Note Generation (Bonus)**

It uses Python with spaCy and simple NLP heuristics to process clinical conversations between a physician and a patient.

---

## 🚀 Features

| Task | Description |
|------|--------------|
| 🧠 **NER Extraction** | Extracts entities like *symptoms*, *diagnosis*, *treatment*, and *prognosis*. |
| 📄 **Summarization** | Converts raw doctor-patient transcript into a structured JSON report. |
| 🔑 **Keyword Extraction** | Identifies important medical terms (e.g., *whiplash injury*, *physiotherapy*). |
| 💬 **Sentiment & Intent** | Classifies patient sentiment (Anxious, Neutral, Reassured) and intent (Seeking reassurance, Reporting symptoms, etc.). |
| 🧾 **SOAP Note Generation** | Generates structured medical notes under Subjective, Objective, Assessment, and Plan sections. |

---

## 🧩 Project Structure

```bash
Emitrr/
│
├── main.py                  # Main script (entry point)
├── medical_nlp_pipeline.py  # Core NLP pipeline (NER, summarization, sentiment)
├── conversation.txt         # Sample physician-patient transcript
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## 🧰 Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Emitrr
```
### 2. Create a Virtual Environment
```bash
python -m venv venv
```
### 3. Activate the environment
```bash
Set-ExecutionPolicy Unrestricted -Scope Process
venv\Scripts\activate
```
### 4. Install Dependencies
``` bash
pip install -r requirements.txt
```

## ⚙️ Run the Pipeline

Option 1 — Run with Transcript File
Save your conversation as conversation.txt and run:
```bash
python main.py -i sample_transcript.txt
```

Option 2 — Save Output to a JSON File
```bash
python main.py -i conversation.txt -o output.json
```

Option 3 — Run Demo Directly
If you want to test the included sample conversation:
```bash
python medical_nlp_pipeline.py
```

## 🧠 Model Notes & Approach
1. NER: Uses spaCy (or fallback regex patterns) for entity recognition.
2. Sentiment & Intent: Simple rule-based classification (can be replaced with BERT fine-tuning).
3. Summarization: Heuristic-based structured JSON generation.
4. SOAP Note: Rule-based mapping aligned with clinical documentation standards.

## 📈 Future Improvements
1. Integrate scispaCy or medSpaCy for advanced biomedical NER.
2. Fine-tune DistilBERT or BioBERT on medical sentiment datasets.
3. Add Flask API endpoint for real-time transcript analysis.

## 👩‍💻 Author
**Asmi Garg**  
B.Tech CSE | AI & NLP Enthusiast  
📧 [asmigarg569@gmail.com](mailto:asmigarg569@gmail.com)  
🌐 [LinkedIn](https://www.linkedin.com/in/asmi-garg/) | [GitHub](https://github.com/Asmigarg)


## 🩹 License
This project is for Emitrr case study assessment purposes only and is not intended for clinical use.
