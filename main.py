# main.py
import argparse
import json
from pipeline import run_pipeline

def load_transcript(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    parser = argparse.ArgumentParser(description="Medical NLP Pipeline CLI")
    parser.add_argument("--input", "-i", required=True, help="Path to transcript file (txt)")
    parser.add_argument("--output", "-o", default="output.json", help="Path to JSON output")
    parser.add_argument("--patient", "-p", default="Janet Jones", help="Patient name for report")
    args = parser.parse_args()

    transcript = load_transcript(args.input)
    result = run_pipeline(transcript, patient_name=args.patient)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"Output written to {args.output}")

if __name__ == "__main__":
    main()
