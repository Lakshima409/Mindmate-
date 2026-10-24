# MindMate - AI Mental Health Support Platform

> MindMate is a Python-based mental health support chatbot intended for educational and development purposes. It provides a friendly space for users to talk about feelings, reflect on mood, and explore coping strategies. It is not a clinically validated system and must not be used for diagnosis, treatment, or emergency decision-making.

## Overview

MindMate is a Flask web application that uses NLP and a PyTorch neural network to classify user messages into predefined intents and return supportive responses. It includes heuristic safety handling for messages that may indicate a crisis. It is a learning and development project, not a substitute for professional mental health care.

## Features

- Conversational chatbot for greetings, emotional concerns, and general mental health questions
- Intent recognition using NLP and a trained neural network
- Mood check-in support
- Coping-strategy responses
- Selected mental health information based on the intent dataset
- Heuristic crisis-response handling
- Response feedback: helpful / not helpful
- Session-isolated SQLite-backed application data

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application and model logic |
| Flask | Web application and request handling |
| PyTorch | Neural network implementation and model inference |
| NLTK | Text processing utilities |
| SQLite | Local database storage |
| JSON | Intent definitions, patterns, and responses |

## Project Structure

```text
MindMate_Chatbot/
├── app.py
├── model.py
├── nlp_utils.py
├── nltk_utils.py
├── safety.py
├── database.py
├── cleanup_response_ownership.py
├── data/
│   ├── intents.json
│   └── data.pth
├── templates/
│   └── index.html
├── Static/
│   └── style.css
├── requirements.txt
├── conftest.py
└── README.md
```

## Prerequisites

- Python environment compatible with the dependencies in `requirements.txt`
- `FLASK_SECRET_KEY` for normal application startup
- The included `data/intents.json` and `data/data.pth` files

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the virtual environment's Python executable directly.

## Running the Application

```powershell
$env:FLASK_SECRET_KEY = "replace-with-a-strong-secret"
python app.py
```

Then open the local address printed by Flask in your browser.

## Testing

```powershell
python -m py_compile app.py model.py nlp_utils.py nltk_utils.py safety.py database.py
python -m pytest -q
python test_import.py
python test_safety.py
python test_torch.py
python test_load.py
python test_session_isolation.py
```

## Safety and Crisis Limitations

- Crisis detection is heuristic and runs before intent prediction.
- It is not clinically validated and may miss or misclassify messages.
- The chatbot cannot assess risk, provide emergency care, or replace qualified professional support.
- Verified crisis resources in this project are Sri Lanka-specific:
  - Mental health support: 1926
  - Police emergency: 119
  - Ambulance: 1990
- If you or someone else may be in immediate danger, contact local emergency services or qualified support immediately.

## Educational and Clinical Disclaimer

MindMate is an educational software project exploring NLP and conversational AI for mental health support. It is not a clinically validated system. Its responses should not be relied upon for diagnosis, treatment, or emergency decision-making.

## Privacy and Data Notes

- Session IDs are used for session isolation.
- Feedback is stored in a local SQLite database.
- Response ownership records are retained for 30 days by default.
- Cleanup is manual:
  ```powershell
  python cleanup_response_ownership.py --days 30
  ```
- Interaction cleanup is available through `cleanup_old_interactions(conn, days=90)`; no automatic scheduler is included.
- Messages are truncated to 500 characters before storage.
- No automatic training occurs from unreviewed feedback.
- Feedback records are not exposed through public endpoints.

## Security and Privacy Reporting

If you discover a security or privacy issue, do not include sensitive narratives, personal identifiers, credentials, tokens, or raw user content in a public report. Report the issue to the repository maintainer through a private channel if available, or open a minimal public issue requesting private contact. Include only the affected component, reproduction steps that avoid sensitive data, and the potential impact.

## Dataset and Model Provenance

Status: not fully established.

- `data/intents.json` contains 15 intent tags and 7,111 patterns.
- No source, author, or license metadata is present in the dataset file.
- 26 patterns contain external URLs across 21 domains. Public accessibility of those URLs does not establish redistribution permission.
- 4,176 long-form narratives and possible identifier matches remain unreviewed.
- 2 handle-like strings and 15 phone-like matches require human review.
- `data/data.pth` is a trained checkpoint with 15 output tags and an 18,131-token vocabulary.
- The checkpoint may encode information derived from unverified training data.
- The checkpoint SHA-256 is:
  `89CBCEC01E884A807B9E385EDF60BAD75BBF22781F90FFE80E87977B5DA38960`
- Dataset and checkpoint provenance, attribution, privacy review, and redistribution permissions remain unresolved.

## Licensing Status

No license file is currently included. The rights to redistribute or reuse the code, dataset, and trained model have not been established by the available evidence. Do not apply a blanket license to the entire repository until code ownership, dataset rights, model rights, and dependency license obligations have been reviewed. If publication proceeds before that review, the repository should clearly state that rights and permissions are unresolved.

## Limitations

- Feedback uses session identifiers, not authenticated user accounts.
- Session ID validation is not strong identity proof.
- Crisis resources are Sri Lanka-specific.
- Model performance depends on the quality and coverage of the training data.
- The dataset and checkpoint have unresolved provenance and privacy questions.

## Author

Lakshima409 — Software Engineering

## Disclaimer

MindMate is an educational software project exploring NLP and conversational AI for mental health support. It is not a clinically validated system, and its responses should not be relied upon for diagnosis, treatment, or emergency decision-making.

> Built as a learning project to explore Python, NLP, machine learning, and responsible chatbot development.
