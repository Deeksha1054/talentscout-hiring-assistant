# TalentScout Hiring Assistant

An AI-powered candidate screening chatbot built with Streamlit and Llama 3.3 70B via Groq. It collects candidate information, generates tailored technical questions, and guides users through a structured hiring flow — all without requiring any API key from the user.

---

## Features

- Conversational screening flow : collects name, email, phone, experience, desired role, location, and tech stack in sequence
- Dynamically generates 3–5 technical interview questions based on the candidate's declared stack
- Resume upload : parses a PDF resume and auto-fills candidate details using the LLM
- Sentiment analysis on every candidate response using TextBlob with keyword boosting
- Multilingual support:  **English**, **Hindi**, **Kannada**, **French**, **German** 
- Light / dark theme toggle (single icon, top-right)
- GDPR-compliant data handling: email and phone masked in all displays, session-only storage
- Graceful fallback: handles gibberish input, rate limit errors, and off-topic questions
- Exit keyword detection: typing "bye", "exit", etc. closes the session cleanly
- Downloadable profile summary as JSON at the end of screening

---

## Installation

**Requirements:** Python 3.9+

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/talentscout-hiring-assistant.git
cd talentscout-hiring-assistant

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
python -m textblob.download_corpora

# 4. Add your Groq API key
# Create .streamlit/secrets.toml and add:
# GROQ_API_KEY = "gsk_..."

# 5. Run
streamlit run app.py
```

Get a free Groq API key at: https://console.groq.com

---

## Usage

1. Open the app at `http://localhost:8501`
2. The bot greets you and begins collecting your details one field at a time
3. Optionally upload your PDF resume in the sidebar to auto-fill your information
4. After your tech stack is entered, the bot generates personalised technical questions
5. On completion, download your profile summary as a JSON file
6. Type `bye` or `exit` at any point to end the session early

---

## Technical Details

| Component | Detail |
|---|---|
| Language | Python 3.9+ |
| Frontend | Streamlit |
| LLM | Llama 3.3 70B via Groq API |
| Sentiment | TextBlob + keyword boosting |
| PDF parsing | pypdf |
| Deployment | Streamlit Cloud |

**Architecture:** The app uses a deterministic stage machine (`STAGES` list) to control conversation flow, while the LLM handles natural language generation at each step. Every LLM call receives the full conversation history for context continuity. Sensitive fields are masked before being injected into any prompt or displayed in the UI.

---

## Prompt Design

**Information gathering:** Each stage has a dedicated system prompt instruction that tells the model exactly one thing to do — acknowledge the previous input and ask the next question. The base prompt injects the current candidate profile and enforces strict rules: one question per message, 2–4 sentence limit, no deviation from hiring topics.

**Technical question generation:** A separate prompt sends the candidate's tech stack and experience level to the LLM and requests a JSON array of 4 questions. The prompt explicitly prohibits code-writing questions and specifies a difficulty mix (foundational → advanced). A regex parser extracts the array reliably, with a hardcoded fallback if parsing fails.

**Context handling:** The full message history is passed on every call. The system prompt also receives a JSON snapshot of collected candidate data, so the model always knows what has been gathered and what remains.

---

## Challenges & Solutions

| Challenge | Solution |
|---|---|
| LLM returning extra text around JSON for questions | Used `re.search(r'\[.*?\]', text, re.DOTALL)` to extract arrays regardless of surrounding text |
| Stage advancing even on invalid input | Validators check each field before storing and advancing; invalid input re-asks the same question |
| Gemini API quota errors during development | Switched to Groq (Llama 3.3 70B) — generous free tier, faster responses |
| Resume parser returning inconsistent JSON | Set `temperature=0.1` for the parse call and stripped markdown fences before `json.loads()` |
| Theme toggle causing full page flicker | Stored theme as a single session state bool; CSS is regenerated on each rerun with the correct palette |
| Sentiment analysis inaccurate on short professional text | Combined TextBlob polarity with a domain-specific keyword boost list for better accuracy |

---

## Deployment

Deployed on Streamlit Cloud. The API key is stored in Streamlit's Secrets manager and never exposed to users or committed to the repository.

Live demo: `https://talentscout-hiring-assistant-wtwuio7sejutqqlt99myqu.streamlit.app/`
