# 🚢 Titanic Dataset Chat Agent

An AI-powered chatbot that answers natural-language questions about the Titanic dataset — including generating charts and visualizations on the fly.

**Live Demo:** [Streamlit App](https://titanic-bot.streamlit.app) · **API:** [Render Backend](https://titanic-bot.onrender.com)

---

## Features

- **Natural Language Q&A** — Ask questions like *"What percentage of passengers were male?"* or *"How many children survived?"*
- **Auto-generated Charts** — Request visualizations such as histograms, bar charts, pie charts, and scatter plots
- **Conversational UI** — Clean chat interface with message history and example prompts
- **Split Architecture** — Decoupled frontend and backend for independent scaling and deployment

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Google Gemini 2.5 Flash (via Google AI Studio) |
| **Agent Framework** | LangChain + langchain-experimental (Pandas DataFrame Agent) |
| **Backend API** | FastAPI + Uvicorn |
| **Frontend** | Streamlit |
| **Visualization** | Matplotlib + Seaborn |
| **Deployment** | Render (backend) · Streamlit Cloud (frontend) |

## Project Structure

```
TailorBot/
├── backend.py            # FastAPI server with LangChain agent
├── app.py                # Streamlit chat frontend
├── Titanic-Dataset.csv   # Dataset (891 passengers, 12 features)
├── requirements.txt      # Python dependencies
├── .env                  # API key (not tracked in git)
├── Procfile              # Render start command
├── render.yaml           # Render deployment config
├── .streamlit/
│   └── config.toml       # Streamlit theme & settings
├── .gitignore
└── .python-version
```

## Getting Started

### Prerequisites

- Python 3.12+
- A [Google AI Studio](https://aistudio.google.com/) API key

### Local Setup

```bash
# Clone the repo
git clone https://github.com/Rajeev-86/Titanic-Bot.git
cd Titanic-Bot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your API key
echo "GOOGLE_API_KEY=your_key_here" > .env
```

### Run Locally

**1. Start the backend:**

```bash
python backend.py
```

The API will be available at `http://localhost:8000`. Check `http://localhost:8000/docs` for the interactive Swagger UI.

**2. Start the frontend:**

```bash
streamlit run app.py
```

The chat UI will open at `http://localhost:8501`.

## API Reference

### `GET /health`

Health check endpoint.

```json
{ "status": "ok" }
```

### `POST /chat`

Send a question and receive a text answer with an optional chart URL.

**Request:**
```json
{ "question": "What was the survival rate by passenger class?" }
```

**Response:**
```json
{
  "text": "The survival rate by passenger class was...",
  "chart_url": "/charts/survival_by_class.png"
}
```

## Deployment

### Backend → Render

1. Push the repo to GitHub
2. Create a new **Web Service** on [Render](https://render.com)
3. Connect the GitHub repo — Render auto-detects `render.yaml`
4. Add the environment variable `GOOGLE_API_KEY` in the Render dashboard
5. Deploy

### Frontend → Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
2. Connect the same repo, set the main file to `app.py`
3. In **Advanced settings → Secrets**, add:
   ```toml
   BACKEND_URL = "https://your-render-service.onrender.com"
   ```
4. Deploy

## Dataset

The [Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset) contains 891 passenger records with the following features:

| Column | Description |
|--------|-------------|
| PassengerId | Unique ID |
| Survived | 0 = No, 1 = Yes |
| Pclass | Ticket class (1, 2, 3) |
| Name | Passenger name |
| Sex | male / female |
| Age | Age in years |
| SibSp | # of siblings/spouses aboard |
| Parch | # of parents/children aboard |
| Ticket | Ticket number |
| Fare | Passenger fare |
| Cabin | Cabin number |
| Embarked | Port of embarkation (C, Q, S) |

## License

This project is for educational purposes.
