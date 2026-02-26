"""
FastAPI backend for Titanic Dataset Chat Agent.
Uses LangChain + Google Gemini to answer natural-language questions
about the Titanic dataset, including generating visualizations.
"""

import os, re, time
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATASET_PATH = Path(__file__).resolve().parent / "Titanic-Dataset.csv"
CHARTS_DIR = Path(__file__).resolve().parent / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# ── Load dataset ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATASET_PATH)

# ── LLM (shared, stateless) ─────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

CHART_KEYWORDS = [
    "chart", "plot", "histogram", "bar", "graph", "visual",
    "distribution", "pie", "scatter", "show me", "draw", "figure",
]


def build_agent():
    """Create a fresh agent instance per request to avoid stale state."""
    return create_pandas_dataframe_agent(
        llm,
        df,
        agent_type="tool-calling",
        verbose=True,
        allow_dangerous_code=True,
        include_df_in_prompt=True,
        number_of_head_rows=5,
    )


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Titanic ChatBot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/charts", StaticFiles(directory=str(CHARTS_DIR)), name="charts")


class Query(BaseModel):
    question: str


class Answer(BaseModel):
    text: str
    chart_url: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=Answer)
def chat(query: Query):
    """Send a natural-language question and get back a text answer + optional chart URL."""
    # Snapshot existing charts so we can detect new ones
    before = set(f for f in CHARTS_DIR.iterdir() if f.suffix == ".png")

    # Enhance prompt for visualization requests
    question = query.question
    if any(kw in question.lower() for kw in CHART_KEYWORDS):
        question += (
            f"\n\nIMPORTANT: Save any chart/plot to the directory {CHARTS_DIR}/ using "
            f"plt.savefig('{CHARTS_DIR}/<descriptive_name>.png', dpi=100, bbox_inches='tight'), "
            f"then plt.close(). Do NOT call plt.show(). Use matplotlib Agg backend. "
            f"After saving, print('CHART_PATH:<descriptive_name>.png')."
        )

    # Build a fresh agent per request
    agent = build_agent()

    max_retries = 3
    raw = None
    for attempt in range(max_retries):
        try:
            raw = agent.invoke({"input": question})
            break
        except Exception as e:
            err_str = str(e)
            if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                wait = 20 * (attempt + 1)
                print(f"Rate limited - retrying in {wait}s ({attempt+1}/{max_retries})...")
                time.sleep(wait)
                if attempt == max_retries - 1:
                    raise HTTPException(
                        status_code=429,
                        detail="Gemini API rate limit reached. Please wait and try again.",
                    )
            else:
                raise HTTPException(status_code=500, detail=err_str)

    if raw is None:
        raise HTTPException(status_code=500, detail="Agent returned no result.")

    raw_output = raw.get("output", str(raw))

    # Gemini 2.5 may return a list of content parts instead of a plain string
    if isinstance(raw_output, list):
        text_parts = []
        for part in raw_output:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)
        output = " ".join(text_parts)
    else:
        output = str(raw_output)

    # Detect new chart files produced during this request
    after = set(f for f in CHARTS_DIR.iterdir() if f.suffix == ".png")
    new_files = sorted(
        list(after - before),
        key=os.path.getmtime,
        reverse=True,
    )

    chart_file = new_files[0].name if new_files else None

    # Also honour CHART_PATH tags printed by agent code
    if "CHART_PATH:" in output:
        match = re.search(r"CHART_PATH:(\S+\.png)", output)
        if match:
            chart_file = match.group(1)
            output = output.replace(match.group(0), "").strip()

    # If output is empty but we have intermediate steps info, try to extract
    if not output.strip():
        # Check if there were tool outputs in the intermediate steps
        steps = raw.get("intermediate_steps", [])
        if steps:
            for action, result in steps:
                if result and str(result).strip():
                    output = str(result).strip()
                    break
        if not output.strip():
            if chart_file:
                output = "Here is the chart you requested."
            else:
                output = "I processed your question but couldn't generate a response. Please try rephrasing."

    chart_url = f"/charts/{chart_file}" if chart_file else None
    return Answer(text=output, chart_url=chart_url)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
