# LinkedIn PostPilot - LinkedIn Content Generator Pro

A Streamlit application that generates LinkedIn post drafts using Gemini and OpenAI.
It includes smart prompt building, hashtag normalization, bullet-friendly post formatting, and a regeneration workflow.

## Features

- Generate LinkedIn posts from:
  - Topic
  - Tone
  - Target audience
  - Perspective
  - Length
  - Technical depth
- Choose between Gemini (free) and OpenAI (premium).
- Optional Gemini fallback when OpenAI fails.
- Smart output formatting with:
  - natural emoji support,
  - bullet-friendly structure,
  - consistent bullet style,
  - 3–5 hashtags enforced at the end of the post.
- Hashtag normalization converts keyword-only trailing lines into proper `#tag` format.
- Regenerate button creates a new variation without mutating the original topic.
- Built-in post scoring with suggestions for hooks, length, structure, hashtags, and CTAs.

## Requirements

- Python 3.11+
- `streamlit`
- `google-genai`
- `openai`
- `python-dotenv`

## Setup

1. Clone the repository.
2. Create and activate a virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Copy environment variables:

```bash
copy .env.example .env
```

5. Add your API keys to `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

## Running Locally

```bash
streamlit run app.py
```

Then open the URL printed by Streamlit (typically `http://localhost:8501`).

## Running with Docker

Build the image:

```bash
docker build -t linkedin-content-generator-pro .
```

Run the container:

```bash
docker run -p 8501:8501 --env-file .env linkedin-content-generator-pro
```

The app is available at `http://localhost:8501`.

## Environment Variables

- `GEMINI_API_KEY`: API key for Gemini.
- `OPENAI_API_KEY`: API key for OpenAI.
- `PORT`: Optional port for Docker deployment (default is `8501`).

## Notes

- A valid API key is required for the selected provider.
- Gemini API keys are expected to start with `AIza`.
- OpenAI API keys are expected to start with `sk-` or `sk-proj-`.
- If OpenAI is selected and fallback is enabled, Gemini is only used if OpenAI fails.
- The app normalizes duplicate hashtag lines and appends a single clean hashtag line at the end.

## Project Structure

- `app.py` — Streamlit application.
- `requirements.txt` — Python dependencies.
- `Dockerfile` — Container configuration.
- `.env.example` — Environment variable template.
- `.gitignore` — Local cleanup rules.
=======
# linkedin-postpilot
AI-powered LinkedIn content assistant I built to generate structured, engaging, and ready-to-post content. Focuses on prompt design, formatting consistency, and output quality.
>>>>>>> 132f77a95e7cbf3fc59b9d65b10e4bad21345138
