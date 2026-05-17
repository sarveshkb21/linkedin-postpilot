# 🚀 LinkedIn Content Generator Pro

An intelligent, multi-provider AI-powered LinkedIn post generator built with **Streamlit**.

This application uses a **smart routing engine** to generate high-quality LinkedIn content using **free-tier LLMs** with automatic fallback, caching, and performance optimization.

---

# 🧠 Key Features

## ⚡ Multi-Provider AI Routing

* Supports:

  * Gemini (Free)
  * Groq (Free)
  * OpenRouter (Free)
* Automatically selects the best available provider
* Falls back seamlessly on failure

---

## 🔁 Intelligent Fallback Engine

* Provider-level failover across Gemini, Groq, and OpenRouter
* Order is dynamic — sorted by health score and latency on every request
* Model-level fallback (Groq + OpenRouter)
* Health-based routing (avoids failing providers)

---

## ⚡ Performance Optimizations

* Request caching (5-minute TTL)
* Thread-safe execution
* Timeout + retry handling
* Latency-aware routing

---

## 🧠 Smart Prompt Engineering

* Audience-aware content generation:

  * Executives
  * Managers
  * Engineers
  * General Audience
* Perspective-based writing:

  * Leader
  * Practitioner
  * Advisor
  * Storyteller
* Dynamic technical depth (Auto mode)

---

## ✍️ Content Quality Enhancements

* Automatic formatting cleanup
* Hashtag normalization (3–5 enforced)
* Engagement scoring (0–10) with per-criterion explanation
* Regeneration with variation
* Side-by-side comparison of previous vs new post
* LinkedIn character counter (colour-coded: optimal / within limit / over limit)
* Post history — last 10 generations, browsable with copy buttons
* Settings persistence — tone, audience, length, perspective, and depth remembered across reruns

---

## 📊 Observability (Built-in)

* Provider used
* Latency tracking
* Engagement score
* Regeneration tracking

---

# 🏗️ Architecture Overview

```text
User Input
   ↓
Prompt Builder
   ↓
Cache Check
   ↓
Thread-safe Execution
   ↓
Provider Routing (Health + Latency)
   ↓
Model Routing (Fallback + Filtering)
   ↓
Execution + Retry + Timeout
   ↓
Post-processing (Formatting + Hashtags)
   ↓
Scoring + UI Output
```

---

# 🔧 Tech Stack

* **Frontend**: Streamlit
* **LLMs**:

  * Google Gemini
  * Groq
  * OpenRouter
* **Language**: Python 3.10+
* **Concurrency**: Thread (threading module)
* **State Management**: Streamlit Session State

---

# 🔑 API Key Setup

You can use **any one or multiple** providers.

## Option 1: Environment Variables

Create `.env` file:

```bash
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
```

---

## Option 2: Streamlit Secrets (for deployment)

```toml
GEMINI_API_KEY = "your_key"
GROQ_API_KEY = "your_key"
OPENROUTER_API_KEY = "your_key"
```

---

# ▶️ Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

# 📦 Requirements

```txt
streamlit
python-dotenv
requests
groq
google-genai
```

---

# ⚠️ Free Tier Limitations

## Gemini

* Rate-limited (429 errors possible)

## Groq

* Models change frequently
* Rate-limited (RPM/TPM based)

## OpenRouter

* ~50 requests/day (free tier)
* Lower priority execution

---

# 🧠 System Behavior (Real-world)

```text
Gemini → quota exceeded
   ↓
Groq → model unavailable
   ↓
OpenRouter → fallback success
```

✔ Designed for resilience
✔ Handles failures automatically

---

# 🔐 Security

* API keys are **not stored**
* Session-based usage only
* Supports environment-based secrets

---

# 📈 Future Enhancements

* AI Ops Dashboard (latency, failures, usage)
* Auto-learning model ranking
* Dynamic model discovery (Groq/OpenRouter APIs)
* FastAPI backend for enterprise use
* Cost-aware routing

---

# 🧠 Key Design Principles

* Fail fast → fallback faster
* Prefer free-tier optimization
* Adapt to provider instability
* Cache aggressively
* Keep user experience seamless

---

# 🤝 Contributing

Feel free to fork, improve, and extend:

* Add new providers
* Improve routing logic
* Enhance UI/UX

---

# 📄 License

MIT License

---

# 🔥 Final Note

This is not just a content generator.

It is a **lightweight AI orchestration engine** designed to:

* Maximize free-tier usage
* Handle real-world API instability
* Deliver consistent output

---

💡 Built for engineers, operators, and builders who want **resilient AI systems**, not just API wrappers.
