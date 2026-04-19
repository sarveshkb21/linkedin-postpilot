# 🚀 LinkedIn Content Generator Pro

An intelligent, free-tier optimized AI-powered LinkedIn post generator with **multi-provider routing, fallback logic, caching, and self-healing execution**.

Built using **Streamlit + LLM APIs (Gemini, Groq, OpenRouter, OpenAI)**.

---

# ✨ Key Features

## 🧠 Smart AI Routing

* Automatically selects the best available provider
* Health-aware + latency-aware routing
* Seamless fallback across:

  * Gemini (Primary)
  * Groq (Fast fallback)
  * OpenRouter (Multi-model fallback)

---

## 🔁 Multi-Level Resilience

* Provider-level fallback
* Model-level fallback (OpenRouter)
* Retry + timeout control
* Self-healing health tracking

---

## ⚡ Performance Optimization

* Prompt-level caching (TTL-based)
* Cache stampede protection
* Concurrency control
* Rate limiting (built-in throttling)

---

## 💰 Cost Control

* Free-tier first architecture
* Automatic prevention of paid model usage (OpenRouter safeguard)
* Optional OpenAI premium usage

---

## 🎯 Content Quality

* Audience-aware generation (Executives, Managers, Engineers)
* Perspective-based writing (Leader, Practitioner, Advisor, Storyteller)
* Technical depth control
* Built-in LinkedIn formatting + hashtag enforcement
* Engagement scoring system

---

# 🏗️ Architecture Overview

```
User Input (Streamlit UI)
        ↓
Prompt Builder
        ↓
AI Execution Engine
   ├── Cache Layer
   ├── In-progress Protection
   ├── Provider Router (Health + Latency)
   ├── Model Router (OpenRouter)
   ├── Retry + Timeout Control
   └── Cost Guard
        ↓
LLM Providers
(Gemini / Groq / OpenRouter / OpenAI)
```

---

# 🔑 API Key Configuration

The app supports **three methods** to configure API keys:

---

## ✅ Option 1: Streamlit Secrets (Recommended for Deployment)

Add in **Streamlit Cloud → Settings → Secrets**

```toml
GEMINI_API_KEY = "your_gemini_key"
GROQ_API_KEY = "your_groq_key"
OPENROUTER_API_KEY = "your_openrouter_key"
OPENAI_API_KEY = "your_openai_key"  # optional
```

---

## ✅ Option 2: Local Development (.env)

Create a `.env` file:

```bash
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
OPENROUTER_API_KEY=your_openrouter_key
OPENAI_API_KEY=your_openai_key
```

⚠️ Add `.env` to `.gitignore`

---

## ✅ Option 3: Runtime Input (UI)

* Enter API key directly in sidebar
* App auto-detects provider based on key format

---

# ▶️ How to Run Locally

```bash
git clone <your-repo>
cd <your-repo>

pip install -r requirements.txt

streamlit run app.py
```

---

# 🧪 Provider Strategy

## Default Flow (Auto Mode)

```
Gemini → Groq → OpenRouter
```

### OpenRouter Model Fallback:

```
Mistral → OpenChat → Llama 13B → Llama 70B
```

---

# 📊 Execution Logic

| Layer            | Purpose                        |
| ---------------- | ------------------------------ |
| Cache            | Avoid duplicate API calls      |
| In-Progress Lock | Prevent concurrent duplication |
| Provider Health  | Avoid failing providers        |
| Latency Tracking | Prefer faster providers        |
| Model Health     | Skip failing models            |
| Retry Logic      | Handle transient errors        |

---

# ⚙️ Configuration Constants

You can tune behavior in code:

```python
MAX_CACHE_SIZE = 100
REQUEST_TIMEOUT = 35
MAX_WORKERS = 2
MODEL_RETRY_LIMIT = 2
```

---

# ⚠️ Known Limitations

* In-memory cache (not distributed)
* No persistent storage
* Rate limits depend on provider free tiers
* Not optimized for very high concurrency (yet)

---

# 🚀 Future Enhancements

* FastAPI-based AI Gateway (decouple UI)
* Redis caching layer
* Provider analytics dashboard
* Cost tracking + optimization engine
* Scheduled post generation
* LinkedIn auto-publishing integration

---

# 🔐 Security Best Practices

* Never commit API keys
* Use `.env` or secrets manager
* Rotate keys periodically
* Avoid logging sensitive data

---

# 🧠 Why This Project Matters

This project demonstrates:

* Multi-provider LLM orchestration
* Resilient AI system design
* Cost-aware AI usage
* Real-world AI infrastructure patterns

---

# 📌 Tech Stack

* Python
* Streamlit
* Gemini API (Google)
* Groq API
* OpenRouter API
* OpenAI API (optional)
* dotenv

---

# 🙌 Contributing

Feel free to:

* Improve routing logic
* Add new providers
* Enhance UI/UX
* Optimize performance

---

# 📄 License

MIT License

---

# ⭐ Final Note

This is not just a content generator.

It is a **lightweight AI execution engine** designed to:

* minimize cost
* maximize availability
* adapt in real-time

---

Happy building 🚀
