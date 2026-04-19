import atexit
import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from threading import Lock
import streamlit as st
from dotenv import load_dotenv


load_dotenv()


GEMINI_MODEL = "gemini-2.5-flash"
OPENAI_MODEL = "gpt-4o-mini"
GROQ_MODEL = "mixtral-8x7b-32768"
OPENROUTER_MODELS = [
    # ⚡ Fast (Primary)
    "mistralai/mistral-7b-instruct",
    "openchat/openchat-7b",
    "meta-llama/llama-3.2-3b-instruct",
    "google/gemma-3-4b",

    # ⚖️ Balanced (Fallback)
    "google/gemma-3-12b",
    "z-ai/glm-4.5-air",
    "minimax/minimax-m2.5",
    "meta-llama/llama-3.3-70b-instruct",

    # 🧠 Heavy (Last resort)
    "nous/hermes-3-405b",
]
ENV_GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "")).strip()
ENV_OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()
ENV_GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "")).strip()
ENV_OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY", "")).strip()
REQUEST_EXECUTOR = ThreadPoolExecutor(max_workers=2)
atexit.register(lambda: REQUEST_EXECUTOR.shutdown(wait=False))

# Simple in-memory cache for duplicate requests
_REQUEST_CACHE = {}
_RATE_LIMIT_TRACKER = {}
MAX_CACHE_SIZE = 100
_PROVIDER_HEALTH = {
    "Gemini (Free)": 0,
    "Groq (Free)": 0,
    "OpenRouter (Free)": 0,
}
_MODEL_HEALTH = {}
_PROVIDER_LATENCY = {}
_IN_PROGRESS = set()
_IN_PROGRESS_LOCK = Lock()


@dataclass
class GenerationResult:
    post: str
    model_used: str
    provider: str


class TimeoutException(Exception):
    pass


def resolve_api_key(user_key: str, env_key: str) -> str:
    user_key = user_key.strip()
    return user_key if user_key else env_key


def is_valid_gemini_key(api_key: str) -> bool:
    api_key = api_key.strip()
    return api_key.startswith("AIza") and len(api_key) > 30


def is_valid_openai_key(api_key: str) -> bool:
    api_key = api_key.strip()
    return api_key.startswith("sk-") or api_key.startswith("sk-proj-")


def is_valid_groq_key(api_key: str) -> bool:
    api_key = api_key.strip()
    return api_key.startswith("gsk_") and len(api_key) > 30


def is_valid_openrouter_key(api_key: str) -> bool:
    api_key = api_key.strip()
    return api_key.startswith("sk-or-") and len(api_key) > 30


def retry_call(func, retries: int = 2):
    for attempt in range(retries):
        try:
            return func()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5)


def get_cached_response(prompt_hash: str) -> str | None:
    """Retrieve cached response if available and recent (within 5 minutes)."""
    cache_key = prompt_hash
    if cache_key in _REQUEST_CACHE:
        cached_data = _REQUEST_CACHE[cache_key]
        if time.time() - cached_data["timestamp"] < 300:  # 5 minute TTL
            return cached_data["response"]
        else:
            del _REQUEST_CACHE[cache_key]
    return None


def cache_response(prompt_hash: str, response: str) -> None:
    """Cache response with timestamp."""
    cache_key = prompt_hash
    _REQUEST_CACHE[cache_key] = {"response": response, "timestamp": time.time()}
    if len(_REQUEST_CACHE) > MAX_CACHE_SIZE:
        _REQUEST_CACHE.pop(next(iter(_REQUEST_CACHE)))


def decay_provider_health():
    for k in _PROVIDER_HEALTH:
        if _PROVIDER_HEALTH[k] > 0:
            _PROVIDER_HEALTH[k] -= 1


def generate_with_fallback_chain(prompt: str, api_keys: dict[str, str]) -> tuple[str, str]:
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()

    cached = get_cached_response(prompt_hash)
    if cached:
        return cached, "Cache"

    with _IN_PROGRESS_LOCK:
        if prompt_hash in _IN_PROGRESS:
            time.sleep(0.5)
            cached = get_cached_response(prompt_hash)
            if cached:
                return cached, "Cache"
        _IN_PROGRESS.add(prompt_hash)

    try:
        last_error = None
        decay_provider_health()
        providers_to_try = sorted(
            FREE_PROVIDER_CHAIN,
            key=lambda x: (
                _PROVIDER_HEALTH.get(x[0], 0),
                _PROVIDER_LATENCY.get(x[0], 999)
            )
        )
        for provider_name, func in providers_to_try:
            api_key = api_keys.get(provider_name)
            if not api_key:
                continue

            if _PROVIDER_HEALTH.get(provider_name, 0) > 3:
                continue

            try:
                start_time = time.time()
                raw = generate_with_control(lambda: func(prompt, api_key))
                latency = time.time() - start_time
                _PROVIDER_LATENCY[provider_name] = latency
                post = enforce_hashtags(clean_post(raw))
                _REQUEST_CACHE[prompt_hash] = {"response": post, "timestamp": time.time()}
                _PROVIDER_HEALTH[provider_name] = 0
                return post, provider_name
            except Exception as e:
                st.warning(f"{provider_name} failed: {str(e)[:80]}")
                _PROVIDER_HEALTH[provider_name] = _PROVIDER_HEALTH.get(provider_name, 0) + 1
                last_error = e
                continue

        raise RuntimeError(f"All providers failed: {last_error}")
    finally:
        with _IN_PROGRESS_LOCK:
             _IN_PROGRESS.discard(prompt_hash)


def call_with_timeout(func, timeout: int = 30):
    future = REQUEST_EXECUTOR.submit(func)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError as exc:
        future.cancel()
        raise TimeoutException("Request timed out") from exc


def generate_with_control(func, timeout: int = 35):
    time.sleep(0.6)
    return call_with_timeout(lambda: retry_call(func), timeout=timeout)


def show_generation_error(exc: Exception) -> None:
    message = str(exc)
    lower_message = message.lower()
    if isinstance(exc, TimeoutException) or "timeout" in lower_message or "timed out" in lower_message:
        st.error("Request timed out. Please try again.")
    elif "authentication" in lower_message or "unauthorized" in lower_message or "api key" in lower_message:
        st.error("Authentication failed. Please verify your API key.")
    else:
        st.error(f"Request failed: {message}. Please try again or enable fallback.")


def clear_sensitive_session_keys() -> None:
    for key in (
        "api_key",
        "primary_api_key",
        "gemini_api_key",
        "openai_api_key",
        "fallback_api_key",
        "fallback_gemini_api_key",
    ):
        st.session_state.pop(key, None)


def persona_instructions(target_audience: str) -> str:
    instructions = {
        "Executives": (
            "Write for executives. Emphasize strategic outcomes, growth, risk, ROI, "
            "market position, operating model impact, and clear business decisions."
        ),
        "Managers": (
            "Write for managers. Emphasize team execution, prioritization, stakeholder "
            "alignment, productivity, process improvement, and measurable delivery."
        ),
        "Engineers": (
            "Write for engineers. Emphasize practical implementation, tradeoffs, systems "
            "thinking, reliability, tooling, debugging, and technical credibility."
        ),
        "General Audience": (
            "Write for a broad professional audience. Keep the message useful, clear, "
            "credible, and easy to understand without relying on jargon."
        ),
    }
    return instructions.get(target_audience, instructions["General Audience"])


def perspective_instructions(perspective: str) -> str:
    instructions = {
        "Leader": "Write with a decisive, vision-led point of view focused on direction and outcomes.",
        "Practitioner": "Write from hands-on experience with practical lessons and concrete details.",
        "Advisor": "Write as a trusted advisor offering clear guidance, framing, and next steps.",
        "Storyteller": "Write with a narrative arc, specific context, tension, and a useful takeaway.",
    }
    return instructions.get(perspective, instructions["Advisor"])


def resolve_depth(technical_depth: str, target_audience: str) -> str:
    if technical_depth != "Auto":
        return technical_depth

    auto_depth = {
        "Executives": "Balanced",
        "Managers": "Balanced",
        "Engineers": "Highly Technical",
        "General Audience": "Non-Technical",
    }
    return auto_depth.get(target_audience, "Balanced")


def build_prompt(topic: str, tone: str, length: str, target_audience: str, perspective: str, depth: str) -> str:
    length_rules = {
        "Short": "90 to 130 words",
        "Medium": "140 to 210 words",
        "Long": "220 to 320 words",
    }

    depth_rules = {
        "Non-Technical": (
            "Avoid implementation detail. Explain concepts in plain business language."
        ),
        "Balanced": (
            "Blend practical technical credibility with plain-language business value."
        ),
        "Highly Technical": (
            "Include concrete technical terms, architecture considerations, tooling ideas, "
            "or implementation tradeoffs while staying readable on LinkedIn."
        ),
    }

    return f"""
Create one LinkedIn post.

Topic:
{topic}

Target audience:
{target_audience}

Audience guidance:
{persona_instructions(target_audience)}

Perspective:
{perspective}

Perspective guidance:
{perspective_instructions(perspective)}

Tone:
{tone}

Length:
{length_rules.get(length, length_rules["Medium"])}

Technical depth:
{depth}

Technical depth guidance:
{depth_rules.get(depth, depth_rules["Balanced"])}

Strict writing rules:
- Start with a strong hook in the first line.
- Use short paragraphs with blank lines between them.
- Do not use markdown formatting.
- Use emojis naturally where they add clarity or emphasis.
- Do not force emojis; use them only when appropriate.
- Avoid excessive or repetitive emoji usage.
- Use bullet points when listing multiple ideas, steps, or comparisons.
- If the content includes multiple points, structure it using bullets for clarity.
- Choose a natural bullet style (e.g., '-', '•', or numbered points).
- Use only ONE bullet style consistently throughout the post.
- Keep each bullet concise (preferably one line).
- Do not overuse bullets; maintain a natural LinkedIn flow.
- End with a clear call to action.
- You MUST include 3 to 5 relevant hashtags at the end.
- Each hashtag MUST start with # (e.g., #AI #CloudComputing).
- Do NOT write plain keywords; always prefix with #.
- Place all hashtags on a new line at the very end.
- Keep the post human, specific, and credible.
- Return only the LinkedIn post text.
""".strip()


def clean_post(text: str) -> str:
    text = text.strip()

    # Remove markdown artifacts
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"[*_`>~]", "", text)

    lines = [line.strip() for line in text.splitlines()]

    formatted = []
    for i, line in enumerate(lines):
        if not line:
            continue

        # ✅ Normalize bullet spacing (fix "-Text" → "- Text")
        line = re.sub(r"^([-•])\s*", r"\1 ", line)

        is_bullet = bool(re.match(r"^[-•\d+.)]", line))

        if formatted:
            prev_line = formatted[-1]
            prev_is_bullet = bool(re.match(r"^[-•\d+.)]", prev_line))

            # Add spacing between paragraphs (but not between bullets)
            if not is_bullet and not prev_is_bullet:
                formatted.append("")
            
            # ✅ NEW: Add spacing AFTER bullet list
            if not is_bullet and prev_is_bullet:
                formatted.append("")

        formatted.append(line)

    return "\n".join(formatted).strip()


def normalize_hashtags(text: str) -> str:
    text = text.strip()
    if not text:
        return text

    lines = text.splitlines()

    # Step 1: remove trailing empty lines
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return text

    hashtag_pattern = r"^(#\w+(\s+#\w+)*)$"
    keyword_pattern = r"^[A-Za-z0-9][A-Za-z0-9 ]{0,80}$"

    keyword_lines = []
    while lines:
        last_line = lines[-1].strip()
        if re.match(hashtag_pattern, last_line):
            lines.pop()
            continue
        if re.match(keyword_pattern, last_line) and "#" not in last_line:
            keyword_lines.append(lines.pop())
            continue
        break

    if not keyword_lines:
        return "\n".join(lines).strip()

    tokens = []
    for line in reversed(keyword_lines):
        tokens.extend(re.split(r"\s+", line.strip()))

    hashtags = []
    for token in tokens:
        clean = token.strip("#")
        if clean:
            tag = f"#{clean}"
            if tag not in hashtags:
                hashtags.append(tag)

    hashtags = hashtags[:5]

    if lines:
        return "\n".join(lines).strip() + "\n\n" + " ".join(hashtags)
    else:
        return " ".join(hashtags)


def enforce_hashtags(text: str) -> str:
    text = text.strip()

    # Extract hashtags from entire text
    hashtags = list(dict.fromkeys(re.findall(r"#\w+", text)))  # unique, preserve order

    # Keep only 3–5 hashtags
    if len(hashtags) < 3:
        defaults = ["#AI", "#Automation", "#Cloud"]
        for tag in defaults:
            if tag not in hashtags:
                hashtags.append(tag)
            if len(hashtags) >= 3:
                break

    hashtags = hashtags[:5]

    # Remove ALL hashtag lines from original text
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        if not re.match(r"^(#\w+(\s+#\w+)*)$", line.strip()):
            cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines).strip()

    # Append ONE clean hashtag line
    return cleaned_text + "\n\n" + " ".join(hashtags)


def generate_with_gemini(prompt: str, api_key: str) -> str:
    if not api_key:
        raise RuntimeError("Gemini API key is required.")

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("google-genai is not installed.") from exc

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    if hasattr(response, "text") and response.text:
        return response.text

    if hasattr(response, "candidates") and response.candidates:
        parts = getattr(response.candidates[0].content, "parts", [])
        if parts and getattr(parts[0], "text", None):
            return parts[0].text

    raise RuntimeError("Gemini returned an empty response.")


def generate_with_openai(prompt: str, api_key: str) -> str:
    if not api_key:
        raise RuntimeError("OpenAI API key is required.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai is not installed.") from exc

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert LinkedIn ghostwriter for technology leaders. "
                    "Return only the final post text with no markdown formatting."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.75,
        max_tokens=400,
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("OpenAI returned an empty response.")

    return content


def generate_with_groq(prompt: str, api_key: str) -> str:
    if not api_key:
        raise RuntimeError("Groq API key is required.")

    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("groq is not installed. Install with: pip install groq") from exc

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert LinkedIn ghostwriter for technology leaders. "
                    "Return only the final post text with no markdown formatting."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.75,
        max_tokens=400,
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Groq returned an empty response.")

    return content


def generate_with_openrouter(prompt: str, api_key: str) -> str:
    if not api_key:
        raise RuntimeError("OpenRouter API key is required.")

    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is not installed.") from exc

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://linkedin-postpilot.streamlit.app",
        "X-Title": "LinkedIn Post Generator",
    }

    payload = {
        "model": OPENROUTER_MODELS[0],
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert LinkedIn ghostwriter for technology leaders. "
                    "Return only the final post text with no markdown formatting."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.75,
        "max_tokens": 400,
    }
    
    
    last_error = None
    max_models_to_try = 2

    # 🔹 Tier definitions
    FAST_MODELS = OPENROUTER_MODELS[:4]
    BALANCED_MODELS = OPENROUTER_MODELS[4:8]
    HEAVY_MODELS = OPENROUTER_MODELS[8:]

    # 🔹 Smart selection
    if len(prompt) < 800:
        selected_models = FAST_MODELS
    else:
        selected_models = FAST_MODELS + BALANCED_MODELS[:2]

    # 🔹 Filter unhealthy
    usable_models = [
        m for m in selected_models
        if _MODEL_HEALTH.get(m, 0) <= 2
    ]

    # 🔹 Fallback safety
    if not usable_models:
        usable_models = FAST_MODELS[:2]

    if not usable_models:
        usable_models = OPENROUTER_MODELS[:2]

    # 🔹 Sort models by health + latency (SMART ROUTING)
    sorted_models = sorted(
        usable_models,
        key=lambda m: (
            _MODEL_HEALTH.get(m, 0),
            _PROVIDER_LATENCY.get(m, 999)
        )
    )

    # 🔹 Execution loop
    for model_name in sorted_models[:max_models_to_try]:
        st.caption(f"OpenRouter trying: {model_name}")
        payload["model"] = model_name

        try:
            start = time.time()

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=20,
            )

            if response.status_code != 200:
                raise RuntimeError(f"{response.status_code}: {response.text}")

            data = response.json()

            # 🔒 Cost protection
            if "usage" in data and data["usage"].get("total_cost", 0) > 0:
                raise RuntimeError(f"Paid model triggered: {model_name}")

            if "error" in data:
                raise RuntimeError(data["error"])

            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            if content:
                _MODEL_HEALTH[model_name] = 0
                _PROVIDER_LATENCY[model_name] = time.time() - start
                return content

            raise RuntimeError("Empty response")

        except Exception as e:
            _MODEL_HEALTH[model_name] = _MODEL_HEALTH.get(model_name, 0) + 1
            last_error = e
            continue

    raise RuntimeError(f"OpenRouter failed: {last_error}")

FREE_PROVIDER_CHAIN = [
    ("Gemini (Free)", generate_with_gemini),
    ("Groq (Free)", generate_with_groq),
    ("OpenRouter (Free)", generate_with_openrouter),
]

def generate_post(
    topic: str,
    tone: str,
    length: str,
    target_audience: str,
    perspective: str,
    technical_depth: str,
    selected_provider: str,
    gemini_api_key: str,
    openai_api_key: str,
    fallback_enabled: bool = False,
    fallback_gemini_api_key: str = "",
    groq_api_key: str = "",
    openrouter_api_key: str = "",
) -> GenerationResult:

    resolved_depth = resolve_depth(technical_depth, target_audience)

    prompt = build_prompt(
        topic,
        tone,
        length,
        target_audience,
        perspective,
        resolved_depth,
    )

    api_keys = {
        "Gemini (Free)": gemini_api_key,
        "Groq (Free)": groq_api_key,
        "OpenRouter (Free)": openrouter_api_key,
    }

    post, provider = generate_with_fallback_chain(prompt, api_keys)

    return GenerationResult(
        post=post,
        model_used="Auto",
        provider=provider,
    )

def render_copy_button(text: str) -> None:
    payload = json.dumps(text)
    st.iframe(
        f"""
        <button id="copy-btn" style="
            background:#0a66c2;
            color:white;
            border:0;
            border-radius:6px;
            padding:0.65rem 0.9rem;
            font-weight:600;
            cursor:pointer;
            width:100%;
        ">Copy post</button>
        <script>
        const button = document.getElementById("copy-btn");
        button.onclick = async () => {{
            await navigator.clipboard.writeText({payload});
            button.innerText = "Copied";
            setTimeout(() => button.innerText = "Copy post", 1400);
        }};
        </script>
        """,
        height=52,
    )


def main() -> None:
    st.set_page_config(
        page_title="LinkedIn Content Generator Pro",
        page_icon="in",
        layout="wide",
    )

    if "initialized" not in st.session_state:
        clear_sensitive_session_keys()
        st.session_state["initialized"] = True

    st.title("\U0001F680 LinkedIn Content Generator Pro")
    st.caption("Tip: Use Gemini or Groq for free tier, OpenAI for premium quality")

    with st.sidebar:
        st.header("Settings")
        st.subheader("Free Tier Provider")
        selected_provider = "Auto (Recommended)"
        provider_label = "Auto"
        selected_env_key = ""

        # Keep password inputs unkeyed so secrets are not stored under named session_state entries.
        primary_api_key = st.text_input("Auto API Key (Primary)", type="password")
        fallback_enabled = False
        fallback_gemini_api_key = ""

        if primary_api_key.strip():
            st.success("Auto: Using user-provided free-tier API key")
        elif ENV_GEMINI_API_KEY or ENV_GROQ_API_KEY or ENV_OPENROUTER_API_KEY:
            st.success("Auto: Using one or more free-tier keys from environment")
        else:
            st.warning("No free-tier API key available. Enter one above or configure an environment key.")

        st.caption("Your API key is used only for this session and not stored.")
        if "STREAMLIT_SERVER_HEADLESS" in os.environ:
            # Running on Streamlit Cloud
            st.info(
                "\U0001F511 On hosted app, API keys are managed by the app owner.\n\n"
                "If you need custom keys, run locally or contact support."
            )
        else:
            # Local development help links
            help_links = {
                "Gemini": "[\U0001F680 Generate Gemini API Key](https://aistudio.google.com/app/apikey)\n\nFree tier available via Google AI Studio",
                "Groq": "[\U0001F680 Generate Groq API Key](https://console.groq.com/keys)\n\nFree tier available via Groq Console",
                "OpenRouter": "[\U0001F680 Create OpenRouter API Key](https://openrouter.ai/keys)\n\nFree tier available via OpenRouter",
                "OpenAI": "[\U0001F510 Create OpenAI API Key](https://platform.openai.com/api-keys)\n\nRequires account and billing setup",
            }
            if provider_label in help_links:
                st.info(f"\U0001F511 Don't have a {provider_label} API key?\n\n{help_links[provider_label]}")

    topic = st.text_area(
        "Topic",
        placeholder="Example: Why platform engineering is becoming essential for enterprise DevOps teams",
        height=150,
    )

    st.subheader("\U0001F4DD Content Setup")
    st.caption("Define how you want your post to be generated")
    tone_col, audience_col, length_col, perspective_col, depth_col = st.columns(5)
    with tone_col:
        tone = st.selectbox(
            "Tone",
            ["Professional", "Conversational", "Thought Leadership", "Bold", "Educational", "Persuasive"],
        )
    with audience_col:
        target_audience = st.selectbox("Target Audience", ["Executives", "Managers", "Engineers", "General Audience"])
    with length_col:
        length = st.selectbox("Length", ["Short", "Medium", "Long"], index=1)
    with perspective_col:
        perspective = st.selectbox(
            "Perspective",
            ["Leader", "Practitioner", "Advisor", "Storyteller"],
        )
    with depth_col:
        technical_depth = st.selectbox(
            "Technical Depth",
            ["Auto", "Non-Technical", "Balanced", "Highly Technical"],
        )

    resolved_depth = resolve_depth(technical_depth, target_audience)
    if technical_depth == "Auto":
        st.caption(f"Auto-selected depth: {resolved_depth}")

    primary_api_key = primary_api_key.strip()
    fallback_gemini_api_key = resolve_api_key(
        fallback_gemini_api_key.strip(),
        ENV_GEMINI_API_KEY,
    )
    
    # Resolve all free-provider API keys by key format or environment
    gemini_api_key = resolve_api_key(
        primary_api_key if is_valid_gemini_key(primary_api_key) else "",
        ENV_GEMINI_API_KEY,
    )
    groq_api_key = resolve_api_key(
        primary_api_key if is_valid_groq_key(primary_api_key) else "",
        ENV_GROQ_API_KEY,
    )
    openrouter_api_key = resolve_api_key(
        primary_api_key if is_valid_openrouter_key(primary_api_key) else "",
        ENV_OPENROUTER_API_KEY,
    )
    openai_api_key = resolve_api_key(
        "",
        ENV_OPENAI_API_KEY,
    )

    selected_api_key = ""
    if gemini_api_key and is_valid_gemini_key(gemini_api_key):
        selected_api_key = gemini_api_key
    elif groq_api_key and is_valid_groq_key(groq_api_key):
        selected_api_key = groq_api_key
    elif openrouter_api_key and is_valid_openrouter_key(openrouter_api_key):
        selected_api_key = openrouter_api_key

    invalid_primary = not selected_api_key
    missing_fallback_key = fallback_enabled and not fallback_gemini_api_key
    invalid_fallback_key = (
        fallback_enabled
        and fallback_gemini_api_key
        and not is_valid_gemini_key(fallback_gemini_api_key)
    )
    topic_missing = not topic.strip()
    generate_disabled = (
        invalid_primary
        or missing_fallback_key
        or invalid_fallback_key
        or topic_missing
    )

    provider_label = "Auto"

    if not selected_api_key:
        _, message_col, _ = st.columns([1, 2, 1])
        with message_col:
            st.warning("\U0001F511 Add an API key in the sidebar or configure one in the environment")
    elif invalid_primary:
        st.error("Invalid API key format")
    elif missing_fallback_key:
        st.error("Fallback enabled but no Gemini API key available (user or environment)")
    elif invalid_fallback_key:
        st.error("Invalid Gemini fallback API key format")
    elif topic_missing:
        st.warning("Please enter a topic to generate a LinkedIn post")
    else:
        st.success(f"Using {provider_label}")
        if fallback_enabled and fallback_gemini_api_key == gemini_api_key and selected_provider == "OpenAI (Premium)":
            st.warning("Primary and fallback Gemini keys are the same")

    generate = st.button(
        "Generate LinkedIn Post",
        type="primary",
        use_container_width=True,
        disabled=generate_disabled,
    )

    if generate:
        if not topic.strip():
            st.warning("Enter a topic first.")
            return

        with st.spinner("Generating your LinkedIn post..."):
            try:
                if selected_provider == "OpenAI (Premium)" and fallback_enabled:
                    st.caption("OpenAI selected. Gemini fallback is enabled and will be used only if OpenAI fails.")
                elif selected_provider == "Auto (Recommended)":
                    st.caption("Auto mode will choose the first available free provider key.")

                start_time = time.time()
                result = generate_post(
                    topic.strip(),
                    tone,
                    length,
                    target_audience,
                    perspective,
                    technical_depth,
                    selected_provider,
                    gemini_api_key,
                    openai_api_key,
                    fallback_enabled=fallback_enabled,
                    fallback_gemini_api_key=fallback_gemini_api_key,
                    groq_api_key=groq_api_key,
                    openrouter_api_key=openrouter_api_key,
                )
                latency = time.time() - start_time
                score, suggestion = score_post(result.post)
                st.session_state["last_result"] = result
                st.session_state["last_score"] = score
                st.session_state["last_suggestion"] = suggestion
                st.session_state["last_latency"] = latency
                st.session_state["fallback_used"] = result.provider == "Gemini (Fallback)"
                st.session_state["last_inputs"] = {
                    "topic": topic.strip(),
                    "tone": tone,
                    "length": length,
                    "target_audience": target_audience,
                    "perspective": perspective,
                    "technical_depth": technical_depth,
                    "provider": selected_provider,
                    "gemini_api_key": gemini_api_key,
                    "openai_api_key": openai_api_key,
                    "groq_api_key": groq_api_key,
                    "openrouter_api_key": openrouter_api_key,
                    "fallback_enabled": fallback_enabled,
                    "fallback_gemini_api_key": fallback_gemini_api_key,
                }
                st.session_state["regen_count"] = 0
                st.session_state.pop("last_regenerated_latency", None)
            except Exception as exc:
                show_generation_error(exc)
                return

    if "last_result" in st.session_state:
        result: GenerationResult = st.session_state["last_result"]
        score = st.session_state["last_score"]
        suggestion = st.session_state["last_suggestion"]
        latency = st.session_state.get("last_latency")
        regenerated_latency = st.session_state.get("last_regenerated_latency")
        fallback_used = st.session_state.get("fallback_used", False)
        regen_count = st.session_state.get("regen_count", 0)

        left, right = st.columns([2, 1])

        with left:
            st.subheader("Generated Post")
            st.success(f"Generated using: {result.provider}")
            st.caption(f"Execution Path: {result.provider} -> {result.model_used}")
            if fallback_used:
                st.warning("Fallback triggered during execution")
            if latency is not None:
                st.caption(f"Latency: {latency:.2f}s")
            if regen_count:
                st.caption(f"Regenerations: {regen_count}")
            if regenerated_latency is not None:
                st.caption(f"Regenerated Latency: {regenerated_latency:.2f}s")
            button_col, copy_col = st.columns([1, 1])
            with button_col:
                if st.button("\U0001F504 Regenerate", use_container_width=True):
                    last_inputs = st.session_state.get("last_inputs")
                    if not last_inputs:
                        st.error("Previous inputs not found. Please generate again.")
                    else:
                        original_topic = last_inputs["topic"]
                        variation_seed = random.randint(100000, 999999)
                        enhanced_topic = (
                            f"{original_topic}\n\n"
                            "Rewrite this post with a different hook, structure, and example while keeping the same core idea. Avoid repeating phrases from the previous version.\n"
                            f"Variation seed: {variation_seed}"
                        )

                        with st.spinner("Regenerating your LinkedIn post..."):
                            try:
                                start_time = time.time()
                                regenerated_result = generate_post(
                                    enhanced_topic,
                                    last_inputs["tone"],
                                    last_inputs["length"],
                                    last_inputs["target_audience"],
                                    last_inputs["perspective"],
                                    last_inputs["technical_depth"],
                                    last_inputs["provider"],
                                    last_inputs["gemini_api_key"],
                                    last_inputs["openai_api_key"],
                                    fallback_enabled=last_inputs["fallback_enabled"],
                                    fallback_gemini_api_key=last_inputs["fallback_gemini_api_key"],
                                    groq_api_key=last_inputs.get("groq_api_key", ""),
                                    openrouter_api_key=last_inputs.get("openrouter_api_key", ""),
                                )
                                regenerated_latency = time.time() - start_time
                                regenerated_score, regenerated_suggestion = score_post(regenerated_result.post)
                                st.session_state["last_result"] = regenerated_result
                                st.session_state["last_score"] = regenerated_score
                                st.session_state["last_suggestion"] = regenerated_suggestion
                                st.session_state["last_latency"] = regenerated_latency
                                st.session_state["last_regenerated_latency"] = regenerated_latency
                                st.session_state["fallback_used"] = regenerated_result.provider == "Gemini (Fallback)"
                                st.session_state["regen_count"] = st.session_state.get("regen_count", 0) + 1
                                st.rerun()
                            except Exception as exc:
                                show_generation_error(exc)

            with copy_col:
                render_copy_button(result.post)

            st.text_area("Output", result.post, height=380, label_visibility="collapsed")

        with right:
            st.subheader("Generation Details")
            st.metric("Engagement Score", f"{score}/10")
            st.write(suggestion)
            st.divider()
            st.write("Provider")
            st.success(result.provider)
            st.write("Model")
            st.code(result.model_used, language="text")


if __name__ == "__main__":
    main()
