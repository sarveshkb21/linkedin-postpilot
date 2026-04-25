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
import streamlit.components.v1 as components
from dotenv import load_dotenv
load_dotenv()
GEMINI_MODEL = "gemini-2.5-flash"
GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
]
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
ENV_GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "")).strip()
ENV_OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY", "")).strip()
REQUEST_EXECUTOR = ThreadPoolExecutor(max_workers=2)
atexit.register(lambda: REQUEST_EXECUTOR.shutdown(wait=False))

_REQUEST_CACHE: dict = {}
_REQUEST_CACHE_LOCK = Lock()
MAX_CACHE_SIZE = 100

_PROVIDER_HEALTH = {
    "Gemini (Free)": 0,
    "Groq (Free)": 0,
    "OpenRouter (Free)": 0,
}
_MODEL_HEALTH = {}
_PROVIDER_LATENCY = {}
_IN_PROGRESS: set = set()
_IN_PROGRESS_LOCK = Lock()
_STATE_LOCK = Lock()  # guards _PROVIDER_HEALTH, _MODEL_HEALTH, _PROVIDER_LATENCY


@dataclass
class GenerationResult:
    post: str
    model_used: str
    provider: str


class TimeoutException(Exception):
    pass


def is_valid_gemini_key(api_key: str) -> bool:
    return api_key.startswith("AIza") and len(api_key) > 30


def is_valid_groq_key(api_key: str) -> bool:
    return api_key.startswith("gsk_") and len(api_key) > 30


def is_valid_openrouter_key(api_key: str) -> bool:
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
    with _REQUEST_CACHE_LOCK:
        cached_data = _REQUEST_CACHE.get(prompt_hash)
        if cached_data is None:
            return None
        if time.time() - cached_data["timestamp"] < 300:  # 5-minute TTL
            return cached_data["response"]
        del _REQUEST_CACHE[prompt_hash]
    return None


def cache_response(prompt_hash: str, response: str) -> None:
    """Cache response with timestamp, evicting the oldest entry when full."""
    with _REQUEST_CACHE_LOCK:
        _REQUEST_CACHE[prompt_hash] = {"response": response, "timestamp": time.time()}
        if len(_REQUEST_CACHE) > MAX_CACHE_SIZE:
            _REQUEST_CACHE.pop(next(iter(_REQUEST_CACHE)))


def decay_provider_health():
    with _STATE_LOCK:
        for k in _PROVIDER_HEALTH:
            if _PROVIDER_HEALTH[k] > 0:
                _PROVIDER_HEALTH[k] -= 1


def generate_with_fallback_chain(prompt: str, api_keys: dict[str, str]) -> tuple[str, str]:
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    cached = get_cached_response(prompt_hash)
    if cached:
        return cached, "Cache"

    with _IN_PROGRESS_LOCK:
        in_progress = prompt_hash in _IN_PROGRESS
        if not in_progress:
            _IN_PROGRESS.add(prompt_hash)

    if in_progress:
        time.sleep(0.5)
        cached = get_cached_response(prompt_hash)
        if cached:
            return cached, "Cache"
        # Only add to in-progress if not already tracked
        with _IN_PROGRESS_LOCK:
            _IN_PROGRESS.add(prompt_hash)

    try:
        last_error = None
        decay_provider_health()
        with _STATE_LOCK:
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
            with _STATE_LOCK:
                health = _PROVIDER_HEALTH.get(provider_name, 0)
            if health > 3:
                continue
            try:
                start_time = time.time()
                raw = generate_with_control(lambda: func(prompt, api_key))
                latency = time.time() - start_time
                post = enforce_hashtags(clean_post(raw))
                cache_response(prompt_hash, post)
                with _STATE_LOCK:
                    _PROVIDER_LATENCY[provider_name] = latency
                    _PROVIDER_HEALTH[provider_name] = 0
                return post, provider_name
            except Exception as e:
                st.warning(f"{provider_name} failed: {str(e)[:80]}")
                with _STATE_LOCK:
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
    # Removed: time.sleep(0.6) — was adding 600 ms of dead latency to every
    # generation. Rate-limit pauses belong inside individual provider functions
    # only when an explicit 429 is received.
    return call_with_timeout(lambda: retry_call(func), timeout=timeout)


def show_generation_error(exc: Exception) -> None:
    message = str(exc)
    lower_message = message.lower()
    if isinstance(exc, TimeoutException) or "timeout" in lower_message or "timed out" in lower_message:
        st.error("Request timed out. Please try again.")
    elif "authentication" in lower_message or "unauthorized" in lower_message or "api key" in lower_message:
        st.error("Authentication failed. Please verify your API key.")
    else:
        st.error(f"Request failed: {message}. Please try again.")


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
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"[*_`>~]", "", text)
    lines = [line.strip() for line in text.splitlines()]
    formatted = []
    for line in lines:
        if not line:
            continue
        line = re.sub(r"^([-•])\s*", r"\1 ", line)
        is_bullet = bool(re.match(r"^[-•\d+.)]", line))
        if formatted:
            prev_line = formatted[-1]
            prev_is_bullet = bool(re.match(r"^[-•\d+.)]", prev_line))
            if not is_bullet and not prev_is_bullet:
                formatted.append("")
            if not is_bullet and prev_is_bullet:
                formatted.append("")
        formatted.append(line)
    return "\n".join(formatted).strip()




def enforce_hashtags(text: str) -> str:
    text = text.strip()
    hashtags = list(dict.fromkeys(re.findall(r"#\w+", text)))
    if len(hashtags) < 3:
        defaults = ["#AI", "#Automation", "#Cloud"]
        for tag in defaults:
            if tag not in hashtags:
                hashtags.append(tag)
            if len(hashtags) >= 3:
                break
    hashtags = hashtags[:5]
    lines = text.splitlines()
    cleaned_lines = [
        line for line in lines
        if not re.match(r"^(#\w+(\s+#\w+)*)$", line.strip())
    ]
    cleaned_text = "\n".join(cleaned_lines).strip()
    return cleaned_text + "\n\n" + " ".join(hashtags)



_CTA_PHRASES = [
    # Original narrow set
    "what do you think",
    "agree",
    "thoughts",
    "comment",
    # Additions — common LinkedIn CTA patterns
    "let me know",
    "share your",
    "drop a comment",
    "i'd love to hear",
    "id love to hear",
    "have you",
    "curious",
    "would you",
    "weigh in",
    "your take",
    "leave a comment",
    "reply below",
    "tell me",
    "join the conversation",
    "follow for",
    "repost if",
]


def score_post(text: str) -> tuple[int, str]:
    score = 5
    suggestions = []

    word_count = len(text.split())
    if 100 <= word_count <= 280:
        score += 1
    else:
        suggestions.append("Adjust length for better engagement (100–280 words).")

    first_line = text.split("\n")[0]
    if len(first_line) > 20:
        score += 1
    else:
        suggestions.append("Improve the opening hook to grab attention.")

    hashtags = re.findall(r"#\w+", text)
    if 3 <= len(hashtags) <= 5:
        score += 1
    else:
        suggestions.append("Use 3–5 relevant hashtags.")

    if "\n\n" in text:
        score += 1
    else:
        suggestions.append("Add spacing for better readability.")

    lower_text = text.lower()
    if any(phrase in lower_text for phrase in _CTA_PHRASES):
        score += 1
    else:
        suggestions.append("Consider adding a call-to-action.")

    final_score = min(score, 10)
    suggestion_text = (
        " ".join(suggestions) if suggestions
        else "Strong post. Well structured and engaging."
    )
    return final_score, suggestion_text


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




def generate_with_groq(prompt: str, api_key: str) -> str:
    if not api_key:
        raise RuntimeError("Groq API key is required.")
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("groq is not installed. Install with: pip install groq") from exc
    client = Groq(api_key=api_key)
    last_error = None
    model_attempts = []
    for model in GROQ_MODELS:
        try:
            model_attempts.append(model)
            response = client.chat.completions.create(
                model=model,
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
                max_tokens=512,
            )
            content = response.choices[0].message.content
            if content:
                return content
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"Groq failed after trying {model_attempts}: {last_error}")


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
    FAST_MODELS = OPENROUTER_MODELS[:4]
    BALANCED_MODELS = OPENROUTER_MODELS[4:8]

    selected_models = FAST_MODELS if len(prompt) < 800 else FAST_MODELS + BALANCED_MODELS[:2]
    with _STATE_LOCK:
        usable_models = [m for m in selected_models if _MODEL_HEALTH.get(m, 0) <= 2]
        if not usable_models:
            usable_models = FAST_MODELS[:2]
        sorted_models = sorted(
            usable_models,
            key=lambda m: (_MODEL_HEALTH.get(m, 0), _PROVIDER_LATENCY.get(m, 999))
        )

    model_attempts = []
    for model_name in sorted_models[:max_models_to_try]:
        model_attempts.append(model_name)
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
            if "usage" in data and data["usage"].get("total_cost", 0) > 0:
                raise RuntimeError(f"Paid model triggered: {model_name}")
            if "error" in data:
                raise RuntimeError(data["error"])
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                with _STATE_LOCK:
                    _MODEL_HEALTH[model_name] = 0
                    _PROVIDER_LATENCY[model_name] = time.time() - start
                return content
            raise RuntimeError("Empty response")
        except Exception as e:
            with _STATE_LOCK:
                _MODEL_HEALTH[model_name] = _MODEL_HEALTH.get(model_name, 0) + 1
            last_error = e
            continue

    raise RuntimeError(f"OpenRouter failed after trying {model_attempts}: {last_error}")


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
    gemini_api_key: str = "",
    groq_api_key: str = "",
    openrouter_api_key: str = "",
) -> GenerationResult:
    resolved_depth = resolve_depth(technical_depth, target_audience)
    prompt = build_prompt(topic, tone, length, target_audience, perspective, resolved_depth)
    api_keys = {
        "Gemini (Free)": gemini_api_key,
        "Groq (Free)": groq_api_key,
        "OpenRouter (Free)": openrouter_api_key,
    }
    post, provider = generate_with_fallback_chain(prompt, api_keys)
    return GenerationResult(post=post, model_used="Auto", provider=provider)


def render_copy_button(text: str) -> None:
    payload = (
        json.dumps(text, ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("/", "\\u002f")
    )
    components.html(
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
    st.title("\U0001F680 LinkedIn Content Generator Pro")
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
        perspective = st.selectbox("Perspective", ["Leader", "Practitioner", "Advisor", "Storyteller"])
    with depth_col:
        technical_depth = st.selectbox("Technical Depth", ["Auto", "Non-Technical", "Balanced", "Highly Technical"])

    resolved_depth = resolve_depth(technical_depth, target_audience)
    if technical_depth == "Auto":
        st.caption(f"Auto-selected depth: {resolved_depth}")

    any_key_ready = (
        is_valid_gemini_key(ENV_GEMINI_API_KEY)
        or is_valid_groq_key(ENV_GROQ_API_KEY)
        or is_valid_openrouter_key(ENV_OPENROUTER_API_KEY)
    )
    topic_missing = not topic.strip()
    generate_disabled = not any_key_ready or topic_missing

    if not any_key_ready:
        _, message_col, _ = st.columns([1, 2, 1])
        with message_col:
            st.warning("\U0001F511 No API key found. Set GEMINI_API_KEY, GROQ_API_KEY, or OPENROUTER_API_KEY in your .env file.")
    elif topic_missing:
        st.warning("Please enter a topic to generate a LinkedIn post")
    else:
        st.success("Ready")

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
                start_time = time.time()
                result = generate_post(
                    topic.strip(), tone, length, target_audience, perspective,
                    technical_depth,
                    gemini_api_key=ENV_GEMINI_API_KEY,
                    groq_api_key=ENV_GROQ_API_KEY,
                    openrouter_api_key=ENV_OPENROUTER_API_KEY,
                )
                latency = time.time() - start_time
                score, suggestion = score_post(result.post)
                st.session_state["last_result"] = result
                st.session_state["last_score"] = score
                st.session_state["last_suggestion"] = suggestion
                st.session_state["last_latency"] = latency
                st.session_state["last_inputs"] = {
                    "topic": topic.strip(),
                    "tone": tone,
                    "length": length,
                    "target_audience": target_audience,
                    "perspective": perspective,
                    "technical_depth": technical_depth,
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
        regen_count = st.session_state.get("regen_count", 0)

        left, right = st.columns([2, 1])
        with left:
            st.subheader("Generated Post")
            st.success(f"Generated using: {result.provider}")
            st.caption(f"Execution Path: {result.provider} -> {result.model_used}")
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
                            "Rewrite this post with a different hook, structure, and example "
                            "while keeping the same core idea. Avoid repeating phrases from "
                            f"the previous version.\nVariation seed: {variation_seed}"
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
                                    gemini_api_key=ENV_GEMINI_API_KEY,
                                    groq_api_key=ENV_GROQ_API_KEY,
                                    openrouter_api_key=ENV_OPENROUTER_API_KEY,
                                )
                                regenerated_latency = time.time() - start_time
                                regenerated_score, regenerated_suggestion = score_post(regenerated_result.post)
                                st.session_state["last_result"] = regenerated_result
                                st.session_state["last_score"] = regenerated_score
                                st.session_state["last_suggestion"] = regenerated_suggestion
                                st.session_state["last_latency"] = regenerated_latency
                                st.session_state["last_regenerated_latency"] = regenerated_latency
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

