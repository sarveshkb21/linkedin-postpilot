import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from threading import Lock, Thread
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
load_dotenv()
GEMINI_MODEL = "gemini-2.5-flash"
WORD_COUNT_RANGES: dict[str, tuple[int, int]] = {
    "Short": (90, 130),
    "Medium": (140, 210),
    "Long": (220, 320),
}
GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
]
OPENROUTER_MODELS = [
    # ⚡ Fast (Primary) — used for all prompts
    "mistralai/mistral-7b-instruct",
    "openchat/openchat-7b",
    "meta-llama/llama-3.2-3b-instruct",
    "google/gemma-3-4b",
    # ⚖️ Balanced (Fallback) — added to pool for prompts ≥ 800 chars
    "google/gemma-3-12b",
    "z-ai/glm-4.5-air",
    "minimax/minimax-m2.5",
    "meta-llama/llama-3.3-70b-instruct",
]
ENV_GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "")).strip()
ENV_GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "")).strip()
ENV_OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY", "")).strip()

_REQUEST_CACHE: dict = {}
_REQUEST_CACHE_LOCK = Lock()
MAX_CACHE_SIZE = 100

# These dicts are intentionally process-level (shared across all Streamlit sessions)
# so that one user's rate-limit failures act as a circuit breaker for all users,
# avoiding thundering-herd retries against a provider that is already degraded.
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
    provider: str
    model: str = ""


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
        except Exception as exc:
            if attempt == retries - 1:
                raise exc

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


def call_with_timeout(func, timeout: int = 30):
    result: list = [None]
    exc: list = [None]

    def target():
        try:
            result[0] = func()
        except Exception as e:
            exc[0] = e

    t = Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        # Thread is still running; it will be cleaned up when the underlying
        # SDK/HTTP call eventually times out. Daemon flag ensures it won't
        # block process exit.
        raise TimeoutException("Request timed out")
    if exc[0] is not None:
        raise exc[0]
    return result[0]


def generate_with_control(func, timeout: int = 35):
    return call_with_timeout(lambda: retry_call(func), timeout=timeout)


def show_generation_error(exc: Exception) -> None:
    message = str(exc)
    lower_message = message.lower()
    print(f"Generation error: {message}", file=sys.stderr)
    if isinstance(exc, TimeoutException) or "timeout" in lower_message or "timed out" in lower_message:
        st.error("Request timed out. Please try again.")
    elif "authentication" in lower_message or "unauthorized" in lower_message or "api key" in lower_message:
        st.error("Authentication failed. Please verify your API key.")
    elif "all providers failed" in lower_message:
        # Extract the provider list from the message, e.g. "[Gemini (Free), Groq (Free)]"
        match = re.search(r"\[([^\]]+)\]", message)
        providers_tried = match.group(1) if match else "all configured providers"
        st.error(
            f"Generation failed after trying {providers_tried}. "
            "Check that your API keys are valid and you haven't exceeded rate limits. "
            "Try again in a few seconds."
        )
    else:
        st.error("Generation failed. Please try again.")


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


def tone_instructions(tone: str) -> str:
    instructions = {
        "Professional": "Use clear, measured language. Credible and direct, no hype or filler.",
        "Conversational": "Write as if speaking to a peer. Warm, natural, and first-person throughout.",
        "Thought Leadership": "Take a clear stance. Challenge assumptions and offer a distinct, defensible point of view.",
        "Bold": "Open with a provocative or counterintuitive statement. Be direct, confident, and unafraid to polarise.",
        "Educational": "Explain clearly, one idea at a time. Prioritise understanding over persuasion.",
        "Persuasive": "Build a logical case with evidence and specific examples. Lead the reader to an obvious conclusion.",
    }
    return instructions.get(tone, instructions["Professional"])


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


def build_prompt(topic: str, tone: str, length: str, target_audience: str, perspective: str, depth: str, instruction: str = "") -> str:
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
    instruction_block = f"\nAdditional instruction: {instruction}\n" if instruction else ""
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
Tone guidance:
{tone_instructions(tone)}
Length:
{length_rules.get(length, length_rules["Medium"])}
Technical depth:
{depth}
Technical depth guidance:
{depth_rules.get(depth, depth_rules["Balanced"])}
Strict writing rules:
- Write in first person (I / we). Speak from direct experience or a clear personal point of view.
- Start with a strong hook in the first line using one of: a counterintuitive statement, a bold claim, a specific number or statistic, a relatable frustration, or a short provocative question.
- Keep the first line under 120 characters — LinkedIn truncates to "...see more" after this, so it must stand alone and compel a click.
- Structure the post as: Hook (first line) → Context or problem → Key insight or lesson → Call to action. Expand body sections proportionally to the target length.
- End with a clear call to action.
- Keep paragraphs to 1–3 sentences, separated by blank lines.
- Do not use markdown formatting.
- Avoid LinkedIn clichés: "thrilled/humbled/excited to share", "game changer", "hot take", "unpopular opinion", "this is so important", "let that sink in".
- Use emojis sparingly and only where they genuinely add clarity or emphasis. Never force them.
- Use bullet points when listing multiple ideas, steps, or comparisons. Choose one style (e.g., '-', '•', or numbered) and use it consistently. Keep each bullet to one line. Do not overuse bullets; maintain a natural LinkedIn flow.
- End with 3 to 5 relevant hashtags on their own line, each starting with # (e.g., #AI #CloudComputing). Never write plain keywords without the # prefix.
- Keep the post human and credible. Include at least one concrete detail: a specific number, a real scenario, a named tool, or a tangible outcome. Avoid generic claims.
{instruction_block}- Return only the LinkedIn post text.
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
        is_bullet = bool(re.match(r"^[-•\d]", line))
        if formatted:
            prev_line = formatted[-1]
            prev_is_bullet = bool(re.match(r"^[-•\d]", prev_line))
            if not is_bullet and not prev_is_bullet:
                formatted.append("")
            if not is_bullet and prev_is_bullet:
                formatted.append("")
            if is_bullet and not prev_is_bullet:
                formatted.append("")
        formatted.append(line)
    return "\n".join(formatted).strip()


def enforce_hashtags(text: str, topic: str = "") -> str:
    text = text.strip()
    hashtags = list(dict.fromkeys(re.findall(r"#\w+", text)))
    if len(hashtags) < 3:
        # Prefer topic-derived tags; fall back to generic ones only if needed
        topic_tags = [
            f"#{w.capitalize()}"
            for w in re.findall(r"\b[A-Za-z]{4,}\b", topic)
        ]
        candidates = topic_tags + ["#Leadership", "#Innovation", "#Growth"]
        existing_lower = {h.lower() for h in hashtags}
        for tag in candidates:
            if tag.lower() not in existing_lower:
                hashtags.append(tag)
                existing_lower.add(tag.lower())
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


def score_post(text: str, length: str = "Medium") -> tuple[int, str]:
    score = 0
    suggestions = []

    lo, hi = WORD_COUNT_RANGES.get(length, WORD_COUNT_RANGES["Medium"])
    word_count = len(text.split())
    if lo <= word_count <= hi:
        score += 2
    else:
        suggestions.append(f"Adjust length for better engagement ({lo}–{hi} words).")

    first_line = text.split("\n")[0]
    if len(first_line) > 20:
        score += 2
    else:
        suggestions.append("Improve the opening hook to grab attention.")

    hashtags = re.findall(r"#\w+", text)
    if 3 <= len(hashtags) <= 5:
        score += 2
    else:
        suggestions.append("Use 3–5 relevant hashtags.")

    if "\n\n" in text:
        score += 2
    else:
        suggestions.append("Add spacing for better readability.")

    lower_text = text.lower()
    if any(phrase in lower_text for phrase in _CTA_PHRASES):
        score += 2
    else:
        suggestions.append("Consider adding a call-to-action.")

    final_score = min(score, 10)
    suggestion_text = (
        " ".join(suggestions) if suggestions
        else "Strong post. Well structured and engaging."
    )
    return final_score, suggestion_text


def generate_with_gemini(prompt: str, api_key: str) -> tuple[str, str]:
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
        return response.text, GEMINI_MODEL
    if hasattr(response, "candidates") and response.candidates:
        parts = getattr(response.candidates[0].content, "parts", [])
        if parts and getattr(parts[0], "text", None):
            return parts[0].text, GEMINI_MODEL
    raise RuntimeError("Gemini returned an empty response.")


def generate_with_groq(prompt: str, api_key: str) -> tuple[str, str]:
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
                return content, model
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"Groq failed after trying {model_attempts}: {last_error}")


def generate_with_openrouter(prompt: str, api_key: str) -> tuple[str, str]:
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
        "max_tokens": 512,
    }

    last_error = None
    FAST_MODELS = OPENROUTER_MODELS[:4]
    BALANCED_MODELS = OPENROUTER_MODELS[4:8]

    selected_models = FAST_MODELS if len(prompt) < 800 else FAST_MODELS + BALANCED_MODELS
    with _STATE_LOCK:
        usable_models = [m for m in selected_models if _MODEL_HEALTH.get(m, 0) <= 2]
        if not usable_models:
            usable_models = FAST_MODELS[:2]
        sorted_models = sorted(
            usable_models,
            key=lambda m: (_MODEL_HEALTH.get(m, 0), _PROVIDER_LATENCY.get(m, 999))
        )

    model_attempts = []
    for model_name in sorted_models:
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
                return content, model_name
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


def generate_with_fallback_chain(prompt: str, api_keys: dict[str, str], topic: str = "") -> tuple[str, str, str]:
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    cached = get_cached_response(prompt_hash)
    if cached:
        return cached, "Cache", ""

    with _IN_PROGRESS_LOCK:
        in_progress = prompt_hash in _IN_PROGRESS
        if not in_progress:
            _IN_PROGRESS.add(prompt_hash)

    if in_progress:
        # Another request for the same prompt is already in flight.
        # Wait up to 38 s for it to complete, then use its cached result.
        deadline = time.time() + 38
        while time.time() < deadline:
            time.sleep(0.5)
            with _IN_PROGRESS_LOCK:
                still_running = prompt_hash in _IN_PROGRESS
            if not still_running:
                break
        cached = get_cached_response(prompt_hash)
        if cached:
            return cached, "Cache", ""
        # In-flight request finished but produced no cache entry (it failed).
        # Fall through and try independently — register ourselves in-progress first.
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
                raw, model = generate_with_control(lambda f=func, k=api_key: f(prompt, k))
                latency = time.time() - start_time
                post = enforce_hashtags(clean_post(raw), topic)
                cache_response(prompt_hash, post)
                with _STATE_LOCK:
                    _PROVIDER_LATENCY[provider_name] = latency
                    _PROVIDER_HEALTH[provider_name] = 0
                return post, provider_name, model
            except Exception as e:
                print(f"{provider_name} failed: {str(e)[:80]}", file=sys.stderr)
                with _STATE_LOCK:
                    _PROVIDER_HEALTH[provider_name] = _PROVIDER_HEALTH.get(provider_name, 0) + 1
                last_error = e
                continue
        tried = [name for name, _ in providers_to_try if api_keys.get(name)]
        raise RuntimeError(f"All providers failed [{', '.join(tried)}]: {last_error}")
    finally:
        with _IN_PROGRESS_LOCK:
            _IN_PROGRESS.discard(prompt_hash)


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
    instruction: str = "",
) -> GenerationResult:
    resolved_depth = resolve_depth(technical_depth, target_audience)
    prompt = build_prompt(topic, tone, length, target_audience, perspective, resolved_depth, instruction)
    api_keys = {
        "Gemini (Free)": gemini_api_key,
        "Groq (Free)": groq_api_key,
        "OpenRouter (Free)": openrouter_api_key,
    }
    post, provider, model = generate_with_fallback_chain(prompt, api_keys, topic=topic)
    return GenerationResult(post=post, provider=provider, model=model)


def render_copy_button(text: str, button_id: str = "copy-btn") -> None:
    # Text is stored in a data-attribute (no user content in script body).
    # " is encoded as &quot; so it's safe inside a double-quoted HTML attribute.
    # JS uses JSON.parse to recover the original string after the browser
    # HTML-decodes &quot; back to ", making the round-trip lossless.
    safe_attr = json.dumps(text, ensure_ascii=True).replace('"', '&quot;')
    components.html(
        f"""
        <button id="{button_id}" data-text="{safe_attr}" style="
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
        const button = document.getElementById("{button_id}");
        button.onclick = async () => {{
            await navigator.clipboard.writeText(JSON.parse(button.dataset.text));
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
            key="pref_tone",
        )
    with audience_col:
        target_audience = st.selectbox("Target Audience", ["Executives", "Managers", "Engineers", "General Audience"], key="pref_audience")
    with length_col:
        length = st.selectbox("Length", ["Short", "Medium", "Long"], index=1, key="pref_length")
    with perspective_col:
        perspective = st.selectbox("Perspective", ["Leader", "Practitioner", "Advisor", "Storyteller"], key="pref_perspective")
    with depth_col:
        technical_depth = st.selectbox("Technical Depth", ["Auto", "Non-Technical", "Balanced", "Highly Technical"], key="pref_depth")

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

    generate = st.button(
        "Generate LinkedIn Post",
        type="primary",
        use_container_width=True,
        disabled=generate_disabled,
    )

    if not any_key_ready:
        st.warning("\U0001F511 No API key found. Set GEMINI_API_KEY, GROQ_API_KEY, or OPENROUTER_API_KEY in your .env file.")
    elif topic_missing:
        st.info("Enter a topic above to get started.")
    else:
        st.success("✓ Ready to generate")

    if generate:
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
                score, suggestion = score_post(result.post, length)
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
                st.session_state.pop("previous_result", None)
                history = st.session_state.get("post_history", [])
                history.insert(0, {"post": result.post, "provider": result.provider, "score": score, "latency": latency})
                st.session_state["post_history"] = history[:10]
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
        previous_result: GenerationResult | None = st.session_state.get("previous_result")

        left, right = st.columns([2, 1])
        with left:
            st.subheader("Generated Post")
            provider_label = f"Generated using: {result.provider}"
            if result.model:
                provider_label += f"  •  Model: {result.model}"
            st.success(provider_label)
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
                        variation_instruction = (
                            "Rewrite this post with a different hook, structure, and example "
                            "while keeping the same core idea. Avoid repeating phrases from "
                            f"the previous version. Variation seed: {variation_seed}"
                        )
                        with st.spinner("Regenerating your LinkedIn post..."):
                            try:
                                start_time = time.time()
                                regenerated_result = generate_post(
                                    original_topic,
                                    last_inputs["tone"],
                                    last_inputs["length"],
                                    last_inputs["target_audience"],
                                    last_inputs["perspective"],
                                    last_inputs["technical_depth"],
                                    gemini_api_key=ENV_GEMINI_API_KEY,
                                    groq_api_key=ENV_GROQ_API_KEY,
                                    openrouter_api_key=ENV_OPENROUTER_API_KEY,
                                    instruction=variation_instruction,
                                )
                                regenerated_latency = time.time() - start_time
                                regenerated_score, regenerated_suggestion = score_post(regenerated_result.post, last_inputs["length"])
                                st.session_state["previous_result"] = result
                                st.session_state["last_result"] = regenerated_result
                                st.session_state["last_score"] = regenerated_score
                                st.session_state["last_suggestion"] = regenerated_suggestion
                                st.session_state["last_latency"] = regenerated_latency
                                st.session_state["last_regenerated_latency"] = regenerated_latency
                                st.session_state["regen_count"] = st.session_state.get("regen_count", 0) + 1
                                history = st.session_state.get("post_history", [])
                                history.insert(0, {"post": regenerated_result.post, "provider": regenerated_result.provider, "score": regenerated_score, "latency": regenerated_latency})
                                st.session_state["post_history"] = history[:10]
                                st.rerun()
                            except Exception as exc:
                                show_generation_error(exc)

            with copy_col:
                render_copy_button(result.post)

            if previous_result:
                st.divider()
                st.caption("Comparing with previous version")
                prev_col, new_col = st.columns(2)
                with prev_col:
                    st.caption("**Previous**")
                    render_copy_button(previous_result.post, button_id="copy-btn-prev")
                    st.text_area("Previous", previous_result.post, height=300, label_visibility="collapsed", key="prev_output")
                with new_col:
                    st.caption("**New**")
                    render_copy_button(result.post, button_id="copy-btn-new")
                    st.text_area("New", result.post, height=300, label_visibility="collapsed", key="new_output")
                if st.button("Dismiss comparison"):
                    st.session_state.pop("previous_result", None)
                    st.rerun()
            else:
                st.text_area("Output", result.post, height=380, label_visibility="collapsed")

            char_count = len(result.post)
            if char_count < 1300:
                st.markdown(f'<span style="color:#28a745">✓ {char_count} characters — optimal for engagement</span>', unsafe_allow_html=True)
            elif char_count <= 3000:
                st.markdown(f'<span style="color:#fd7e14">⚠ {char_count} characters — within LinkedIn limit</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span style="color:#dc3545">✗ {char_count} characters — exceeds LinkedIn\'s 3,000 character limit</span>', unsafe_allow_html=True)

        with right:
            st.subheader("Generation Details")
            st.metric("Engagement Score", f"{score}/10")
            st.write(suggestion)
            with st.expander("How is this scored?"):
                st.caption("Each criterion adds 2 points (max 10):")
                st.caption("• Word count within target range")
                st.caption("• Opening hook longer than 20 characters")
                st.caption("• 3–5 relevant hashtags present")
                st.caption("• Paragraph spacing (blank lines) present")
                st.caption("• Call-to-action detected")
            st.divider()
            st.write("Provider")
            provider_text = result.provider
            if result.model:
                provider_text += f"  •  Model: {result.model}"
            st.success(provider_text)

        post_history = st.session_state.get("post_history", [])
        if len(post_history) > 1:
            with st.expander(f"\U0001F4CB Post History ({len(post_history)} posts)"):
                for i, entry in enumerate(post_history):
                    label = f"#{i + 1} · {entry['provider']} · Score: {entry['score']}/10"
                    if entry.get("latency"):
                        label += f" · {entry['latency']:.1f}s"
                    st.caption(label)
                    st.text_area(
                        f"history_{i}",
                        entry["post"],
                        height=150,
                        label_visibility="collapsed",
                        key=f"history_area_{i}",
                    )
                    render_copy_button(entry["post"], button_id=f"copy-btn-hist-{i}")
                    if i < len(post_history) - 1:
                        st.divider()


if __name__ == "__main__":
    main()

