"""
llm_explain.py — LLM Integration for Natural Language Explanations

Generates a human-readable 2-sentence explanation of why the GNN classified
a paper the way it did, based on GNNExplainer + attention weight output.

Priority:
  1. xAI (Grok) API  — OpenAI-compatible, $25 free credit (XAI_API_KEY)
  2. Groq API        — free tier, llama-3.3-70b  (GROQ_API_KEY)
  3. Gemini REST API — fallback                  (GEMINI_API_KEY)
  4. Template        — always works, no key needed

Setup (.env file):
    XAI_API_KEY=xai-your_key_here     ← get at https://console.x.ai/
    GROQ_API_KEY=gsk_your_key_here    ← get at https://console.groq.com/keys

Usage:
    from llm_explain import generate_explanation
    text = generate_explanation(
        node_id=142,
        predicted_class="Reinforcement Learning",
        confidence=87.3,
        top_words=["reward", "policy", "agent"],
        top_neighbor_classes=["Reinforcement Learning", "Neural Networks"],
    )
"""

from __future__ import annotations

import os
from typing import List


# ── API key loading ────────────────────────────────────────────────────────────
def _load_key(var_name: str) -> str | None:
    """Load an API key from environment or .env file."""
    key = os.environ.get(var_name)
    if key:
        return key
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{var_name}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


# ── Prompt engineering ─────────────────────────────────────────────────────────
def _build_prompt(
    node_id: int,
    predicted_class: str,
    confidence: float,
    top_words: List[str],
    top_neighbor_classes: List[str],
) -> str:
    """
    Engineered prompt to produce a concise, grounded, non-hallucinating explanation.
    """
    words_str = ", ".join(f'"{w}"' for w in top_words)
    neighbors_str = ", ".join(top_neighbor_classes) if top_neighbor_classes else predicted_class

    return f"""You are an AI assistant explaining a Graph Neural Network's classification decision to a student.

The GNN analyzed Paper #{node_id} from the Cora citation network and classified it as "{predicted_class}" with {confidence:.1f}% confidence.

Evidence used by the GNN:
- Top content keywords in this paper: {words_str}
- Research categories of its most-cited neighbors: {neighbors_str}

Write EXACTLY 2 sentences explaining why this paper is classified as "{predicted_class}".
- Sentence 1: Use the paper's keywords as evidence.
- Sentence 2: Use the citation neighborhood as evidence.
- Write for a data mining student. Keep it clear and concise.
- Do NOT use the words "GNN", "model", "algorithm", or "training".
- Do NOT start both sentences with "This paper"."""


# ── xAI (Grok) call — primary ─────────────────────────────────────────────────
def _call_xai(prompt: str, api_key: str) -> str:
    """
    Call xAI API using OpenAI-compatible client.
    Model: grok-3-mini (fast + cheap) with grok-2 as fallback.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

        for model in ["grok-3-mini", "grok-2-1212"]:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=200,
                )
                text = response.choices[0].message.content.strip()
                if text:
                    return text
            except Exception as model_err:
                err_str = str(model_err).lower()
                if "rate" in err_str or "quota" in err_str or "404" in err_str:
                    continue
                raise

        return "[xAI: all models unavailable]"
    except ImportError:
        return "[openai SDK not installed — run: pip install openai]"
    except Exception as e:
        return f"[xAI error: {str(e)}]"


# ── Groq call — secondary fallback ────────────────────────────────────────────
def _call_groq(prompt: str, api_key: str) -> str:
    """
    Call Groq API synchronously.
    Model: llama-3.3-70b-versatile (free tier, very fast).
    """
    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        for model in ["llama-3.3-70b-versatile", "llama3-8b-8192", "gemma2-9b-it"]:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=200,
                    top_p=0.8,
                )
                text = response.choices[0].message.content.strip()
                if text:
                    return text
            except Exception as model_err:
                err_str = str(model_err).lower()
                if "rate" in err_str or "quota" in err_str:
                    continue
                raise

        return "[Groq: all models rate-limited]"
    except ImportError:
        return "[Groq SDK not installed]"
    except Exception as e:
        return f"[Groq error: {str(e)}]"


# ── Gemini REST fallback ───────────────────────────────────────────────────────
def _call_gemini_rest(prompt: str, api_key: str) -> str:
    """Fallback Gemini REST call (synchronous via requests)."""
    try:
        import urllib.request
        import json
        import time

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 200, "topP": 0.8},
        }).encode()

        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return f"[Gemini REST error: {str(e)}]"


# ── Template fallback (always works) ──────────────────────────────────────────
def _template_explanation(
    node_id: int,
    predicted_class: str,
    confidence: float,
    top_words: List[str],
    top_neighbor_classes: List[str],
) -> str:
    words_str = ", ".join(top_words[:3]) if top_words else "domain-specific terminology"
    neighbor_str = top_neighbor_classes[0] if top_neighbor_classes else predicted_class

    return (
        f"Paper #{node_id} was classified as \"{predicted_class}\" ({confidence:.0f}% confidence) "
        f"because it heavily uses terms like {words_str}, which are characteristic vocabulary "
        f"in {predicted_class} research. "
        f"Furthermore, the papers it cites are predominantly about {neighbor_str}, "
        f"placing it firmly within the same research community.\n\n"
        f"*(Add GROQ_API_KEY to your .env file for AI-generated explanations — "
        f"free at https://console.groq.com/keys)*"
    )


# ── Public API ─────────────────────────────────────────────────────────────────
def generate_explanation(
    node_id: int,
    predicted_class: str,
    confidence: float,
    top_words: List[str],
    top_neighbor_classes: List[str],
) -> str:
    """
    Generate a natural language explanation for a GNN classification decision.

    Priority:
      1. Groq API (free, fast — set GROQ_API_KEY in .env)
      2. Gemini REST API (set GEMINI_API_KEY in .env)
      3. Template fallback (always works)
    """
    prompt = _build_prompt(node_id, predicted_class, confidence, top_words, top_neighbor_classes)

    # ── 1. xAI / Grok (primary) ────────────────────────────────────────────
    xai_key = _load_key("XAI_API_KEY")
    if xai_key:
        result = _call_xai(prompt, xai_key)
        if not result.startswith("["):
            return result

    # ── 2. Groq (secondary) ────────────────────────────────────────────────
    groq_key = _load_key("GROQ_API_KEY")
    if groq_key:
        result = _call_groq(prompt, groq_key)
        if not result.startswith("["):
            return result

    # ── 3. Gemini REST (tertiary) ──────────────────────────────────────────
    gemini_key = _load_key("GEMINI_API_KEY")
    if gemini_key:
        result = _call_gemini_rest(prompt, gemini_key)
        if not result.startswith("["):
            return result

    # ── 4. Template fallback ───────────────────────────────────────────────
    return _template_explanation(node_id, predicted_class, confidence, top_words, top_neighbor_classes)


# ── CLI test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing LLM explanation pipeline...")
    xai_key = _load_key("XAI_API_KEY")
    groq_key = _load_key("GROQ_API_KEY")
    gemini_key = _load_key("GEMINI_API_KEY")
    print(f"  XAI_API_KEY:    {'✅ found' if xai_key else '❌ not set'}")
    print(f"  GROQ_API_KEY:   {'✅ found' if groq_key else '❌ not set'}")
    print(f"  GEMINI_API_KEY: {'✅ found' if gemini_key else '❌ not set'}")
    print()

    explanation = generate_explanation(
        node_id=42,
        predicted_class="Reinforcement Learning",
        confidence=87.3,
        top_words=["reward", "policy", "agent", "action", "state"],
        top_neighbor_classes=["Reinforcement Learning", "Neural Networks", "Probabilistic Methods"],
    )
    print("=" * 60)
    print("Generated Explanation:")
    print("=" * 60)
    print(explanation)
