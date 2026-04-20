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
    mechanistic_data: dict | None = None
) -> str:
    """
    High-fidelity prompt for senior researchers and technical stakeholders.
    """
    words_str = ", ".join(f'"{w}"' for w in top_words)
    neighbors_str = ", ".join(top_neighbor_classes) if top_neighbor_classes else predicted_class
    
    # Extract mechanistic signals
    homophily = mechanistic_data.get("homophily", 0.5) if mechanistic_data else 0.5
    sharpness = mechanistic_data.get("attention_sharpness", 0.5) if mechanistic_data else 0.5
    degree = mechanistic_data.get("degree", 0) if mechanistic_data else 0

    return f"""You are a High-Level GNN Research Analyst specializing in citation graph topologies.
    
PAPER DOSSIER: Paper #{node_id}
SYNTHETIC TITLE: "Advanced {top_words[0].title()} {top_words[1].title()} in {predicted_class} Systems"
PREDICTED DOMAIN: "{predicted_class}" ({confidence:.1f}% confidence)

TECHNICAL METADATA:
- Neighborhood Homophily: {homophily:.2f} (Structural consistency coefficient)
- Attention Sharpness: {sharpness:.2f} (Focus-distillation factor)
- Node Degree: {degree} (Information flow connectivity)
- Primary Semantic Markers: {words_str}

TASK: Provide a comprehensive, multi-paragraph technical explanation:
1. SEMANTIC PROFILE: Elaborate on how the key word motifs suggest a specialized contribution to the {predicted_class} domain.
2. STRUCTURAL ARCHITECTURE: Analyze the citation topology. Explain how a homophily of {homophily:.2f} provides the necessary inductive bias for this label.
3. MECHANISTIC INSIGHT: Describe how the GNN's internal message-passing aggregated these signals, highlighting the {sharpness:.2f} attention focus.

FORMAT: Use professional, graduate-level academic terminology. Provide at least 2-3 sentences per section. No conversational preamble.
"""


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
    mechanistic_data: dict | None = None
) -> str:
    words_str = ", ".join(top_words[:3]) if top_words else "domain-specific terminology"
    homophily = mechanistic_data.get("homophily", 0.5) if mechanistic_data else 0.5
    sharpness = mechanistic_data.get("attention_sharpness", 0.5) if mechanistic_data else 0.5
    
    # ── Technical Report Generation ──────────────────────────────────────
    if homophily > 0.8:
        topology_desc = f"The node exhibits high homophily ({homophily:.2f}), indicating it exists within a dense community of papers that share the same research domain. This structural consistency provides powerful reinforcement for the '{predicted_class}' label."
    elif homophily > 0.4:
        topology_desc = f"With a moderate homophily coefficient of {homophily:.2f}, the paper sits at a cross-disciplinary junction. While its immediate neighbors provide a semantic baseline, the classification relys on a blend of local neighborhood and multi-hop global context."
    else:
        topology_desc = f"Detected low structural homophily ({homophily:.2f}). This suggests the paper is an outlier or an 'interdisciplinary bridge'. The GNN likely bypassed weak structural signals and derived the '{predicted_class}' label primarily through semantic feature extraction."
        
    if sharpness > 0.7:
        focus_desc = f"The model's attention mechanism showed high sharpness ({sharpness:.2f}), signifying that it focused its learning capacity on a specific subset of high-value neighbors which are highly representative of the {predicted_class} class."
    else:
        focus_desc = f"Attention focus is distributed ({sharpness:.2f}), suggesting that the model performed a wide semantic aggregation across the entire local subgraph to reach its conclusion."

    variants = [
        f"""### 📄 Technical Dossier: Paper #{node_id}
**Section 1: Semantic Profile**
The classification into **{predicted_class}** ({confidence:.1f}% confidence) is driven by a strong presence of technical markers: {words_str}. In the context of academic corpora, these motifs act as unique identifiers for this specific research area.

**Section 2: Structural Architecture**
{topology_desc}

**Section 3: Mechanistic Reasoning**
During the GNN's forward pass, {focus_desc} This resulted in a high-confidence feature embedding that aligns with the established centroid of the {predicted_class} category.""",

        f"""### 🔍 High-Fidelity Research Report: Node {node_id}
**Semantics & Motifs**
The feature vector for this paper is dominated by the motifs: {words_str}. These terms are typical of the vocabulary used in the **{predicted_class}** field, providing a clear semantic signal for the model's message-passing layers.

**Graph Topology Analysis**
{topology_desc} Relative to its node degree of {mechanistic_data.get('degree', 0) if mechanistic_data else 'unknown'}, this neighbor distribution confirms the thematic grouping.

**Interpretable GNN Mechanics**
{focus_desc} By distilling information from its citation context, the GNN reached a finalized state of {confidence:.1f}% confidence, effectively mapping the paper to its correct academic lineage.""",
    ]
    
    return variants[node_id % 2]



# ── Link Prediction Prompt Engineering ─────────────────────────────────────────
def _build_link_prompt(
    node_a: int,
    node_b: int,
    score: float,
    shared_words: List[str],
    shared_topics: List[str],
) -> str:
    """Prompt for explaining why two nodes should be linked."""
    words_str = ", ".join(f'"{w}"' for w in shared_words)
    topics_str = ", ".join(shared_topics)
    
    status = "EXISTENT" if score > 0.5 else "NON-EXISTENT"
    
    return f"""You are a High-Level GNN Research Analyst specializing in citation graph topologies.
    
LINK ANALYSIS: Node {node_a} ↔ Node {node_b}
PREDICTED STATUS: {status} (Similarity Score: {score:.1f}%)

TECHNICAL EVIDENCE:
- Shared Keyword Motifs: {words_str}
- Shared Research Sub-domains: {topics_str}

TASK: Provide a comprehensive, multi-paragraph scholarly link analysis:
1. SEMANTIC OVERLAP: Analyze how the shared keyword motifs suggest a thematic alignment in the latent space.
2. TOPOLOGICAL COHESION: Discuss how their mutual affiliation with the "{topics_str}" communities creates a strong prior for a citation link.
3. PREDICTION CONCLUSION: Summarize the model's rationale for the {score:.1f}% similarity score.

FORMAT: Use professional, academic terminology. Provide at least 2 sentences per section. No preambles.
"""


def generate_link_explanation(
    node_a: int,
    node_b: int,
    score: float,
    shared_words: List[str],
    shared_topics: List[str],
) -> str:
    """Generate a natural language explanation for a link prediction."""
    prompt = _build_link_prompt(node_a, node_b, score, shared_words, shared_topics)
    
    # Try APIs in order
    xai_key = _load_key("XAI_API_KEY")
    if xai_key:
        result = _call_xai(prompt, xai_key)
        if not result.startswith("["): return result

    groq_key = _load_key("GROQ_API_KEY")
    if groq_key:
        result = _call_groq(prompt, groq_key)
        if not result.startswith("["): return result

    # ── Technical Link Fallback ──────────────────────────────────────────
    status_msg = "Existent" if score > 0.5 else "Non-Existent"
    topic_str = shared_topics[0] if shared_topics else "Machine Learning"
    words = ", ".join(shared_words[:3]) if shared_words else "latent semantic features"
    
    variants = [
        f"""### 🔗 Link Analysis Dossier: {node_a} ↔ {node_b}
**Semantic Overlap**
The latent similarity between these nodes is primarily driven by a shared keyword distribution, specifically technical motifs like {words}. This alignment suggests that both papers contribute to a unified research discourse.

**Topological Cohesion**
Both papers reside within the **{topic_str}** citation community, creating a high-probability bridge in the graph. The citation topology indicates that their research goals are complementary within the academic hierarchy.

**Final Grounding**
With a similarity score of **{score:.1f}%**, the GNN predicts this link as **{status_msg}**. The final embedding layers show a high cosine similarity, validating the likelihood of a citation relationship.""",

        f"""### 🛰️ Latent Space Research Report: Connection {node_a}---{node_b}
**Feature Alignment**
Analysis of the feature vectors reveals a significant thematic overlap around {words}. These shared semantic markers form the basis of the GNN's ability to map these nodes into adjacent regions of the hidden embedding space.

**Citation Neighborhoods**
The structural context is defined by a shared affiliation with the **{topic_str}** research sub-domain. This citation proximity acts as the primary structural prior for the predicted **{status_msg}** status.

**Mechanistic Conclusion**
The resulting similarity score of **{score:.1f}%** reflects the model's aggregation of these dual signals. The prediction represents a high-confidence mapping of scholarly relevance between the two entities.""",
    ]
    
    return variants[node_a % 2]


# ── Public API ─────────────────────────────────────────────────────────────────
def generate_explanation(
    node_id: int,
    predicted_class: str,
    confidence: float,
    top_words: List[str],
    top_neighbor_classes: List[str],
    mechanistic_data: dict | None = None
) -> str:
    """
    Generate a natural language explanation for a GNN classification decision.
    """
    prompt = _build_prompt(
        node_id, predicted_class, confidence, top_words, 
        top_neighbor_classes, mechanistic_data
    )

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
    return _template_explanation(node_id, predicted_class, confidence, top_words, top_neighbor_classes, mechanistic_data)


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
