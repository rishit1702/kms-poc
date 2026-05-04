"""
Chat orchestrator with advanced context engineering.

Pipeline:
  user query
    -> query expansion (LLM rewrites into 3 search variants)
    -> multi-query retrieval (search KB with each variant)
    -> dedupe + rerank by best distance per chunk
    -> distance-based relevance filtering
    -> structured context assembly (per-source attribution)
    -> LLM synthesis with multi-source instruction
    -> answer + cited sources
"""
from typing import List, Dict
from app.services.knowledge_base import kb
from app.services.llm import get_llm


SYSTEM_PROMPT = """You are the VVDN Knowledge Management System Assistant - an internal AI assistant for VVDN Technologies employees.

# Your Role
You answer employee questions using ONLY the KNOWLEDGE BASE CONTEXT provided. Topics include:
- Strategic plans, project briefs, and roadmaps
- HR policies, onboarding info, leave and reimbursement
- Engineering standards, tech stacks, security guidelines
- Team rosters, meeting notes, and 1:1 topics
- Internal AI tooling (KMS, Recruitment Bot, GPU cluster)

# Strict Rules
1. Answer ONLY from the KNOWLEDGE BASE CONTEXT below. Never invent facts, dates, names, or numbers.
2. If multiple sources contain relevant info, SYNTHESIZE across them. Do not just quote one source - combine.
3. Cite the source filename inline after each factual claim, in the format: (Source: filename).
4. If sources conflict, mention both versions and cite each.
5. If the context lacks the answer, respond with EXACTLY:
   "I don\'t have enough information in the knowledge base to answer that. You may want to upload relevant documents or check with the appropriate team."
6. Preserve exact technical terms, dates, version numbers, proper nouns, and numerical values EXACTLY as they appear in the context.
7. NEVER convert currencies (no INR-to-USD, no lakh-to-million, no crore-to-USD). NEVER add "approximately X" parentheticals with conversions or estimates. If the context says "2 crore INR", you say "2 crore INR" and nothing more.
8. NEVER do arithmetic on numbers from the context. NEVER restate values in different units.
9. Be concise. Use bullet points for lists. Use prose for explanations. No filler.

# Citation Format Example
Context: "[Source 1 | type=text | file=Project_Atlas_Tech_Spec.txt | relevance=high] Lead: Sandeep Mishra. Team size: 14."
Question: "Who leads Project Atlas and how big is the team?"
Answer: "Project Atlas is led by Sandeep Mishra with a team of 14 engineers (Source: Project_Atlas_Tech_Spec.txt)."

# Multi-Source Synthesis Example
Context contains chunks from Project_Atlas_Tech_Spec.txt AND VVDN_Q1_2026_Strategy.txt about Atlas.
Answer should combine: "Project Atlas is a 5G Open RAN platform (Source: Project_Atlas_Tech_Spec.txt) approved on March 5, 2026 with a budget of 2 crore INR (Source: VVDN_Q1_2026_Strategy.txt)."

# Quality Checks (apply silently before responding)
- Did I cite a source for every factual claim?
- Did I combine information across sources where they overlap?
- Did I avoid stating anything not in the context?
"""


QUERY_EXPANSION_PROMPT = """You are a query rewriter for an internal knowledge base search system.

Given a user question, produce 3 search query variations that would help retrieve relevant documents.
The variations should:
- Use different keywords and synonyms
- Capture different aspects of the question
- Stay focused on the same intent

Output ONLY the 3 queries, one per line. No numbering, no explanation, no markdown.

User question: {query}

3 search queries:"""


def expand_query(query: str) -> List[str]:
    """Use the LLM to rewrite the query into multiple search variants."""
    try:
        llm = get_llm()
        prompt = QUERY_EXPANSION_PROMPT.format(query=query)
        response = llm.generate("", prompt, "")
        # Parse - take non-empty lines, strip leading numbering/bullets
        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        cleaned = []
        for l in lines:
            for prefix in ("1.", "2.", "3.", "-", "*", "1)", "2)", "3)"):
                if l.startswith(prefix):
                    l = l[len(prefix):].strip()
            if l:
                cleaned.append(l)
        # Always include the original query as the first variant
        variants = [query] + cleaned[:3]
        # Dedupe while preserving order
        seen, out = set(), []
        for v in variants:
            if v.lower() not in seen:
                out.append(v)
                seen.add(v.lower())
        return out[:4]
    except Exception:
        # If expansion fails, fall back to just the original query
        return [query]


def multi_query_retrieve(queries: List[str], k_per_query: int = 4) -> List[Dict]:
    """Search KB with each query variant, dedupe by chunk text, keep best distance per chunk."""
    seen = {}  # chunk_text -> best chunk dict
    for q in queries:
        chunks = kb.search(q, k=k_per_query)
        for c in chunks:
            text = c["text"]
            existing = seen.get(text)
            if existing is None or c.get("distance", 1.0) < existing.get("distance", 1.0):
                seen[text] = c
    # Sort by ascending distance (most relevant first)
    merged = sorted(seen.values(), key=lambda c: c.get("distance", 1.0))
    return merged


def filter_relevant_chunks(chunks: List[Dict], threshold: float = 1.5, max_chunks: int = 8) -> List[Dict]:
    """Drop low-relevance chunks; cap context size."""
    filtered = [c for c in chunks if c.get("distance", 0) < threshold]
    return filtered[:max_chunks]


def build_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks into a structured context block with metadata."""
    if not chunks:
        return "(No relevant content found in the knowledge base for this query.)"
    parts = []
    for i, c in enumerate(chunks, 1):
        src = c["metadata"].get("source", "unknown")
        ctype = c["metadata"].get("type", "doc")
        dist = c.get("distance", 1.0)
        if dist < 0.8:
            relevance = "high"
        elif dist < 1.2:
            relevance = "medium"
        else:
            relevance = "low"
        header = f"[Source {i} | type={ctype} | file={src} | relevance={relevance}]"
        parts.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def answer_question(query: str, k: int = 5) -> dict:
    """End-to-end RAG with query expansion and multi-query retrieval."""
    # Step 1: expand the query into multiple search variants
    query_variants = expand_query(query)

    # Step 2: retrieve with all variants, dedupe and merge
    raw_chunks = multi_query_retrieve(query_variants, k_per_query=max(k, 4))

    # Step 3: filter low-relevance noise
    chunks = filter_relevant_chunks(raw_chunks, threshold=1.5, max_chunks=max(k, 6))

    # Step 4: build structured context
    context = build_context(chunks)

    # Step 5: synthesize answer
    llm = get_llm()
    answer = llm.generate(SYSTEM_PROMPT, query, context)

    # Step 6: return answer + sources + the expansion trace (useful for demo)
    return {
        "answer": answer,
        "sources": [
            {
                "source": c["metadata"].get("source"),
                "type": c["metadata"].get("type"),
            }
            for c in chunks
        ],
        "chunks_used": len(chunks),
        "query_variants": query_variants,
    }
