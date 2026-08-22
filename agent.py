"""
agent.py — Agentic RAG harness for Hacker House Goa 2026
=========================================================

Pipeline
--------
  question
    → Input guardrail  (blocks junk / unsafe / prompt-injection)
    → LangChain 1.x agent  (create_agent + CompiledStateGraph)
        · Mistral tool-calling LLM
        · retrieve_hhgoa_info tool with DB retry
        · max_iterations cap via middleware
        · LLM call retried on transient errors
    → Output / grounding guardrail
    → Structured output  (Pydantic AgentAnswer via Mistral structured output)
    → JSONL timing log

LangChain version: 1.x  (AgentExecutor no longer exists — uses CompiledStateGraph)
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import chromadb
from langchain.agents import AgentState, create_agent
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from pydantic import BaseModel, Field

import config
from stt import transcribe_audio

# ─── constants ────────────────────────────────────────────────────────────────

RETRIEVAL_DISTANCE_THRESHOLD = 0.75
RETRIEVAL_TOP_K = config.TOP_K
MAX_DB_RETRIES = 2
MAX_LLM_RETRIES = 2
MAX_ANSWER_CHARS = 1500

LOG_PATH = Path("./logs/pipeline_timings.jsonl")

_UNSAFE_RE = re.compile(
    r"\b(ignore previous|forget instructions|jailbreak|prompt injection|"
    r"roleplay as|act as (an? )?(evil|unrestricted)|dan mode)\b",
    re.IGNORECASE,
)

_FACTUAL_KEYWORDS_RE = re.compile(
    r"\b(deadline|prize|winner|rule|task|schedule|date|when|where|submit|"
    r"register|team|judge|criteria|shortlist|hacker house|hhgoa)\b",
    re.IGNORECASE,
)


# ─── logging ──────────────────────────────────────────────────────────────────

def _log_event(stage: str, **fields) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"stage": stage, **fields}
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")


# ─── vector store (singleton) ─────────────────────────────────────────────────

_vectorstore: Optional[Chroma] = None


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        embeddings = MistralAIEmbeddings(model=config.MISTRAL_EMBED_MODEL)
        _vectorstore = Chroma(
            client=client,
            collection_name=config.COLLECTION_NAME,
            embedding_function=embeddings,
        )
    return _vectorstore


def _similarity_search_with_retry(query: str, k: int):
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_DB_RETRIES + 1):
        try:
            return get_vectorstore().similarity_search_with_score(query, k=k)
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_DB_RETRIES:
                time.sleep(0.4 * (2 ** attempt))
    raise last_exc  # type: ignore[misc]


# ─── retrieval tool ───────────────────────────────────────────────────────────

@tool
def retrieve_hhgoa_info(query: str) -> str:
    """Look up factual information about Hacker House Goa 2026: dates,
    deadlines, rules, tasks, schedule, prizes, or FAQ.
    Only call this for factual questions about the event —
    never for greetings, small talk, or off-topic queries."""
    start_ts = time.time()
    t0 = time.perf_counter()

    try:
        results = _similarity_search_with_retry(query, RETRIEVAL_TOP_K)
    except Exception as exc:
        _log_event(
            "retrieval",
            query=query,
            start_ts=start_ts,
            duration_ms=(time.perf_counter() - t0) * 1000,
            error=str(exc),
        )
        return "NO_RELEVANT_INFO_FOUND"

    kept = [(doc, score) for doc, score in results if score <= RETRIEVAL_DISTANCE_THRESHOLD]

    _log_event(
        "retrieval",
        query=query,
        start_ts=start_ts,
        duration_ms=(time.perf_counter() - t0) * 1000,
        num_results=len(results),
        num_kept=len(kept),
        best_distance=min((s for _, s in results), default=None),
    )

    if not kept:
        return "NO_RELEVANT_INFO_FOUND"

    return "\n".join(
        f"[{doc.metadata.get('source_doc', 'unknown')}] "
        f"{doc.metadata.get('window_text', doc.page_content)}"
        for doc, _ in kept
    )


# ─── agent (singleton) ────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a friendly, helpful assistant for Hacker House Goa 2026 (HHGoa 2026). "
    "You may chat casually about greetings, Goa, hackathons in general, and small talk "
    "without calling any tool.\n\n"
    "RULES — follow these strictly:\n"
    "1. For ANY factual claim about HHGoa 2026 specifically (dates, deadlines, prizes, "
    "   rules, tasks, schedule, judging criteria, team sizes, submission format) — "
    "   you MUST call retrieve_hhgoa_info first and ground your answer solely in "
    "   what it returns.\n"
    "2. If retrieve_hhgoa_info returns NO_RELEVANT_INFO_FOUND, say so clearly and "
    "   naturally. Do NOT guess or fabricate HHGoa-specific facts.\n"
    "3. Never invent HHGoa facts even if the user asks repeatedly or casually.\n"
    "4. Always format your responses using clean, structured Markdown syntax. Use headings (e.g. ### Header), bold text for key terms, bullet points (`- `) or numbered lists for key items, and concise sections.\n"
    "5. Keep answers concise, clear, and well-structured.\n"
    "6. If a question is completely unrelated to HHGoa or general hackathon topics, "
    "   politely say you are only here to help with HHGoa 2026."
)

_agent_graph = None


def get_agent():
    global _agent_graph
    if _agent_graph is None:
        llm = ChatMistralAI(model=config.MISTRAL_CHAT_MODEL, temperature=0.2)
        _agent_graph = create_agent(
            llm,
            [retrieve_hhgoa_info],
            system_prompt=_SYSTEM_PROMPT,
        )
    return _agent_graph


# ─── output schema ────────────────────────────────────────────────────────────

class AgentAnswer(BaseModel):
    answer: str = Field(description="Natural-language answer in structured Markdown syntax to show the user.")
    used_retrieval: bool = Field(description="True if retrieve_hhgoa_info was called.")
    source_docs: list[str] = Field(description="Unique source filenames grounding the answer.")
    is_refusal: bool = Field(description="True if declining because no grounded info found.")
    guardrail_triggered: bool = Field(description="True if an input or output guardrail fired.")
    guardrail_reason: str = Field(description="Short explanation if a guardrail fired, else empty.")


_STRUCTURING_SYSTEM = (
    "Extract a structured record from an assistant reply. "
    "Keep answer text as-is (preserve all markdown formatting, bullet points, headers, and bold text). "
    "used_retrieval, source_docs, guardrail_triggered, guardrail_reason are supplied "
    "as ground-truth — copy them unchanged. "
    "Set is_refusal=true only if the reply declines a factual HHGoa question "
    "because no grounded info was found."
)

_structuring_llm = None


def _get_structuring_llm():
    global _structuring_llm
    if _structuring_llm is None:
        _structuring_llm = ChatMistralAI(
            model=config.MISTRAL_CHAT_MODEL, temperature=0
        ).with_structured_output(AgentAnswer)
    return _structuring_llm


def _structure_output(
    question: str,
    raw_answer: str,
    used_retrieval: bool,
    source_docs: list[str],
    guardrail_triggered: bool = False,
    guardrail_reason: str = "",
) -> AgentAnswer:
    user_msg = (
        f"Question: {question}\n"
        f"Assistant reply: {raw_answer}\n"
        f"used_retrieval (ground-truth, copy as-is): {used_retrieval}\n"
        f"source_docs (ground-truth, copy as-is): {source_docs}\n"
        f"guardrail_triggered (copy as-is): {guardrail_triggered}\n"
        f"guardrail_reason (copy as-is): {guardrail_reason}\n"
    )
    structured: AgentAnswer = _get_structuring_llm().invoke([
        ("system", _STRUCTURING_SYSTEM),
        ("human", user_msg),
    ])
    # Always pin computed ground-truth fields
    structured.used_retrieval = used_retrieval
    structured.source_docs = source_docs
    structured.guardrail_triggered = guardrail_triggered
    structured.guardrail_reason = guardrail_reason
    return structured


# ─── guardrails ───────────────────────────────────────────────────────────────

def _check_input(question: str) -> Optional[str]:
    q = question.strip()
    if len(q) < 3:
        return "Input too short to process."
    if len(q) > 800:
        return "Input too long. Please ask a shorter question."
    if _UNSAFE_RE.search(q):
        return "I can't help with that kind of request."
    return None


def _check_grounding(
    raw_answer: str,
    used_retrieval: bool,
    source_docs: list[str],
    question: str,
) -> Optional[str]:
    answer_lower = raw_answer.lower()
    refusal_phrases = ["don't have that", "no information", "couldn't find",
                       "not sure", "check the official", "no relevant"]
    if any(p in answer_lower for p in refusal_phrases):
        return None
    if _FACTUAL_KEYWORDS_RE.search(question) and not used_retrieval:
        return "Factual HHGoa question answered without retrieval — answer may be ungrounded."
    if used_retrieval and not source_docs:
        return "Retrieval returned no grounded sources for this answer."
    return None


# ─── parse LangChain 1.x AgentState messages ──────────────────────────────────

def _parse_agent_state(state: AgentState) -> tuple[str, bool, list[str]]:
    """
    Extract (final_answer, used_retrieval, source_docs) from the
    AgentState messages list produced by LangChain 1.x create_agent.
    """
    messages = state.get("messages", [])

    # Final answer is the last AIMessage that is NOT a tool call
    final_answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            # In LangChain 1.x an AIMessage with tool_calls has content="" —
            # skip those, we want the final text response
            if msg.content and not getattr(msg, "tool_calls", None):
                final_answer = str(msg.content)
                break
    if not final_answer:
        # Fallback: take the last AIMessage content regardless
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                final_answer = str(msg.content)
                break

    # Detect tool usage and extract source docs from ToolMessage observations
    used_retrieval = False
    source_docs: list[str] = []
    seen: set[str] = set()

    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "retrieve_hhgoa_info":
            used_retrieval = True
            content = msg.content or ""
            if content != "NO_RELEVANT_INFO_FOUND":
                for line in content.split("\n"):
                    if line.startswith("[") and "]" in line:
                        doc = line[1:line.index("]")]
                        if doc not in seen:
                            seen.add(doc)
                            source_docs.append(doc)

    return final_answer, used_retrieval, source_docs


# ─── LLM invoke with retry ────────────────────────────────────────────────────

def _invoke_with_retry(question: str) -> AgentState:
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_LLM_RETRIES + 1):
        try:
            return get_agent().invoke(
                {"messages": [HumanMessage(content=question)]}
            )
        except Exception as exc:
            last_exc = exc
            err_str = str(exc).lower()
            if any(k in err_str for k in ("rate", "timeout", "502", "503", "500", "connection")):
                if attempt < MAX_LLM_RETRIES:
                    wait = 1.0 * (2 ** attempt)
                    _log_event("llm_retry", attempt=attempt + 1, error=str(exc), wait_s=wait)
                    time.sleep(wait)
                    continue
            raise
    raise last_exc  # type: ignore[misc]


def _truncate(text: str, max_chars: int = MAX_ANSWER_CHARS) -> str:
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "… [truncated]"


# ─── public entry point ───────────────────────────────────────────────────────

def run_agent(question: str) -> dict:
    """
    Full harness entry point. Always returns a dict matching AgentAnswer.
    Never raises — all errors become structured refusal responses.
    """
    start_ts = time.time()
    t0 = time.perf_counter()

    # ── 1. Input guardrail ────────────────────────────────────────────────────
    block_reason = _check_input(question)
    if block_reason:
        _log_event(
            "guardrail_input",
            question=question,
            reason=block_reason,
            start_ts=start_ts,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )
        return AgentAnswer(
            answer=block_reason,
            used_retrieval=False,
            source_docs=[],
            is_refusal=True,
            guardrail_triggered=True,
            guardrail_reason=block_reason,
        ).model_dump()

    # ── 2. Agent execution ────────────────────────────────────────────────────
    try:
        state = _invoke_with_retry(question)
    except Exception as exc:
        duration_ms = (time.perf_counter() - t0) * 1000
        _log_event("agent_error", question=question, error=str(exc),
                   start_ts=start_ts, duration_ms=duration_ms)
        return AgentAnswer(
            answer="Sorry, I ran into a problem. Please try again.",
            used_retrieval=False,
            source_docs=[],
            is_refusal=True,
            guardrail_triggered=False,
            guardrail_reason=f"Agent error: {exc}",
        ).model_dump()

    raw_answer, used_retrieval, source_docs = _parse_agent_state(state)
    raw_answer = _truncate(raw_answer)

    duration_ms = (time.perf_counter() - t0) * 1000
    _log_event(
        "generation",
        question=question,
        start_ts=start_ts,
        duration_ms=duration_ms,
        used_retrieval=used_retrieval,
        num_source_docs=len(source_docs),
    )

    # ── 3. Output / grounding guardrail ───────────────────────────────────────
    grounding_issue = _check_grounding(raw_answer, used_retrieval, source_docs, question)
    guardrail_triggered = grounding_issue is not None
    guardrail_reason = grounding_issue or ""

    if guardrail_triggered:
        _log_event("guardrail_output", question=question, reason=grounding_issue,
                   start_ts=start_ts, duration_ms=duration_ms)

    # ── 4. Structure output ───────────────────────────────────────────────────
    try:
        structured = _structure_output(
            question=question,
            raw_answer=raw_answer,
            used_retrieval=used_retrieval,
            source_docs=source_docs,
            guardrail_triggered=guardrail_triggered,
            guardrail_reason=guardrail_reason,
        )
    except Exception as exc:
        _log_event("structuring_error", error=str(exc), start_ts=start_ts)
        return AgentAnswer(
            answer=raw_answer,
            used_retrieval=used_retrieval,
            source_docs=source_docs,
            is_refusal=False,
            guardrail_triggered=guardrail_triggered,
            guardrail_reason=guardrail_reason,
        ).model_dump()

    return structured.model_dump()


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run the HHGoa agentic RAG harness.")
    parser.add_argument("--text", help="A single question to ask.")
    parser.add_argument("--audio", help="Path to an audio file (runs STT → agent).")
    args = parser.parse_args()

    if args.audio:
        transcript = transcribe_audio(args.audio)
        print(json.dumps({"transcript": transcript, **run_agent(transcript)},
                         indent=2, ensure_ascii=False))
    elif args.text:
        print(json.dumps(run_agent(args.text), indent=2, ensure_ascii=False))
    else:
        q = input("Question: ")
        print(json.dumps(run_agent(q), indent=2, ensure_ascii=False))
