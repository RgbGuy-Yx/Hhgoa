import argparse
import json
import sys
import time
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from pydantic import BaseModel, Field

import config
from stt import transcribe_audio

RETRIEVAL_DISTANCE_THRESHOLD = 0.75
RETRIEVAL_TOP_K = 3
MAX_DB_RETRIES = 2
LOG_PATH = Path("./logs/pipeline_timings.jsonl")


def _log_event(stage: str, **fields) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"stage": stage, **fields}
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")


_vectorstore = None


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
    last_exc = None
    for attempt in range(MAX_DB_RETRIES + 1):
        try:
            return get_vectorstore().similarity_search_with_score(query, k=k)
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_DB_RETRIES:
                time.sleep(0.5 * (2 ** attempt))
    raise last_exc


RETRIEVE_TOOL_DESCRIPTION = (
    "Look up factual information about Hacker House Goa 2026: dates, "
    "deadlines, rules, tasks, schedule, or FAQ. Only use this for factual "
    "questions about the event itself - never call it for greetings or "
    "general small talk."
)


@tool(description=RETRIEVE_TOOL_DESCRIPTION)
def retrieve_hhgoa_info(query: str) -> str:
    start_ts = time.time()
    start = time.perf_counter()

    try:
        results = _similarity_search_with_retry(query, RETRIEVAL_TOP_K)
    except Exception as exc:
        _log_event(
            "retrieval",
            query=query,
            start_ts=start_ts,
            duration_ms=(time.perf_counter() - start) * 1000,
            error=str(exc),
        )
        return "NO_RELEVANT_INFO_FOUND"

    kept = [(doc, score) for doc, score in results if score <= RETRIEVAL_DISTANCE_THRESHOLD]

    _log_event(
        "retrieval",
        query=query,
        start_ts=start_ts,
        duration_ms=(time.perf_counter() - start) * 1000,
        num_results=len(results),
        num_kept=len(kept),
        best_distance=min((s for _, s in results), default=None),
    )

    if not kept:
        return "NO_RELEVANT_INFO_FOUND"

    return "\n".join(
        f"[{doc.metadata.get('source_doc', 'unknown')}] {doc.metadata.get('window_text', doc.page_content)}"
        for doc, _ in kept
    )


AGENT_SYSTEM_PROMPT = (
    "You're a friendly assistant for Hacker House Goa 2026. Greet users naturally, "
    "chat casually for small talk (greetings, \"how's it going\", banter about "
    "Goa/hackathons in general) - no tool call needed for any of that.\n\n"
    "But for ANY factual claim about Hacker House Goa 2026 specifically - dates, "
    "deadlines, rules, tasks, prizes, judging criteria - you must call the "
    "retrieve_hhgoa_info tool first and ground your answer strictly in what it "
    "returns. If the tool returns NO_RELEVANT_INFO_FOUND, say so clearly and "
    "naturally (e.g. \"hmm, I don't have that info on hand - might want to check "
    "the official page\") rather than guessing.\n\n"
    "Never invent HHGoa-specific facts, even if asked casually or repeatedly, "
    "and never answer a factual HHGoa question without calling the tool first."
)

_agent_executor = None


def get_agent_executor() -> AgentExecutor:
    global _agent_executor
    if _agent_executor is None:
        llm = ChatMistralAI(model=config.MISTRAL_CHAT_MODEL, temperature=0.2)
        tools = [retrieve_hhgoa_info]
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", AGENT_SYSTEM_PROMPT),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        agent = create_tool_calling_agent(llm, tools, prompt)
        _agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            return_intermediate_steps=True,
            verbose=False,
        )
    return _agent_executor


def _extract_retrieval_info(intermediate_steps):
    used_retrieval = len(intermediate_steps) > 0
    source_docs = []
    for action, observation in intermediate_steps:
        if action.tool == "retrieve_hhgoa_info" and observation != "NO_RELEVANT_INFO_FOUND":
            for line in observation.split("\n"):
                if line.startswith("[") and "]" in line:
                    source_docs.append(line[1:line.index("]")])

    seen = set()
    unique_sources = []
    for s in source_docs:
        if s not in seen:
            seen.add(s)
            unique_sources.append(s)
    return used_retrieval, unique_sources


class AgentAnswer(BaseModel):
    answer: str = Field(description="The final natural-language answer to show the user.")
    used_retrieval: bool = Field(description="Whether the retrieve_hhgoa_info tool was called this turn.")
    source_docs: list[str] = Field(description="Unique source_doc filenames the answer is grounded in.")
    is_refusal: bool = Field(description="True if this declines to answer because no grounded info was found.")


STRUCTURING_SYSTEM_PROMPT = (
    "You package an assistant's reply into a structured record. Keep the "
    "answer text as-is (light cleanup only, don't change its meaning). The "
    "used_retrieval and source_docs values given to you are already correct "
    "(computed from the actual tool trace) - copy them through unchanged. "
    "Set is_refusal to true only if the reply is declining to answer a "
    "factual question because no grounded info was found."
)

_structuring_llm = None


def get_structuring_llm():
    global _structuring_llm
    if _structuring_llm is None:
        _structuring_llm = ChatMistralAI(
            model=config.MISTRAL_CHAT_MODEL, temperature=0
        ).with_structured_output(AgentAnswer)
    return _structuring_llm


def structure_agent_output(question: str, raw_answer: str, used_retrieval: bool, source_docs: list) -> AgentAnswer:
    user_msg = (
        f"Question: {question}\n"
        f"Assistant reply: {raw_answer}\n"
        f"used_retrieval (already correct, keep as-is): {used_retrieval}\n"
        f"source_docs (already correct, keep as-is): {source_docs}\n"
    )
    structured: AgentAnswer = get_structuring_llm().invoke(
        [
            ("system", STRUCTURING_SYSTEM_PROMPT),
            ("human", user_msg),
        ]
    )
    structured.used_retrieval = used_retrieval
    structured.source_docs = source_docs
    return structured


def run_agent(question: str) -> dict:
    start_ts = time.time()
    start = time.perf_counter()

    result = get_agent_executor().invoke({"input": question, "chat_history": []})

    duration_ms = (time.perf_counter() - start) * 1000
    intermediate_steps = result.get("intermediate_steps", [])
    used_retrieval, source_docs = _extract_retrieval_info(intermediate_steps)

    _log_event(
        "generation",
        question=question,
        start_ts=start_ts,
        duration_ms=duration_ms,
        used_retrieval=used_retrieval,
        num_tool_calls=len(intermediate_steps),
    )

    structured = structure_agent_output(question, result["output"], used_retrieval, source_docs)
    return structured.model_dump()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run the agentic RAG pipeline.")
    parser.add_argument("--text", help="A single question to ask.")
    parser.add_argument("--audio", help="Path to an audio file (runs STT -> agent).")
    args = parser.parse_args()

    if args.audio:
        transcript = transcribe_audio(args.audio)
        print(json.dumps({"transcript": transcript, **run_agent(transcript)}, indent=2, ensure_ascii=False))
    elif args.text:
        print(json.dumps(run_agent(args.text), indent=2, ensure_ascii=False))
    else:
        question = input("Question: ")
        print(json.dumps(run_agent(question), indent=2, ensure_ascii=False))
