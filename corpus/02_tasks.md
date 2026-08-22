# Build Tasks and Tracks

Hacker House Goa 2026 defines specialized tracks with dedicated bounties and shortlisting challenges. Every participating team commits to a track on Day 1.

## Task 2: Voice-Enabled RAG Model (#RAGInGoa)

Build an end-to-end voice-to-answer retrieval-augmented generation (RAG) system:
- **Speech Input**: Real voice-to-text input supporting multilingual/Indic languages (e.g. via Sarvam AI saaras:v3), not typed text.
- **Engineered Retrieval**: Advanced chunking strategies (such as sentence-window expansion and markdown structure parsing) rather than naive character splitting.
- **Low Latency**: Blazing-fast end-to-end pipeline execution with latency targets under 200ms for retrieval.
- **Latency Benchmarking**: Rigorously benchmarked across real queries capturing P50, P70, and P100 latency distributions.
- **Production Harness**: Built inside a robust harness with automatic retries, structured Pydantic I/O, and graceful error recovery.
- **Guardrails**: Input and output guardrails that know when *not* to answer ungrounded or out-of-domain questions.
- Tag and showcase submissions using `#RAGInGoa` to be featured on the ecosystem Radar. Shortlisting round closes on August 22 at 11:59 PM IST.

## Autonomous Multi-Agent Systems Track

Build collaborative autonomous agent swarms capable of complex multi-step reasoning, dynamic tool usage, and state graph orchestration (LangChain 1.x StateGraph / LangGraph) under strict latency constraints.

## Decentralized Compute and Edge Systems

Build resilient local-first or decentralized applications operating over peer-to-peer mesh networks with zero central cloud reliance, utilizing conflict-free replicated data types (CRDTs) and local vector embeddings.

## AI + Web3 Consumer Products

Design and launch consumer-facing decentralized AI agents with on-chain verification, cryptographic micropayments, and real-time execution.
