import re

SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')
HEADER_RE = re.compile(r'^(#{1,6})\s+(.*)')


def split_into_blocks(text: str):
    lines = text.split("\n")
    blocks = []
    current_section = "Introduction"
    buffer = []

    def flush():
        if buffer:
            blocks.append((current_section, " ".join(buffer).strip()))
            buffer.clear()

    for line in lines:
        stripped = line.strip()
        header_match = HEADER_RE.match(stripped)
        if header_match:
            flush()
            current_section = header_match.group(2).strip()
            continue
        if not stripped:
            flush()
            continue
        buffer.append(stripped)
    flush()

    return [(section, para) for section, para in blocks if para]


def split_sentences(paragraph: str):
    sentences = SENTENCE_SPLIT_RE.split(paragraph.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_document(text: str, source_doc: str, window_size: int = 2):
    blocks = split_into_blocks(text)
    chunks = []
    idx = 0

    for section, paragraph in blocks:
        sentences = split_sentences(paragraph)
        for i, sentence in enumerate(sentences):
            start = max(0, i - window_size)
            end = min(len(sentences), i + window_size + 1)
            window_text = " ".join(sentences[start:end])
            chunks.append(
                {
                    "id": f"{source_doc}::{idx}",
                    "sentence": sentence,
                    "window_text": window_text,
                    "source_doc": source_doc,
                    "section": section,
                    "sentence_index": idx,
                }
            )
            idx += 1

    return chunks
