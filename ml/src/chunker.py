import re
from typing import NamedTuple

from ml.src.protect import detect_protected_spans
from ml.src.tokenizer import count_tokens
from ml.src.types import ChunkTrace, ContextBlock, Message


class InternalChunk(NamedTuple):
    trace: ChunkTrace
    qa_parent_indices: list[int]


def _split_text_into_pieces(text: str, max_chunk_tokens: int = 150) -> list[str]:
    text = text.strip()
    if not text:
        return []

    # 1. Paragraph split
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    pieces: list[str] = []

    for para in paragraphs:
        tokens = count_tokens(para)
        if tokens <= max_chunk_tokens:
            pieces.append(para)
        else:
            # 2. Line or sentence split
            lines = [line.strip() for line in para.split("\n") if line.strip()]
            cur_buffer: list[str] = []
            cur_count = 0

            for line in lines:
                line_tokens = count_tokens(line)
                if line_tokens > max_chunk_tokens:
                    # Split on sentence boundaries
                    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip()]
                    for s in sentences:
                        s_tokens = count_tokens(s)
                        if cur_count + s_tokens > max_chunk_tokens and cur_buffer:
                            pieces.append("\n".join(cur_buffer))
                            cur_buffer = [s]
                            cur_count = s_tokens
                        else:
                            cur_buffer.append(s)
                            cur_count += s_tokens
                elif cur_count + line_tokens > max_chunk_tokens and cur_buffer:
                    pieces.append("\n".join(cur_buffer))
                    cur_buffer = [line]
                    cur_count = line_tokens
                else:
                    cur_buffer.append(line)
                    cur_count += line_tokens

            if cur_buffer:
                pieces.append("\n".join(cur_buffer))

    return pieces or [text]


def chunk_inputs(
    system_prompt: str,
    history: list[Message],
    context_blocks: list[ContextBlock],
    max_chunk_tokens: int = 150,
) -> list[InternalChunk]:
    chunks: list[InternalChunk] = []
    global_idx = 0

    # 1. System Prompt
    if system_prompt and system_prompt.strip():
        sys_pieces = _split_text_into_pieces(system_prompt, max_chunk_tokens=256)
        for p_idx, piece in enumerate(sys_pieces):
            prot = detect_protected_spans(piece, source_type="system")
            trace = ChunkTrace(
                id=f"system:{p_idx}",
                source_type="system",
                source=None,
                original_index=global_idx,
                text=piece,
                token_count=count_tokens(piece),
                selected=False,
                protected=prot.is_protected,
                score=0.0,
                reason="; ".join(prot.reasons) if prot.reasons else "system chunk",
            )
            chunks.append(InternalChunk(trace=trace, qa_parent_indices=[]))
            global_idx += 1

    # 2. History
    # Identify the last user turn index to protect recent user turn
    last_user_msg_idx = -1
    for i, msg in enumerate(history):
        if msg.role == "user":
            last_user_msg_idx = i

    last_user_chunk_indices: list[int] = []

    for msg_idx, msg in enumerate(history):
        if not msg.content.strip():
            continue
        is_recent_user = (msg_idx == last_user_msg_idx)
        msg_pieces = _split_text_into_pieces(msg.content, max_chunk_tokens=max_chunk_tokens)

        current_msg_chunk_indices: list[int] = []

        for p_idx, piece in enumerate(msg_pieces):
            prot = detect_protected_spans(
                piece,
                source_type="history",
                is_recent_user_turn=is_recent_user,
            )
            prefix = f"{msg.role}: "
            chunk_text = f"{prefix}{piece}"
            trace = ChunkTrace(
                id=f"history:{msg_idx}:{p_idx}",
                source_type="history",
                source=f"{msg.role}:{msg_idx}",
                original_index=global_idx,
                text=chunk_text,
                token_count=count_tokens(chunk_text),
                selected=False,
                protected=prot.is_protected,
                score=0.0,
                reason="; ".join(prot.reasons) if prot.reasons else f"history {msg.role} turn",
            )
            
            qa_parents = list(last_user_chunk_indices) if msg.role == "assistant" else []
            chunks.append(InternalChunk(trace=trace, qa_parent_indices=qa_parents))
            current_msg_chunk_indices.append(global_idx)
            global_idx += 1

        if msg.role == "user":
            last_user_chunk_indices = current_msg_chunk_indices

    # 3. Context Blocks
    for block in context_blocks:
        if not block.content.strip():
            continue
        block_pieces = _split_text_into_pieces(block.content, max_chunk_tokens=max_chunk_tokens)
        for p_idx, piece in enumerate(block_pieces):
            prot = detect_protected_spans(piece, source_type="context")
            trace = ChunkTrace(
                id=f"context:{block.id}:{p_idx}",
                source_type="context",
                source=block.source or block.id,
                original_index=global_idx,
                text=piece,
                token_count=count_tokens(piece),
                selected=False,
                protected=prot.is_protected,
                score=0.0,
                reason="; ".join(prot.reasons) if prot.reasons else "context block",
            )
            chunks.append(InternalChunk(trace=trace, qa_parent_indices=[]))
            global_idx += 1

    return chunks
