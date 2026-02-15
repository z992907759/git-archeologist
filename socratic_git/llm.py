"""Local mlx-lm generation."""

try:
    from mlx_lm import generate, load
except Exception:  # pragma: no cover - dependency/runtime environment issues
    generate = None
    load = None


def generate_answer(
    question: str,
    contexts: list[dict],
    model_name: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
) -> str:
    """Generate answer from retrieved commit contexts."""
    if load is None or generate is None:
        raise RuntimeError("Missing dependency: mlx-lm")

    system_instruction = (
        "你是资深开发者，正在分析 git 历史。你只能根据提供的 commit 上下文回答问题。\n"
        "如果上下文没有提到相关原因/信息，就回答：我不知道（根据现有提交上下文无法判断）。\n"
        "先引用证据（commit hash / message / diff）再给结论；证据不足就说不知道。禁止编造未出现的原因。\n"
        "回答必须以一行 'Evidence Hashes: ...' 开头；若无法给出证据则写 'Evidence Hashes: none'。\n"
        "若 introducing commit 的 message+diff 没有明确动机/引入描述，结论必须是："
        "I don't know (insufficient evidence in commit message/diff).\n"
        "禁止使用“可能/大概/推测”等词作为结论。"
    )

    context_blocks = []
    for i, ctx in enumerate(contexts[:8], start=1):
        commit = ctx.get("commit", {})
        text = ctx.get("text", "")
        text_lines = text.splitlines()[:60]
        text_preview = "\n".join(text_lines)
        block = (
            f"[COMMIT {i}]\n"
            f"id: {commit.get('hash', '')}\n"
            f"date: {commit.get('date', '')}\n"
            f"author: {commit.get('author', '')}\n"
            f"message: {commit.get('message', '')}\n"
            f"text:\n{text_preview}"
        )
        context_blocks.append(block)

    context_text = "\n\n".join(context_blocks) if context_blocks else "(no context)"
    prompt = (
        f"[SYSTEM]\n{system_instruction}\n\n"
        f"[CONTEXT]\n{context_text}\n\n"
        f"[QUESTION]\n{question}\n\n"
        "[ANSWER]"
    )

    model, tokenizer = load(model_name)
    answer = generate(model, tokenizer, prompt=prompt, max_tokens=300, verbose=False)
    return answer.strip()
