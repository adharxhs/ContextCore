const $ = (id) => document.getElementById(id);
const apiUrl = window.CONTEXTCORE_API || `${window.location.origin}/api`;
const escapeHtml = (value) => { const node = document.createElement("div"); node.textContent = value; return node.innerHTML; };

function getBlocks() {
  return $("blocks").value.split(/\n\s*\n/).map((content, index) => ({
    id: `support-${index + 1}`, content: content.trim(), source: "Support knowledge base",
  })).filter((block) => block.content);
}

function render(data, original, executionMode) {
  const fallback = executionMode === "fallback";
  const reduction = data.input_tokens ? Math.round(data.saved_tokens / data.input_tokens * 100) : 0;
  const protectedCount = data.selected_chunks.filter((item) => item.protected).length;
  const labels = ["Input", "Retained", "Saved", "Latency"];
  const values = fallback
    ? [["-", "heuristic estimate"], ["-", "not comparable"], ["-", "not reported"], [data.compression_ms, "ms"]]
    : [[data.input_tokens, "tokens"], [data.output_tokens, "tokens"], [data.saved_tokens, "tokens"], [data.compression_ms, "ms"]];
  $("metrics").innerHTML = values.map(([n, u], i) => `<article><small>${labels[i]}</small><strong>${n}</strong><em>${u}</em></article>`).join("");
  $("hero-savings").textContent = fallback ? "FALLBACK OUTPUT" : `${reduction}% CONTEXT REMOVED`;
  $("verdict-value").textContent = fallback ? "--" : `${reduction}%`;
  $("verdict-subtitle").textContent = fallback ? "Heuristic counts are not comparable." : `${data.input_tokens} to ${data.output_tokens} tokens`;
  $("verdict-protected").textContent = fallback ? "--" : protectedCount;
  $("verdict-output").style.width = fallback ? "0%" : `${Math.max(0, Math.min(100, 100 - reduction))}%`;
  $("original").textContent = original;
  $("compressed").textContent = data.compressed_text;
  $("original-count").textContent = fallback ? "heuristic count not shown" : `${data.input_tokens} tokens`;
  $("retained-count").textContent = fallback ? "heuristic count not shown" : `${data.output_tokens} tokens`;

  const trace = [...data.selected_chunks, ...data.dropped_chunks]
    .sort((a, b) => a.original_index - b.original_index);
  $("trace-body").innerHTML = trace.map((item) => {
    const score = Math.max(0, Math.min(100, Math.round(item.score * 100)));
    const state = item.protected ? "protected" : item.selected ? "selected" : "dropped";
    return `<article class="trace-row ${state}" tabindex="0"><div class="trace-state ${state}">${item.protected ? "Protected" : item.selected ? "Kept" : "Dropped"}</div><div><strong>${escapeHtml(item.id)}</strong><p>${escapeHtml(item.reason)}</p><p class="trace-text">${escapeHtml(item.text)}</p></div><div class="score"><b>${score}</b><span class="score-bar" aria-label="Relative score ${score}%"><i style="width:${score}%"></i></span><small>relative score</small></div><div class="tokens">${item.token_count}<small>tokens</small></div></article>`;
  }).join("");

  const execution = $("execution");
  execution.textContent = executionMode === "engine"
    ? "Real engine result"
    : executionMode === "fallback"
      ? "OFFLINE FALLBACK - NOT A BENCHMARK RESULT"
      : "Execution mode unavailable - not a benchmark result";
  execution.className = `execution ${executionMode}`;
}

$("compress").addEventListener("click", async () => {
  const button = $("compress");
  const status = $("status");
  const blocks = getBlocks();
  const payload = { system_prompt: $("system").value, history: [], context_blocks: blocks, query: $("query").value, token_budget: Number($("budget").value), scorer: $("scorer").value };
  button.disabled = true;
  status.textContent = "Selecting the evidence...";
  try {
    const response = await fetch(`${apiUrl}/v1/compress`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error?.message || "Could not compress context.");
    const executionMode = response.headers.get("X-ContextCore-Execution") || "unknown";
    render(data, `${payload.system_prompt}\n\n${blocks.map((block) => block.content).join("\n\n")}`, executionMode);
    status.textContent = executionMode === "engine"
      ? "Done. Real engine output with a complete provenance trace."
      : executionMode === "fallback"
        ? "Done using offline fallback. Heuristic token estimates are intentionally hidden."
        : "Done, but the server did not report its execution mode.";
  } catch (error) {
    status.textContent = `Connection issue: ${error.message}`;
  } finally {
    button.disabled = false;
  }
});
