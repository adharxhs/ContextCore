const $ = (id) => document.getElementById(id);
const apiUrl = window.CONTEXTCORE_API || `${window.location.origin}/api`;
const escapeHtml = (value) => { const node = document.createElement("div"); node.textContent = value; return node.innerHTML; };
function getBlocks() { return $("blocks").value.split(/\n\s*\n/).map((content, index) => ({ id: `support-${index + 1}`, content: content.trim(), source: "Support knowledge base" })).filter((block) => block.content); }
function render(data, original, executionMode) {
  const reduction = data.input_tokens ? Math.round(data.saved_tokens / data.input_tokens * 100) : 0;
  const labels = ["Input", "Saved", "Reduction", "Latency"], values = [[data.input_tokens,"tokens"],[data.saved_tokens,"tokens"],[`${reduction}%`,`less context`],[data.compression_ms,"ms"]];
  $("metrics").innerHTML = values.map(([n,u], i) => `<article><small>${labels[i]}</small><strong>${n}</strong><em>${u}</em></article>`).join("");
  $("original").textContent = original; $("compressed").textContent = data.compressed_text;
  $("original-count").textContent = `${data.input_tokens} tokens`; $("retained-count").textContent = `${data.output_tokens} tokens`;
  const trace = [...data.selected_chunks, ...data.dropped_chunks].sort((a,b) => a.source_type.localeCompare(b.source_type) || a.original_index - b.original_index);
  $("trace-body").innerHTML = trace.map((item) => `<article class="trace-row" tabindex="0"><div class="trace-state ${item.protected ? "protected" : item.selected ? "selected" : "dropped"}">${item.protected ? "Protected" : item.selected ? "Kept" : "Dropped"}</div><div><strong>${escapeHtml(item.id)}</strong><p>${escapeHtml(item.reason)}</p><p class="trace-text">${escapeHtml(item.text)}</p></div><div class="score">${Math.round(item.score * 100)}<small>score</small></div><div class="tokens">${item.token_count}<small>tokens</small></div></article>`).join("");
  let execution = $("execution");
  if (!execution) {
    execution = document.createElement("p");
    execution.id = "execution";
    $("metrics").before(execution);
  }
  execution.textContent = executionMode === "engine" ? "Real engine result" : executionMode === "fallback" ? "Offline fallback — not a benchmark result" : "Execution mode unavailable — not a benchmark result";
  execution.className = `execution ${executionMode}`;
}
$("compress").addEventListener("click", async () => {
  const button = $("compress"), status = $("status"), blocks = getBlocks();
  const payload = {system_prompt:$("system").value,history:[],context_blocks:blocks,query:$("query").value,token_budget:Number($("budget").value),scorer:$("scorer").value};
  button.disabled = true; status.textContent = "Selecting the evidence…";
  try { const response = await fetch(`${apiUrl}/v1/compress`, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)}); const data = await response.json(); if (!response.ok) throw new Error(data.detail || "Could not compress context."); const executionMode = response.headers.get("X-ContextCore-Execution") || "unknown"; render(data, `${payload.system_prompt}\n\n${blocks.map((block) => block.content).join("\n\n")}`, executionMode); status.textContent = executionMode === "engine" ? "Done. Real engine output with a complete provenance trace." : executionMode === "fallback" ? "Done using offline fallback. Do not treat this as a benchmark result." : "Done, but the server did not report its execution mode."; }
  catch (error) { status.textContent = `Connection issue: ${error.message}`; }
  finally { button.disabled = false; }
});
