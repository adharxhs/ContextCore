const $ = (id) => document.getElementById(id);
const apiUrl = window.CONTEXTCORE_API || `${window.location.origin}/api`;
const escapeHtml = (value) => { const node = document.createElement("div"); node.textContent = value; return node.innerHTML; };

let orderedTrace = [];
let activeFilter = "all";
let lockedChunk = "";

const DEMO_CASE = {
  query: "What is the approved refund deadline for order 88219?",
  system:
    "You are a support assistant for Acme Commerce. Answer only from the provided context. " +
    "Never invent account balances, dates, order statuses, or policy terms, and always cite the governing policy.",
  budget: "250",
  scorer: "bm25",
  blocks: [
    "Refund Policy: The approved refund deadline for eligible orders is 5 business days from the approval date, excluding weekends and public holidays. Refunds are issued to the original payment method.",
    "Order 88219: approval for a full refund of $248.00 recorded on 2026-10-01. Transaction reference RF-88219-4471. Status: refund initiated.",
    "Refund tracking: Customers receive an email confirmation and a transaction reference once the refund is initiated.",
    "Returns: Clothing may be returned within 30 days of delivery in original condition. Final-sale items are not returnable.",
    "Chargeback Policy: If a customer disputes a charge, the bank may freeze the amount for up to 45 calendar days during the investigation.",
    "Shipping: Standard takes 3 to 5 business days. Express takes 1 to 2 business days. Free shipping applies on orders over $75.",
    "Loyalty Program: Members earn 1 point per dollar spent. Points expire after 12 months of inactivity.",
    "Office News: The support lounge added plants, a coffee corner, and a reading shelf this quarter. The team celebrated an internal award last June.",
    "Brand Notes: The company palette uses deep green, warm cream, and a leaf motif across all campaign materials.",
    "Site Search: The storefront index reindexes nightly at 03:00 UTC. Search ranking favors exact SKU matches.",
    "Team Chat: The support team uses an internal chat for escalations. Agents must close the loop on every ticket within 2 business days.",
  ].join("\n\n"),
};

function getBlocks() {
  return $("blocks").value
    .split(/\n\s*\n/)
    .map((content, index) => ({
      id: `support-${index + 1}`,
      content: content.trim(),
      source: "Support knowledge base",
    }))
    .filter((block) => block.content);
}

function chunkState(item) {
  if (item.protected) return "protected";
  return item.selected ? "selected" : "dropped";
}

function stateLabel(item) {
  if (item.protected) return "Protected";
  return item.selected ? "Kept" : "Dropped";
}

function stateReason(item) {
  if (item.protected) return `protected — ${item.reason}`;
  if (item.selected) return `retained — ${item.reason}`;
  return `dropped — ${item.reason}`;
}

function visibleTrace() {
  if (activeFilter === "all") return orderedTrace;
  return orderedTrace.filter((item) => chunkState(item) === activeFilter);
}

function highlightChunk(chunkId, locked) {
  if (locked) lockedChunk = lockedChunk === chunkId ? "" : chunkId;
  const target = lockedChunk || chunkId;
  document.querySelectorAll("[data-chunk]").forEach((node) => {
    node.classList.toggle("highlighted", Boolean(target) && node.dataset.chunk === target);
  });
}

function renderChunks(targetId, chunks) {
  const target = $(targetId);
  target.innerHTML = "";
  chunks.forEach((item) => {
    const article = document.createElement("article");
    article.className = `chunk ${chunkState(item)}`;
    article.dataset.chunk = item.id;
    article.tabIndex = 0;
    article.setAttribute(
      "aria-label",
      `Chunk ${item.original_index + 1}: ${stateLabel(item)}. ${item.reason}`
    );
    const score = Math.max(0, Math.min(100, Math.round(item.score * 100)));
    article.innerHTML =
      `<header><b>#${item.original_index + 1}</b><code>${escapeHtml(item.id)}</code>` +
      `<span class="chunk-state ${chunkState(item)}">${stateLabel(item)}</span></header>` +
      `<p>${escapeHtml(item.text)}</p>` +
      `<footer><span>${item.token_count} tokens</span>` +
      `<span>score ${score}%</span></footer>`;
    article.addEventListener("mouseenter", () => highlightChunk(item.id, false));
    article.addEventListener("mouseleave", () => highlightChunk("", false));
    article.addEventListener("click", () => highlightChunk(item.id, true));
    article.addEventListener("focus", () => highlightChunk(item.id, false));
    article.addEventListener("blur", () => highlightChunk("", false));
    target.appendChild(article);
  });
}

function renderTrace() {
  const counts = {
    all: orderedTrace.length,
    protected: orderedTrace.filter((c) => chunkState(c) === "protected").length,
    selected: orderedTrace.filter((c) => chunkState(c) === "selected").length,
    dropped: orderedTrace.filter((c) => chunkState(c) === "dropped").length,
  };
  $("trace-count").textContent =
    `${counts.all} chunks: ${counts.protected} protected, ${counts.selected} kept, ${counts.dropped} dropped`;

  document.querySelectorAll(".filter").forEach((button) => {
    const key = button.dataset.filter;
    button.classList.toggle("active", key === activeFilter);
    const label =
      key === "all" ? "All" : key === "selected" ? "Kept" : key[0].toUpperCase() + key.slice(1);
    button.textContent = `${label} ${counts[key]}`;
  });

  const body = $("trace-body");
  body.innerHTML = "";
  visibleTrace().forEach((item) => {
    const score = Math.max(0, Math.min(100, Math.round(item.score * 100)));
    const state = chunkState(item);
    const row = document.createElement("article");
    row.className = `trace-row ${state}`;
    row.dataset.chunk = item.id;
    row.tabIndex = 0;
    row.innerHTML =
      `<div class="trace-state ${state}">${stateLabel(item)}</div>` +
      `<div><strong>${escapeHtml(item.id)}</strong>` +
      `<p>${escapeHtml(stateReason(item))}</p>` +
      `<p class="trace-text">${escapeHtml(item.text)}</p></div>` +
      `<div class="score"><b>${score}</b>` +
      `<span class="score-bar" aria-label="Relative score ${score}%"><i style="width:${score}%"></i></span>` +
      `<small>relative score</small></div>` +
      `<div class="tokens">${item.token_count}<small>tokens</small></div>`;
    row.addEventListener("mouseenter", () => highlightChunk(item.id, false));
    row.addEventListener("mouseleave", () => highlightChunk("", false));
    row.addEventListener("click", () => highlightChunk(item.id, true));
    row.addEventListener("focus", () => highlightChunk(item.id, false));
    row.addEventListener("blur", () => highlightChunk("", false));
    body.appendChild(row);
  });
}

function executionText(data, executionMode) {
  if (data.budget_exceeded)
    return "BUDGET EXCEEDED — PROTECTED CONTENT RETAINED";
  if (executionMode === "engine" && data.tokenizer === "heuristic")
    return "ENGINE TOKENIZER FALLBACK — HEURISTIC COUNTS";
  if (executionMode === "engine" && data.tokenizer === "cl100k_base")
    return "Real engine result";
  if (executionMode === "engine")
    return "REAL ENGINE — TOKENIZER UNREPORTED";
  if (executionMode === "fallback")
    return "OFFLINE FALLBACK — NOT A BENCHMARK RESULT";
  return "Execution mode unavailable — not a benchmark result";
}

function render(data, executionMode, scorer) {
  const verifiedTokenizer = data.tokenizer === "cl100k_base";
  const comparableTokens = executionMode === "engine" && verifiedTokenizer;
  const reduction = data.input_tokens
    ? Math.round((data.saved_tokens / data.input_tokens) * 100)
    : 0;
  const protectedCount = data.selected_chunks.filter((item) => item.protected).length;

  const labels = ["Input", "Retained", "Saved", "Latency"];
  const values = !comparableTokens
    ? [["-", "heuristic estimate"], ["-", "not comparable"], ["-", "not reported"], [data.compression_ms, "ms"]]
    : [[data.input_tokens, "tokens"], [data.output_tokens, "tokens"], [data.saved_tokens, "tokens"], [data.compression_ms, "ms"]];
  $("metrics").innerHTML = values
    .map(([n, u], i) => `<article><small>${labels[i]}</small><strong>${n}</strong><em>${u}</em></article>`)
    .join("");

  $("hero-savings").textContent = !comparableTokens
    ? "NON-BENCHMARK OUTPUT"
    : `${reduction}% CONTEXT REMOVED`;
  $("verdict-value").textContent = !comparableTokens ? "--" : `${reduction}%`;
  $("verdict-subtitle").textContent = !comparableTokens
    ? "Heuristic counts are not comparable."
    : `${data.input_tokens} to ${data.output_tokens} tokens with ${scorer} in ${data.compression_ms} ms`;
  $("verdict-protected").textContent = !comparableTokens ? "--" : protectedCount;
  $("verdict-output").style.width = !comparableTokens
    ? "0%"
    : `${Math.max(0, Math.min(100, 100 - reduction))}%`;

  orderedTrace = [...data.selected_chunks, ...data.dropped_chunks].sort(
    (a, b) => a.original_index - b.original_index
  );
  lockedChunk = "";
  activeFilter = "all";
  renderChunks("original", orderedTrace);
  renderChunks("compressed", orderedTrace.filter((item) => item.selected));
  renderTrace();

  $("original-count").textContent = !comparableTokens
    ? "heuristic count not shown"
    : `${orderedTrace.length} numbered chunks / ${data.input_tokens} tokens`;
  $("retained-count").textContent = !comparableTokens
    ? "heuristic count not shown"
    : `${data.selected_chunks.length} numbered chunks / ${data.output_tokens} tokens`;

  const warning = $("budget-warning");
  if (data.budget_exceeded) {
    warning.hidden = false;
    warning.textContent =
      `BUDGET EXCEEDED: output ${data.output_tokens} tokens against a ${$("budget").value} token budget. ` +
      `Protected content was retained; inspect the ledger for flagged reasons.`;
  } else {
    warning.hidden = true;
    warning.textContent = "";
  }

  const execution = $("execution");
  execution.textContent = executionText(data, executionMode);
  execution.className = `execution ${executionMode}${data.budget_exceeded ? " budget-exceeded" : ""}`;

  return executionMode === "engine" && verifiedTokenizer
    ? "Done. Real engine output with verified tokenizer counts and a complete provenance trace."
    : executionMode === "engine"
      ? "Done. Real engine output, but token counts are not verified for benchmarking."
      : executionMode === "fallback"
        ? "Done using offline fallback. Heuristic token estimates are intentionally hidden."
        : "Done, but the server did not report its execution mode.";
}

$("load-demo").addEventListener("click", () => {
  $("query").value = DEMO_CASE.query;
  $("system").value = DEMO_CASE.system;
  $("budget").value = DEMO_CASE.budget;
  $("scorer").value = DEMO_CASE.scorer;
  $("blocks").value = DEMO_CASE.blocks;
  $("status").textContent = "DEMO CASE LOADED: refund deadline for order 88219 at a 250-token budget";
});

document.querySelectorAll(".filter").forEach((button) => {
  button.addEventListener("click", () => {
    activeFilter = button.dataset.filter;
    lockedChunk = "";
    renderTrace();
  });
});

$("compress").addEventListener("click", async () => {
  const button = $("compress");
  const status = $("status");
  const execution = $("execution");
  const blocks = getBlocks();
  const payload = {
    system_prompt: $("system").value,
    history: [],
    context_blocks: blocks,
    query: $("query").value,
    token_budget: Number($("budget").value),
    scorer: $("scorer").value,
  };
  button.disabled = true;
  status.textContent = "Selecting the evidence...";
  execution.textContent = "Running compression…";
  execution.className = "execution running";
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 20000);
  try {
    const response = await fetch(`${apiUrl}/v1/compress`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    const data = await response.json();
    if (!response.ok)
      throw new Error(data.error?.message || "Could not compress context.");
    const headerMode = response.headers.get("X-ContextCore-Execution");
    const executionMode = headerMode === data.execution_mode ? headerMode : "unknown";
    status.textContent = render(data, executionMode, payload.scorer);
  } catch (error) {
    const message = error.name === "AbortError"
      ? "Compression timed out after 20 seconds. Try BM25 or check that the backend is running."
      : `Connection issue: ${error.message}`;
    status.textContent = message;
    execution.textContent = "Compression request failed";
    execution.className = "execution error";
  } finally {
    window.clearTimeout(timeout);
    button.disabled = false;
  }
});
