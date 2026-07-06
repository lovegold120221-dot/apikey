'use strict';
// Eburon AI API Gateway
// OpenAI-compatible front for Ollama, with branded model allowlist,
// streaming + thinking + vision + structured outputs + tool calling + web search + embeddings.
// No external dependencies. Node >= 18.

const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const PORT = parseInt(process.env.EBURON_PORT || '8088', 10);
const HOST = process.env.EBURON_HOST || '127.0.0.1';
const OLLAMA = (process.env.OLLAMA_BASE || 'http://127.0.0.1:11434').replace(/\/$/, '');
const PUBLIC_DIR = path.join(__dirname, '..', 'public');
const MAX_TOOL_ITER = 6;
const FOUNDER = 'Jo Lernout';
const MAKER = 'Eburon AI';

const CAP_LABELS = {
  'streaming': 'Streaming',
  'thinking': 'Thinking',
  'structured-outputs': 'Structured Outputs',
  'vision': 'Vision',
  'embeddings': 'Embeddings',
  'tool-calling': 'Tool Calling',
  'web-search': 'Web Search'
};

const MODELS = {
  'eburon-pro': {
    id: 'eburon-pro', name: 'Eburon Pro', owned_by: 'eburon-ai', role: 'chat',
    context_length: 32768, parameters: '1.8B', quantization: 'Q4_K_M', family: 'Eburon Pro',
    description: 'Flagship deep-reasoning agentic coding model. Autonomous software engineering with explicit chain-of-thought: decomposes objectives, plans, generates production code, and verifies its own work.',
    capabilities: ['streaming', 'thinking', 'structured-outputs', 'embeddings'],
    accent: '#7C9CFF'
  },
  'eburon-vision': {
    id: 'eburon-vision', name: 'Eburon Vision', owned_by: 'eburon-ai', role: 'chat',
    context_length: 32768, parameters: '0.9B', quantization: 'Q8_0', family: 'Eburon Vision',
    description: 'Multimodal agentic coding model with vision. Grounds engineering in screenshots, mockups and diagrams, then implements. Supports live tool calling and web search.',
    capabilities: ['streaming', 'thinking', 'structured-outputs', 'vision', 'tool-calling', 'web-search'],
    accent: '#22D3EE'
  },
  'eburon-embed': {
    id: 'eburon-embed', name: 'Eburon Embed', owned_by: 'eburon-ai', role: 'embed',
    context_length: 8192, parameters: '0.1B', family: 'Eburon Embed', dimensions: 768,
    description: 'Text embedding model for semantic search, retrieval and clustering. 768-dimensional vectors.',
    capabilities: ['embeddings'],
    accent: '#34D399'
  }
};

const CHAT_MODELS = Object.keys(MODELS).filter(k => MODELS[k].role === 'chat');
const EMBED_MODEL = Object.keys(MODELS).find(k => MODELS[k].role === 'embed');

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml', '.png': 'image/png', '.ico': 'image/x-icon',
  '.woff2': 'font/woff2', '.map': 'application/json'
};

// ---------- low-level HTTP ----------
function fetchUrl(targetUrl, opts = {}) {
  return new Promise((resolve, reject) => {
    let u;
    try { u = new URL(targetUrl); } catch (e) { return reject(new Error('bad url')); }
    const lib = u.protocol === 'https:' ? https : http;
    const headers = Object.assign({}, opts.headers || {});
    const body = opts.body == null ? null : Buffer.isBuffer(opts.body) ? opts.body : Buffer.from(String(opts.body));
    if (body && !headers['Content-Length']) headers['Content-Length'] = body.length;
    const req = lib.request({
      hostname: u.hostname, port: u.port || (u.protocol === 'https:' ? 443 : 80),
      path: u.pathname + u.search, method: opts.method || 'GET', headers
    }, res => {
      const chunks = [];
      res.on('data', c => chunks.push(c));
      res.on('end', () => resolve({ status: res.statusCode, headers: res.headers, body: Buffer.concat(chunks) }));
    });
    req.on('error', reject);
    req.setTimeout(opts.timeout || 60000, () => req.destroy(new Error('timeout')));
    if (body) req.write(body);
    req.end();
  });
}

async function ollamaJson(ep, payload) {
  const r = await fetchUrl(OLLAMA + ep, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    timeout: 600000
  });
  let data = null;
  try { data = JSON.parse(r.body.toString('utf8')); } catch (e) {}
  return { status: r.status, data };
}

function ollamaStream(payload, onMessage) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify(payload);
    const u = new URL(OLLAMA + '/api/chat');
    const lib = u.protocol === 'https:' ? https : http;
    const req = lib.request({
      hostname: u.hostname, port: u.port || (u.protocol === 'https:' ? 443 : 80),
      path: u.pathname, method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) }
    }, res => {
      if (res.statusCode >= 400) {
        const chunks = []; res.on('data', c => chunks.push(c));
        res.on('end', () => reject(new Error('ollama ' + res.statusCode + ': ' + Buffer.concat(chunks).toString().slice(0, 500))));
        return;
      }
      let buf = '', last = null;
      res.on('data', chunk => {
        buf += chunk.toString('utf8');
        let idx;
        while ((idx = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, idx).trim();
          buf = buf.slice(idx + 1);
          if (!line) continue;
          try { const obj = JSON.parse(line); onMessage(obj); last = obj; } catch (e) { /* skip */ }
        }
      });
      res.on('end', () => resolve(last));
      res.on('error', reject);
    });
    req.on('error', reject);
    req.setTimeout(600000, () => req.destroy(new Error('timeout')));
    req.write(body);
    req.end();
  });
}

// ---------- web search ----------
function decodeEntities(s) {
  return s.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&#39;/g, "'").replace(/&nbsp;/g, ' ');
}
function stripTags(s) { return decodeEntities(s.replace(/<[^>]+>/g, '')).replace(/\s+/g, ' ').trim(); }

async function webSearch(query) {
  if (!query) return [];
  const u = 'https://html.duckduckgo.com/html/?q=' + encodeURIComponent(query);
  let r;
  try {
    r = await fetchUrl(u, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml', 'Accept-Language': 'en-US,en;q=0.9'
      },
      timeout: 20000
    });
  } catch (e) {
    return [{ title: 'Search failed', url: '', snippet: 'Web search request error: ' + e.message }];
  }
  const html = r.body.toString('utf8');
  const out = [];
  const blocks = html.split(/<div class="result results_links /);
  for (let i = 1; i < blocks.length && out.length < 6; i++) {
    const b = blocks[i];
    const a = b.match(/class="result__a"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/);
    if (!a) continue;
    let href = a[1];
    const uddg = (href.match(/uddg=([^&]+)/) || [])[1];
    if (uddg) href = decodeURIComponent(uddg);
    if (/duckduckgo\.com\/y\.js|ad_domain=/.test(href)) continue; // skip ads
    const s = b.match(/class="result__snippet"[^>]*>([\s\S]*?)<\/a>/);
    out.push({ title: stripTags(a[2]), url: href, snippet: s ? stripTags(s[1]) : '' });
  }
  return out;
}

// ---------- image handling ----------
async function imageUrlToBase64(urlStr) {
  const m = urlStr.match(/^data:([^;]+)?;base64,(.*)$/s);
  if (m) return m[2];
  const r = await fetchUrl(urlStr, { timeout: 30000 });
  return r.body.toString('base64');
}

// ---------- message conversion ----------
async function convertMessages(messages) {
  const out = [];
  for (const m of messages || []) {
    if (m.role === 'system') { out.push({ role: 'system', content: typeof m.content === 'string' ? m.content : (Array.isArray(m.content) ? m.content.map(p => p.text || '').join('') : '') }); continue; }
    if (m.role === 'tool') { out.push({ role: 'tool', content: typeof m.content === 'string' ? m.content : JSON.stringify(m.content) }); continue; }
    if (m.role === 'assistant' && m.tool_calls) {
      out.push({
        role: 'assistant', content: m.content || '',
        tool_calls: m.tool_calls.map(tc => ({
          function: { name: tc.function.name, arguments: typeof tc.function.arguments === 'string' ? tc.function.arguments : JSON.stringify(tc.function.arguments || {}) }
        }))
      });
      continue;
    }
    let content = m.content;
    let images = undefined;
    if (Array.isArray(content)) {
      let text = '';
      const imgs = [];
      for (const part of content) {
        if (part.type === 'text') text += part.text || '';
        else if (part.type === 'image_url' && part.image_url && part.image_url.url) {
          try { imgs.push(await imageUrlToBase64(part.image_url.url)); } catch (e) { /* skip bad image */ }
        }
      }
      content = text;
      if (imgs.length) images = imgs;
    }
    const rec = { role: m.role, content: content == null ? '' : content };
    if (images) rec.images = images;
    out.push(rec);
  }
  return out;
}

function responseFormatToNative(rf) {
  if (!rf) return undefined;
  if (rf.type === 'json_object') return 'json';
  if (rf.type === 'json_schema') return (rf.json_schema && rf.json_schema.schema) ? rf.json_schema.schema : (rf.json_schema || undefined);
  if (rf.type === 'text') return undefined;
  return undefined;
}

function normalizeToolCalls(tcs) {
  if (!tcs) return undefined;
  return tcs.map((tc, i) => ({
    id: tc.id || ('call_' + crypto.randomBytes(4).toString('hex')),
    index: tc.index != null ? tc.index : i,
    type: 'function',
    function: { name: tc.function && tc.function.name, arguments: typeof (tc.function && tc.function.arguments) === 'string' ? (tc.function.arguments) : JSON.stringify((tc.function && tc.function.arguments) || {}) }
  }));
}

// ---------- response helpers ----------
function setCors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', '*');
  res.setHeader('Access-Control-Expose-Headers', 'X-Usage');
}
function sendJSON(res, status, obj) {
  setCors(res);
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(obj));
}
function sendError(res, status, message, code, type) {
  sendJSON(res, status, { error: { message, type: type || 'api_error', param: null, code: code || null } });
}
function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = []; let size = 0;
    req.on('data', c => { size += c.length; if (size > 60 * 1024 * 1024) { req.destroy(); reject(new Error('body too large')); return; } chunks.push(c); });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

function genId() { return 'chatcmpl-' + Date.now().toString(36) + crypto.randomBytes(3).toString('hex'); }

function writeChunk(res, id, model, delta, finishReason) {
  res.write('data: ' + JSON.stringify({
    id, object: 'chat.completion.chunk', created: Math.floor(Date.now() / 1000),
    model, system_fingerprint: 'fp_eburon',
    choices: [{ index: 0, delta, finish_reason: finishReason || null }]
  }) + '\n\n');
}

// ---------- handlers ----------
async function handleModels(res) {
  const created = Math.floor(Date.now() / 1000);
  const data = Object.values(MODELS).map(m => ({
    id: m.id, object: 'model', created, owned_by: m.owned_by,
    name: m.name, family: m.family, founder: FOUNDER, maker: MAKER,
    context_length: m.context_length, parameters: m.parameters, quantization: m.quantization,
    dimensions: m.dimensions || undefined, role: m.role,
    description: m.description,
    capabilities: m.capabilities,
    capabilities_labeled: m.capabilities.map(c => CAP_LABELS[c] || c)
  }));
  sendJSON(res, 200, { object: 'list', data, founder: FOUNDER, maker: MAKER });
}

function handleTags(res) {
  const created = Math.floor(Date.now() / 1000);
  const models = Object.values(MODELS).map(m => ({
    name: m.id, model: m.id, modified_at: new Date().toISOString(),
    size: 0, digest: 'eburon-' + m.id,
    details: { parent_model: '', format: 'gguf', family: m.family, parameter_size: m.parameters, quantization_level: m.quantization, context_length: m.context_length },
    capabilities: m.capabilities
  }));
  sendJSON(res, 200, { models });
}

async function handleEmbeddings(res, bodyStr) {
  let body;
  try { body = JSON.parse(bodyStr || '{}'); } catch (e) { return sendError(res, 400, 'Invalid JSON'); }
  const model = body.model && MODELS[body.model] ? body.model : EMBED_MODEL;
  if (MODELS[model].role !== 'embed') return sendError(res, 400, 'Model ' + body.model + ' does not support embeddings. Use ' + EMBED_MODEL + '.');
  const input = body.input;
  const payload = { model, input };
  if (body.options) payload.options = body.options;
  const r = await ollamaJson('/api/embed', payload);
  if (r.status >= 400 || !r.data || !r.data.embeddings) {
    return sendError(res, r.status || 500, (r.data && r.data.error) || 'Embedding failed', null, 'api_error');
  }
  const data = r.data.embeddings.map((vec, i) => ({ object: 'embedding', index: i, embedding: vec }));
  sendJSON(res, 200, {
    object: 'list', model, data,
    usage: { prompt_tokens: r.data.prompt_eval_count || 0, total_tokens: r.data.prompt_eval_count || 0 }
  });
}

async function handleChat(res, bodyStr, isStream) {
  let body;
  try { body = JSON.parse(bodyStr || '{}'); } catch (e) { return sendError(res, 400, 'Invalid JSON'); }
  const model = body.model;
  if (!model || !MODELS[model]) return sendError(res, 404, 'Model not found: ' + model + '. Available: ' + CHAT_MODELS.join(', '));
  if (MODELS[model].role !== 'chat') return sendError(res, 400, 'Model ' + model + ' is not a chat model. Use ' + CHAT_MODELS.join(', '));

  const think = body.think !== undefined ? !!body.think : true;
  const tools = Array.isArray(body.tools) && body.tools.length ? body.tools : null;
  const format = responseFormatToNative(body.response_format);

  const options = {};
  if (body.max_tokens != null) options.num_predict = body.max_tokens;
  if (body.max_completion_tokens != null) options.num_predict = body.max_completion_tokens;
  if (body.temperature != null) options.temperature = body.temperature;
  if (body.top_p != null) options.top_p = body.top_p;
  if (body.top_k != null) options.top_k = body.top_k;
  if (body.seed != null) options.seed = body.seed;
  if (body.stop != null) options.stop = Array.isArray(body.stop) ? body.stop : [body.stop];
  if (body.frequency_penalty != null) options.frequency_penalty = body.frequency_penalty;
  if (body.presence_penalty != null) options.presence_penalty = body.presence_penalty;
  if (body.repeat_penalty != null) options.repeat_penalty = body.repeat_penalty;

  let nativeMessages;
  try { nativeMessages = await convertMessages(body.messages || []); }
  catch (e) { return sendError(res, 400, 'Message conversion failed: ' + e.message); }

  const basePayload = { model, stream: true, think, options };
  if (tools) basePayload.tools = tools;
  if (format != null) basePayload.format = format;
  if (body.keep_alive != null) basePayload.keep_alive = body.keep_alive;

  const chatId = genId();
  const created = Math.floor(Date.now() / 1000);

  if (isStream) {
    setCors(res);
    res.writeHead(200, {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache', 'Connection': 'keep-alive', 'X-Accel-Buffering': 'no'
    });
    writeChunk(res, chatId, model, { role: 'assistant' }, null);
  }

  const messages = nativeMessages.slice();
  let finalContent = '', finalReasoning = '', finalToolCalls = null;
  let usage = null;

  try {
    for (let iter = 0; iter < (tools ? MAX_TOOL_ITER : 1); iter++) {
      let content = '', reasoning = '', toolCalls = null, lastDone = null;
      await ollamaStream(Object.assign({}, basePayload, { messages }), msg => {
        if (msg.message) {
          if (msg.message.content) { content += msg.message.content; if (isStream) writeChunk(res, chatId, model, { content: msg.message.content }, null); }
          if (msg.message.thinking) { reasoning += msg.message.thinking; if (isStream) writeChunk(res, chatId, model, { reasoning: msg.message.thinking }, null); }
          if (msg.message.tool_calls) toolCalls = msg.message.tool_calls;
        }
        if (msg.done) { lastDone = msg; usage = msg; }
      });

      if (!toolCalls) {
        finalContent = content; finalReasoning = reasoning;
        if (isStream) {
          writeChunk(res, chatId, model, {}, 'stop');
          res.write('data: [DONE]\n\n');
          res.end();
        }
        break;
      }

      // execute tool calls
      const normCalls = toolCalls.map(tc => {
        const name = tc.function && tc.function.name;
        const args = normalizeArgs(tc.function && tc.function.arguments, name);
        return { name, args };
      });
      const assistantMsg = {
        role: 'assistant', content: content || '',
        tool_calls: normCalls.map(c => ({ function: { name: c.name, arguments: JSON.stringify(c.args) } }))
      };
      messages.push(assistantMsg);
      for (const c of normCalls) {
        const name = c.name;
        const query = argQuery(c.args);
        if (isStream) res.write('event: eburon.tool\ndata: ' + JSON.stringify({ tool: name, args: c.args, status: 'running', query }) + '\n\n');
        let result;
        if (name === 'web_search') {
          result = { query, results: await webSearch(query) };
        } else {
          result = { error: 'Tool "' + name + '" is not executable in the Eburon gateway. Built-in tools: web_search.' };
        }
        if (isStream) res.write('event: eburon.tool\ndata: ' + JSON.stringify({ tool: name, args: c.args, status: 'done', result }) + '\n\n');
        messages.push({ role: 'tool', content: JSON.stringify(result) });
      }
      // loop again so the model can use the tool results
    }

    // fallback: if the tool loop exhausted without a final answer, force one without tools
    if (tools && !finalContent) {
      let content = '', reasoning = '';
      await ollamaStream(Object.assign({}, basePayload, { messages, tools: undefined }), msg => {
        if (msg.message) {
          if (msg.message.content) { content += msg.message.content; if (isStream) writeChunk(res, chatId, model, { content: msg.message.content }, null); }
          if (msg.message.thinking) { reasoning += msg.message.thinking; if (isStream) writeChunk(res, chatId, model, { reasoning: msg.message.thinking }, null); }
        }
        if (msg.done) usage = msg;
      });
      finalContent = content; finalReasoning = reasoning;
      if (isStream) { writeChunk(res, chatId, model, {}, 'stop'); res.write('data: [DONE]\n\n'); res.end(); }
    }

    if (!isStream) {
      const message = { role: 'assistant', content: finalContent };
      if (finalReasoning) message.reasoning = finalReasoning;
      const tcNorm = normalizeToolCalls(finalToolCalls);
      if (tcNorm) message.tool_calls = tcNorm;
      sendJSON(res, 200, {
        id: chatId, object: 'chat.completion', created, model, system_fingerprint: 'fp_eburon',
        choices: [{ index: 0, message, finish_reason: finalToolCalls ? 'tool_calls' : 'stop' }],
        usage: {
          prompt_tokens: (usage && usage.prompt_eval_count) || 0,
          completion_tokens: (usage && usage.eval_count) || 0,
          total_tokens: ((usage && usage.prompt_eval_count) || 0) + ((usage && usage.eval_count) || 0)
        }
      });
    }
  } catch (e) {
    if (isStream) {
      try { writeChunk(res, chatId, model, { content: '\n\n[error: ' + e.message + ']' }, 'stop'); res.write('data: [DONE]\n\n'); res.end(); } catch (_) {}
    } else {
      sendError(res, 502, 'Upstream error: ' + e.message);
    }
  }
}

function safeParse(s) { try { return JSON.parse(s); } catch (e) { return null; } }

function normalizeArgs(raw, name) {
  if (raw == null) return {};
  if (typeof raw === 'object') return raw;
  const s = String(raw).trim();
  if (!s) return {};
  const parsed = safeParse(s);
  if (parsed && typeof parsed === 'object') return parsed;
  const m = s.match(/["']?query["']?\s*[:=]\s*["']?(.+?)["']?\s*$/i);
  if (m) return { query: m[1].trim() };
  if (name === 'web_search') return { query: s };
  return { input: s };
}
function argQuery(args) {
  if (!args || typeof args !== 'object') return '';
  return args.query || args.q || args.search || Object.values(args).find(v => typeof v === 'string') || '';
}

// ---------- collaboration: builder <-> reviewer loop ----------
const REVIEWER_SYSTEM = [
  'You are the Eburon AI Reviewer, the quality gate in a dual-model agentic pipeline by Eburon AI (founder: Jo Lernout).',
  'A builder model produced a draft toward a user goal. Critically review it for: correctness, completeness, code quality, adherence to the goal, edge cases, and security.',
  'Be specific and actionable. Never approve incomplete or incorrect work.',
  'End your response with a verdict as a fenced json block using three backticks and json, with this exact shape:',
  '{"verdict":"approve" or "revise","score":integer 1-10,"summary":"one line","issues":["specific problem 1","..."],"suggestions":["concrete fix 1","..."],"next_action":"precise instruction to the builder for the next attempt, or \\"done\\" if approved"}',
  'Approve (verdict approve, score>=8) ONLY when the output is correct, complete, and production-ready. Otherwise revise with concrete, numbered fixes the builder can act on directly.'
].join('\n');

function extractVerdict(text) {
  if (!text) return null;
  let candidates = [];
  const fences = [...text.matchAll(/```(?:json)?\s*([\s\S]*?)```/g)];
  for (const f of fences) candidates.push(f[1]);
  const braces = [...text.matchAll(/\{[\s\S]*\}/g)];
  for (const b of braces) candidates.push(b[0]);
  candidates.push(text);
  for (const c of candidates) {
    try {
      const o = JSON.parse(c.trim());
      if (o && typeof o === 'object' && (o.verdict === 'approve' || o.verdict === 'revise')) {
        return normalizeVerdict(o);
      }
    } catch (e) { /* try regex salvage */ }
  }
  // regex salvage for slightly-malformed JSON
  const vm = text.match(/"verdict"\s*:\s*"(approve|revise)"/i);
  if (!vm) return null;
  const sm = text.match(/"score"\s*:\s*(\d+)/);
  const sum = text.match(/"summary"\s*:\s*"((?:[^"\\]|\\.)*)"/);
  const nam = text.match(/"next_action"\s*:\s*"((?:[^"\\]|\\.)*)"/);
  const issues = extractJsonStringArray(text, 'issues');
  const suggestions = extractJsonStringArray(text, 'suggestions');
  return normalizeVerdict({
    verdict: vm[1],
    score: sm ? parseInt(sm[1], 10) : (vm[1] === 'approve' ? 9 : 6),
    summary: sum ? sum[1].replace(/\\"/g, '"') : '',
    issues, suggestions,
    next_action: nam ? nam[1].replace(/\\"/g, '"') : (vm[1] === 'approve' ? 'done' : 'Improve per issues above.')
  });
}
function extractJsonStringArray(text, key) {
  const m = text.match(new RegExp('"' + key + '"\\s*:\\s*\\[([\\s\\S]*?)\\]', 'i'));
  if (!m) return [];
  return [...m[1].matchAll(/"((?:[^"\\]|\\.)*)"/g)].map(x => x[1].replace(/\\"/g, '"').replace(/\\n/g, '\n')).filter(s => s.trim());
}
function normalizeVerdict(o) {
  if (!Array.isArray(o.issues)) o.issues = o.issues ? [String(o.issues)] : [];
  if (!Array.isArray(o.suggestions)) o.suggestions = o.suggestions ? [String(o.suggestions)] : [];
  if (typeof o.score !== 'number') o.score = o.verdict === 'approve' ? 9 : 6;
  if (typeof o.next_action !== 'string') o.next_action = o.verdict === 'approve' ? 'done' : 'Improve the draft per the issues above.';
  if (typeof o.summary !== 'string') o.summary = '';
  return o;
}

function collabEvent(res, event, obj) {
  res.write('event: eburon.collab.' + event + '\n');
  res.write('data: ' + JSON.stringify(obj) + '\n\n');
}

async function runModelOnce(payload) {
  let content = '', reasoning = '';
  await ollamaStream(payload, msg => {
    if (msg.message) {
      if (msg.message.content) content += msg.message.content;
      if (msg.message.thinking) reasoning += msg.message.thinking;
    }
  });
  return { content, reasoning };
}

async function handleCollab(res, bodyStr) {
  let body;
  try { body = JSON.parse(bodyStr || '{}'); } catch (e) { return sendError(res, 400, 'Invalid JSON'); }
  const builderModel = body.builder || body.builder_model || 'eburon-pro';
  const reviewerModel = body.reviewer || body.reviewer_model || 'eburon-vision';
  if (!MODELS[builderModel] || MODELS[builderModel].role !== 'chat') return sendError(res, 404, 'Builder model not found: ' + builderModel);
  if (!MODELS[reviewerModel] || MODELS[reviewerModel].role !== 'chat') return sendError(res, 404, 'Reviewer model not found: ' + reviewerModel);
  const maxIter = Math.min(Math.max(parseInt(body.max_iterations || '3', 10), 1), 6);
  const think = body.think !== undefined ? !!body.think : true;
  const approveScore = parseFloat(body.approve_score || '8');

  let goalMessages;
  try { goalMessages = await convertMessages(body.messages || []); }
  catch (e) { return sendError(res, 400, 'Message conversion failed: ' + e.message); }
  if (!goalMessages.length) return sendError(res, 400, 'No messages provided');

  const options = {};
  if (body.max_tokens != null) options.num_predict = body.max_tokens; else options.num_predict = 400;
  if (body.temperature != null) options.temperature = body.temperature; else options.temperature = 0.5;
  if (body.top_p != null) options.top_p = body.top_p;
  options.num_ctx = body.num_ctx || 8192;

  setCors(res);
  res.writeHead(200, {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache', 'Connection': 'keep-alive', 'X-Accel-Buffering': 'no'
  });

  collabEvent(res, 'start', { builder: builderModel, reviewer: reviewerModel, max_iterations: maxIter, builder_name: MODELS[builderModel].name, reviewer_name: MODELS[reviewerModel].name });

  const goalText = goalMessages.map(m => (m.role === 'user' ? m.content : '')).filter(Boolean).join('\n').slice(0, 4000);
  let builderMessages = goalMessages.slice();
  let bestContent = '', bestScore = -1, lastVerdict = null;
  const chatId = genId();
  const created = Math.floor(Date.now() / 1000);

  try {
    for (let iter = 1; iter <= maxIter; iter++) {
      // ---- builder turn ----
      collabEvent(res, 'builder.start', { iteration: iter, model: builderModel });
      const builderPayload = { model: builderModel, stream: true, think, options, messages: builderMessages };
      let bContent = '', bReasoning = '';
      await ollamaStream(builderPayload, msg => {
        if (msg.message) {
          if (msg.message.content) { bContent += msg.message.content; collabEvent(res, 'builder.delta', { iteration: iter, content: msg.message.content }); }
          if (msg.message.thinking) { bReasoning += msg.message.thinking; collabEvent(res, 'builder.delta', { iteration: iter, reasoning: msg.message.thinking }); }
        }
      });
      collabEvent(res, 'builder.done', { iteration: iter, content: bContent, reasoning: bReasoning });

      if (bContent.trim().length > bestContent.trim().length * 0.5 || bestScore < 0) { bestContent = bContent; }

      // ---- reviewer turn ----
      collabEvent(res, 'review.start', { iteration: iter, model: reviewerModel });
      const reviewUserMsg = 'USER GOAL:\n' + goalText + '\n\nBUILDER DRAFT (iteration ' + iter + '):\n' + bContent + '\n\nReview this draft against the goal. Return ONLY the verdict json object, no prose, no markdown fences.';
      const verdictSchema = { type: 'object', properties: { verdict: { type: 'string', enum: ['approve', 'revise'] }, score: { type: 'integer' }, summary: { type: 'string' }, issues: { type: 'array', items: { type: 'string' } }, suggestions: { type: 'array', items: { type: 'string' } }, next_action: { type: 'string' } }, required: ['verdict', 'score', 'summary', 'issues', 'suggestions', 'next_action'] };
      const reviewerPayload = { model: reviewerModel, stream: true, think: false, format: verdictSchema, options: { num_predict: 400, temperature: 0.2 }, messages: [{ role: 'system', content: REVIEWER_SYSTEM }, { role: 'user', content: reviewUserMsg }] };
      let rContent = '', rReasoning = '';
      await ollamaStream(reviewerPayload, msg => {
        if (msg.message) {
          if (msg.message.content) { rContent += msg.message.content; collabEvent(res, 'review.delta', { iteration: iter, content: msg.message.content }); }
          if (msg.message.thinking) { rReasoning += msg.message.thinking; collabEvent(res, 'review.delta', { iteration: iter, reasoning: msg.message.thinking }); }
        }
      });
      let verdict = extractVerdict(rContent);
      if (!verdict) {
        verdict = { verdict: 'revise', score: 5, summary: 'Review incomplete; defaulting to revise.', issues: ['Reviewer did not return a parseable verdict.'], suggestions: ['Re-run the builder and request a cleaner, complete output.'], next_action: 'Produce a complete, clean response to the original goal.' };
      }
      lastVerdict = verdict;
      if (verdict.score > bestScore) { bestScore = verdict.score; bestContent = bContent; }
      collabEvent(res, 'review.done', { iteration: iter, content: rContent, reasoning: rReasoning, verdict });

      const approved = verdict.verdict === 'approve' || verdict.score >= approveScore;
      collabEvent(res, 'iteration', { iteration: iter, verdict: verdict.verdict, score: verdict.score, approved, finished: approved || iter >= maxIter });

      if (approved) break;

      // ---- feed reviewer feedback to builder for next iteration ----
      const feedback = 'REVIEWER FEEDBACK (score ' + verdict.score + '/10):\nIssues:\n- ' + verdict.issues.join('\n- ') + '\nSuggestions:\n- ' + verdict.suggestions.join('\n- ') + '\nNext action: ' + verdict.next_action + '\n\nProduce a fully revised, complete response to the original goal that fixes every issue. Output the full artifact/code, not a diff.';
      builderMessages = goalMessages.concat([{ role: 'assistant', content: bContent }, { role: 'user', content: feedback }]);
    }

    // ---- final ----
    collabEvent(res, 'final', { content: bestContent, score: bestScore, iterations: maxIter, verdict: lastVerdict });
    // also emit a normal OpenAI-style final chunk so generic clients get content
    writeChunk(res, chatId, builderModel, { role: 'assistant', content: bestContent }, 'stop');
    res.write('data: [DONE]\n\n');
    res.end();
  } catch (e) {
    try { collabEvent(res, 'error', { message: e.message }); writeChunk(res, chatId, builderModel, { content: '\n\n[error: ' + e.message + ']' }, 'stop'); res.write('data: [DONE]\n\n'); res.end(); } catch (_) {}
  }
}

// ---------- static files ----------
function serveStatic(res, urlPath) {
  let p = decodeURIComponent(urlPath.split('?')[0]);
  if (p === '/' || p === '') p = '/index.html';
  const fp = path.join(PUBLIC_DIR, p);
  if (!fp.startsWith(PUBLIC_DIR)) return sendError(res, 403, 'Forbidden');
  fs.readFile(fp, (err, buf) => {
    if (err) {
      // SPA fallback
      if (p !== '/index.html') { fs.readFile(path.join(PUBLIC_DIR, 'index.html'), (e2, b2) => { if (e2) return sendError(res, 404, 'Not found'); setCors(res); res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' }); res.end(b2); }); return; }
      return sendError(res, 404, 'Not found');
    }
    const ext = path.extname(fp).toLowerCase();
    setCors(res);
    res.writeHead(200, { 'Content-Type': MIME[ext] || 'application/octet-stream', 'Cache-Control': 'public, max-age=60' });
    res.end(buf);
  });
}

// ---------- router ----------
const server = http.createServer(async (req, res) => {
  const u = new URL(req.url, 'http://' + (req.headers.host || 'localhost'));
  const p = u.pathname;
  const method = req.method;

  if (method === 'OPTIONS') { setCors(res); res.writeHead(204); res.end(); return; }

  try {
    if (method === 'GET' && (p === '/v1/models' || p === '/api/models' || p === '/models')) return handleModels(res);
    if (method === 'GET' && p === '/api/tags') return handleTags(res);
    if (method === 'GET' && (p === '/health' || p === '/v1/health')) return sendJSON(res, 200, { status: 'ok', maker: MAKER, founder: FOUNDER, ollama: OLLAMA, models: Object.keys(MODELS) });

    if (method === 'POST' && (p === '/v1/chat/completions' || p === '/chat/completions' || p === '/api/chat/completions')) {
      const body = await readBody(req);
      let parsed = null; try { parsed = JSON.parse(body); } catch (e) {}
      const isStream = !parsed || parsed.stream !== false;
      return handleChat(res, body, isStream);
    }
    if (method === 'POST' && (p === '/v1/embeddings' || p === '/api/embeddings' || p === '/embeddings')) {
      const body = await readBody(req);
      return handleEmbeddings(res, body);
    }
    if (method === 'POST' && (p === '/v1/chat/collab' || p === '/api/chat/collab' || p === '/chat/collab')) {
      const body = await readBody(req);
      return handleCollab(res, body);
    }

    // everything else → static frontend
    if (method === 'GET') return serveStatic(res, req.url);
    return sendError(res, 404, 'Not found: ' + method + ' ' + p);
  } catch (e) {
    return sendError(res, 500, 'Server error: ' + e.message);
  }
});

server.listen(PORT, HOST, () => {
  console.log('[Eburon API] listening on http://' + HOST + ':' + PORT + '  ->  ' + OLLAMA);
  console.log('[Eburon API] models: ' + Object.keys(MODELS).join(', '));
});

process.on('uncaughtException', e => console.error('[Eburon API] uncaught:', e));
process.on('unhandledRejection', e => console.error('[Eburon API] unhandled:', e));
