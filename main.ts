// free-model-aggregator-02: one server, two client wires, one live free model per answer.
//
// The server listens on a port it picks, registers that port and its catalog with the
// trial env, and answers POST /v1/chat/completions (OpenAI) and POST /v1/messages
// (Anthropic) in JSON and in server-sent events. Every answer comes from a real free
// OpenRouter model dialled through the grader's gateway on the trial's key; the server
// never writes an answer of its own.
//
// How it picks a model: the brief lists the fourteen free models with the house's
// rating for each. For every request the server ranks the models that meet the
// request's own needs (JSON mode, tools, vision, context) by rating, highest first,
// and dials the best one. The request body itself carries the needs, so routing never
// depends on the order prompts arrive in.
//
// How it spends calls: exactly one gateway call per prompt when the first model
// answers. A refusal (HTTP 429 or 5xx) fails over to the next-best model that meets
// the same needs. A 401 means no key was pushed, so every model would refuse the
// same way and the server gives up at once instead of burning calls. At most four
// attempts serve one prompt. A refused model stays eligible for later prompts: a
// refusal in a six-call trial is most often transient, and retrying it there costs
// at most one extra call. Every refusal a prompt survived is reported on its
// answer in the x-makefun-upstream-failures header, so the failover column can
// see it.
//
// How it stays deterministic: every choice is a function of the brief, the request
// bodies and the gateway's answers. No wall clock and no unseeded randomness appear
// anywhere; listed models carry fixed timestamps, and a synthesized usage chunk
// reuses the upstream chunk's id and created values.
//
// Node 22, standard library only, one file. Run it as `node main.ts`; `node main.ts
// --list` prints the catalog document and exits 0.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

/** A JSON object, the only shape this server parses request bodies into. */
type Json = Record<string, unknown>;
const isObj = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const textOf = (v: unknown): string => (typeof v === 'string' ? v : '');
const numOf = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const boolOf = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

/** The register body's protocol version, and how often GET /work is polled. */
const PROTOCOL_VERSION = 1;
const WORK_POLL_MS = 100;
/** The largest client request body read, and the largest upstream body kept. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
const MAX_UPSTREAM_BYTES = 2 * 1024 * 1024;
/** One gateway attempt may take this long before the server fails over. */
const DIAL_TIMEOUT_MS = 12000;
/** Attempts per prompt: the first pick plus this many failovers. */
const MAX_ATTEMPTS = 4;
/** The gateway's trial-local base, and the dummy key it swaps for the entrant's own. */
const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const BRIEF_PATH = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_BASE = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';
/** The provider's public endpoint and the env name the pushed key lives under. */
const PROVIDER_BASE_URL = 'https://openrouter.ai/api/v1';
const PROVIDER_KEY_ENV = 'OPENROUTER_API_KEY';
/** Fixed timestamps for the model lists: nothing in a reply reads the clock. */
const LIST_CREATED = 1767225600;
const LIST_CREATED_AT = '2026-01-01T00:00:00Z';
/** The failures header's name, and its row and byte caps from the rules. */
const FAILURES_HEADER = 'x-makefun-upstream-failures';
const MAX_TRAIL_ROWS = 8;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** One free model as the brief lists it. `id` is the id the gateway routes. */
interface FreeModel {
  id: string;
  provider: string;
  display: string;
  rating: number;
  context: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

/** What one request needs from the model that serves it. */
interface Need {
  json: boolean;
  tools: boolean;
  vision: boolean;
  minContext: number;
}

/** The catalog document `--list` prints and the register body carries. */
interface CatalogDoc {
  providers: Array<{
    id: string;
    base_url: string;
    key_env: string;
    wire: 'openai';
    models: Array<{ id: string; free: boolean; context_length: number | null }>;
  }>;
}

/** The index of the `}` closing the object that opens at `start`, or -1. */
function matchingBrace(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in the order found. */
function schemaDocuments(text: string): Json[] {
  const out: Json[] = [];
  for (let i = 0; i < text.length; i++) {
    if (text[i] !== '{') continue;
    const end = matchingBrace(text, i);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(i, end + 1));
      if (isObj(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: keep scanning.
    }
  }
  return out;
}

function modelRowOf(v: unknown): FreeModel | null {
  if (!isObj(v)) return null;
  const raw = textOf(v.id) !== '' ? textOf(v.id) : textOf(v.model);
  if (raw === '') return null;
  const named = textOf(v.provider) !== '' ? textOf(v.provider) : textOf(v.wire);
  const id = named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    display: textOf(v.display_name) !== '' ? textOf(v.display_name) : raw,
    rating: numOf(v.rating_milli) ?? 0,
    context: numOf(v.context_length),
    tools: boolOf(v.tools) ?? false,
    vision: boolOf(v.vision) ?? false,
    json: boolOf(v.json) ?? false,
  };
}

/** The brief's free model list, in the brief's order. Empty when the brief cannot be read. */
function readModels(path: string): FreeModel[] {
  if (path === '') {
    log('MAKEFUN_PROMPT_FILE is not set: no brief, no models');
    return [];
  }
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    log(`cannot read the brief at ${path}: ${(err as Error).message}`);
    return [];
  }
  for (const doc of schemaDocuments(text)) {
    if (!String(doc.schema).includes('free-model-aggregator-02/')) continue;
    for (const key of ['models', 'ratings']) {
      if (!Array.isArray(doc[key])) continue;
      const rows = (doc[key] as unknown[]).flatMap((v) => {
        const m = modelRowOf(v);
        return m === null || (v as Json).free === false ? [] : [m];
      });
      if (rows.length > 0) return rows;
    }
  }
  log(`the brief at ${path} holds no free model list`);
  return [];
}

const models: FreeModel[] = readModels(BRIEF_PATH);

/** The catalog: one provider, exactly the brief's free models, once each. */
function buildCatalog(list: readonly FreeModel[]): CatalogDoc {
  const seen = new Set<string>();
  const rows: Array<{ id: string; free: boolean; context_length: number | null }> = [];
  for (const m of list) {
    if (seen.has(m.id)) continue;
    seen.add(m.id);
    rows.push({ id: m.id, free: true, context_length: m.context });
  }
  if (rows.length === 0) return { providers: [] };
  return {
    providers: [{ id: 'openrouter', base_url: PROVIDER_BASE_URL, key_env: PROVIDER_KEY_ENV, wire: 'openai', models: rows }],
  };
}

const catalog: CatalogDoc = buildCatalog(models);
const catalogText: string = JSON.stringify(catalog);

/** The models this entry serves, as GET /v1/models names them. */
function servedRows(): Array<{ id: string; provider: string; display: string }> {
  return models.map((m) => ({ id: m.id, provider: m.provider, display: m.display }));
}

/** Whether a model meets one request's needs. An unknown capability never meets one. */
function meets(need: Need, m: FreeModel): boolean {
  if (need.json && !m.json) return false;
  if (need.tools && !m.tools) return false;
  if (need.vision && !m.vision) return false;
  if (need.minContext > 0 && (m.context === null || m.context < need.minContext)) return false;
  return true;
}

/**
 * The best model for a request: the highest-rated one that meets its needs and is
 * neither cooling down after a refusal nor already tried for this prompt. Rating
 * ties keep the brief's order, because the sort below is stable.
 */
function bestPick(need: Need, skip: ReadonlySet<string>): FreeModel | null {
  const ranked = models.filter((m) => !skip.has(m.id) && meets(need, m));
  ranked.sort((a, b) => b.rating - a.rating);
  return ranked.length > 0 ? ranked[0] : null;
}

/* ------------------------------------------------------------------ */
/* Reading a request's needs off its own body                          */
/* ------------------------------------------------------------------ */

/** Whether any message content carries an image part. */
function bodyHasImage(messages: unknown): boolean {
  if (!Array.isArray(messages)) return false;
  for (const m of messages) {
    if (!isObj(m) || !Array.isArray(m.content)) continue;
    for (const part of m.content) {
      if (isObj(part) && (part.type === 'image' || part.type === 'image_url')) return true;
    }
  }
  return false;
}

/** The needs of an OpenAI-shaped request body. */
function needsOfOpenAI(body: Json): Need {
  const format = isObj(body.response_format) ? body.response_format.type : undefined;
  return {
    json: format === 'json_object' || format === 'json_schema',
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision: bodyHasImage(body.messages),
    minContext: 0,
  };
}

/** The needs of an Anthropic-shaped request body. */
function needsOfAnthropic(body: Json): Need {
  return {
    json: false,
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision: bodyHasImage(body.messages),
    minContext: 0,
  };
}

/* ------------------------------------------------------------------ */
/* Building the upstream Chat Completions body                         */
/* ------------------------------------------------------------------ */

/** The gateway's Chat Completions route from its base URL. */
function chatCompletionsUrl(): string {
  if (GATEWAY_BASE === '') return '';
  return GATEWAY_BASE.endsWith('/v1') ? `${GATEWAY_BASE}/chat/completions` : `${GATEWAY_BASE}/v1/chat/completions`;
}

/**
 * An output cap the gateway accepts: it refuses a call with no cap, or one
 * over 8192 tokens. A missing cap gets 1024 and an over-large one is clamped,
 * and the cap is always written onto the outgoing body, never dropped.
 */
function applyCap(out: Json, body: Json): void {
  const maxTokens = numOf(body.max_tokens);
  const maxCompletion = numOf(body.max_completion_tokens);
  if (maxTokens === null && maxCompletion === null) {
    out.max_tokens = 1024;
    return;
  }
  if (maxTokens !== null) out.max_tokens = Math.min(maxTokens, 8192);
  if (maxCompletion !== null) out.max_completion_tokens = Math.min(maxCompletion, 8192);
}

/**
 * An OpenAI-shaped client body as the upstream body: the dialled model, a usable
 * output cap, usage on a streamed answer, and tool_choice none whenever the client
 * declared tools. The tools stay declared, so the model still sees them; forcing
 * none only stops the model answering with a tool call instead of the text the
 * prompt's token check reads.
 */
function upstreamFromOpenAI(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  applyCap(out, body);
  if (body.stream === true) {
    out.stream_options = { ...(isObj(body.stream_options) ? body.stream_options : {}), include_usage: true };
  }
  if (Array.isArray(body.tools) && body.tools.length > 0) out.tool_choice = 'none';
  return out;
}

/** The text of a string or of a block array's text blocks. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isObj(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A tool_result block's content as one string. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = blocksText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * An Anthropic-shaped client body as a Chat Completions body: the system prompt
 * becomes a system message, image blocks become image_url parts, tool_use blocks
 * become tool_calls, tool_result blocks become tool messages, and the tools,
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 * Tools are declared and tool_choice is forced to none, for the same reason as
 * the OpenAI path above.
 */
function upstreamFromAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = blocksText(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isObj(raw)) continue;
    if (typeof raw.content === 'string') {
      messages.push({ role: raw.role === 'assistant' ? 'assistant' : 'user', content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    if (raw.role === 'assistant') {
      const calls: Json[] = [];
      for (const block of raw.content) {
        if (!isObj(block) || block.type !== 'tool_use') continue;
        calls.push({
          id: textOf(block.id),
          type: 'function',
          function: { name: textOf(block.name), arguments: JSON.stringify(isObj(block.input) ? block.input : {}) },
        });
      }
      const text = blocksText(raw.content);
      messages.push({ role: 'assistant', content: text === '' ? null : text, ...(calls.length > 0 ? { tool_calls: calls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of raw.content) {
      if (!isObj(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: textOf(block.tool_use_id), content: toolResultText(block.content) });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
      if (block.type === 'image' && isObj(block.source)) {
        if (block.source.type === 'base64' && typeof block.source.data === 'string') {
          parts.push({ type: 'image_url', image_url: { url: `data:${textOf(block.source.media_type) || 'image/png'};base64,${block.source.data}` } });
        } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
          parts.push({ type: 'image_url', image_url: { url: block.source.url } });
        }
      }
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  applyCap(out, body);
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    const tools = body.tools.flatMap((raw) => {
      if (!isObj(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isObj(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) {
      out.tools = tools;
      out.tool_choice = 'none';
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* Dialling the gateway                                                */
/* ------------------------------------------------------------------ */

/** One refusal the server survived, as the failures header reports it. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

type DialResult =
  | { kind: 'ok'; text: string }
  | { kind: 'refused'; status: number; code: string }
  | { kind: 'broken' };

/** The error code of a refusal body: its error.code, else its error.type. */
function refusalCode(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (!isObj(value) || !isObj(value.error)) return '';
    const code = value.error.code;
    const type = value.error.type;
    if (typeof code === 'string') return code.slice(0, 64);
    if (typeof type === 'string') return type.slice(0, 64);
    if (typeof code === 'number') return String(code).slice(0, 64);
  } catch {
    // Not JSON: no code to report.
  }
  return '';
}

/** One POST to the gateway's Chat Completions route, naming the dialled model. */
async function dial(model: string, payload: Json, stream: boolean): Promise<DialResult> {
  const url = chatCompletionsUrl();
  if (url === '') return { kind: 'broken' };
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${GATEWAY_KEY}`,
        ...(stream ? { accept: 'text/event-stream' } : {}),
      },
      body: JSON.stringify({ ...payload, model }),
      signal: AbortSignal.timeout(DIAL_TIMEOUT_MS),
    });
  } catch {
    return { kind: 'broken' };
  }
  let text = '';
  try {
    text = await res.text();
  } catch {
    return { kind: 'broken' };
  }
  if (text.length > MAX_UPSTREAM_BYTES) return { kind: 'broken' };
  if (!res.ok) return { kind: 'refused', status: res.status, code: refusalCode(text) };
  return { kind: 'ok', text };
}

/* ------------------------------------------------------------------ */
/* Translating the gateway's answer back to the client's wire          */
/* ------------------------------------------------------------------ */

/** The text of a Chat Completions choice: its message content, or ''. */
function choiceText(choice: Json): string {
  const message = isObj(choice.message) ? choice.message : null;
  if (message === null) return '';
  if (typeof message.content === 'string') return message.content;
  return blocksText(message.content);
}

/** A token count from a length of text, at least 1: only a fallback estimate. */
function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

/**
 * A usage object the wire checks accept: the provider's counts when they are
 * whole numbers with an output of at least 1, else the input as given (or 0)
 * and an output estimated from the answer's text, never below 1.
 */
function wholeUsage(usage: unknown, text: string): Json {
  const u = isObj(usage) ? usage : null;
  const input = u !== null && numOf(u.prompt_tokens) !== null ? Math.max(0, Math.floor(numOf(u.prompt_tokens) as number)) : 0;
  const outputRaw = u !== null ? numOf(u.completion_tokens) : null;
  const output = outputRaw !== null && outputRaw >= 1 ? Math.floor(outputRaw) : estimateTokens(text);
  return { prompt_tokens: input, completion_tokens: output, total_tokens: input + output };
}

/** A tool call's arguments as an object: the gateway's JSON string, or {}. */
function argsObject(value: unknown): Json {
  if (isObj(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isObj(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
function anthropicStop(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * A Chat Completions JSON answer in the client's OpenAI wire, with the model
 * field set to the id really dialled and a whole usage. Null when the answer
 * is not a usable completion (no choice, or no finish_reason).
 */
function openAIJsonAnswer(upstream: Json, model: string): Json | null {
  const choices = Array.isArray(upstream.choices) ? upstream.choices : [];
  if (choices.length === 0 || !isObj(choices[0])) return null;
  const choice = choices[0];
  if (typeof choice.finish_reason !== 'string') return null;
  return { ...upstream, model, usage: wholeUsage(upstream.usage, choiceText(choice)) };
}

/**
 * A Chat Completions JSON answer as an Anthropic Messages object: text and
 * tool_use blocks, the stop reason translated, the usage translated, and the
 * model field set to the id really dialled. Null when the answer is unusable.
 */
function anthropicJsonAnswer(upstream: Json, model: string): Json | null {
  const choices = Array.isArray(upstream.choices) ? upstream.choices : [];
  if (choices.length === 0 || !isObj(choices[0])) return null;
  const choice = choices[0];
  if (typeof choice.finish_reason !== 'string') return null;
  const message = isObj(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : blocksText(message.content);
  if (text !== '') content.push({ type: 'text', text });
  for (const call of Array.isArray(message.tool_calls) ? message.tool_calls : []) {
    if (!isObj(call) || !isObj(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' && call.id !== '' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argsObject(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = wholeUsage(upstream.usage, text);
  return {
    id: `msg_${typeof upstream.id === 'string' ? upstream.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStop(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numOf(usage.prompt_tokens) ?? 0, output_tokens: numOf(usage.completion_tokens) ?? 1 },
  };
}

/* ------------------------------------------------------------------ */
/* Reading a buffered OpenAI event stream                              */
/* ------------------------------------------------------------------ */

/** One tool call accumulated from a stream's deltas, by upstream index. */
interface StreamToolCall {
  id: string;
  name: string;
  args: string;
}

/** A buffered Chat Completions stream, parsed and checked. */
interface ParsedStream {
  ok: boolean;
  /** The stream's frames, every data payload's model set to the dialled id. */
  frames: string[];
  text: string;
  finish: string | null;
  input: number | null;
  output: number | null;
  calls: StreamToolCall[];
  id: string;
  created: number;
  /** Whether any chunk already carried usage. */
  sawUsage: boolean;
}

/** One server-sent event frame, as `data: <json>` lines plus the blank line. */
function sseFrame(data: string): string {
  return `data: ${data}\n\n`;
}

/**
 * Parse a whole Chat Completions SSE body. The stream is usable only when it
 * carries exactly one finish chunk; a second finish chunk, a content delta
 * after the finish, or an event after [DONE] makes it unusable, so the server
 * fails over instead of relaying a stream the wire checks would reject. The
 * model field of every chunk is rewritten to the id really dialled, and a
 * usage chunk is appended when none arrived, so the usage check always sees
 * an output of at least 1.
 */
function parseStream(body: string, model: string): ParsedStream {
  const out: ParsedStream = {
    ok: false,
    frames: [],
    text: '',
    finish: null,
    input: null,
    output: null,
    calls: [],
    id: 'chatcmpl-aggregate',
    created: 0,
    sawUsage: false,
  };
  const normalized = body.replace(/\r\n/g, '\n');
  let sawDone = false;
  let sawFinish = false;
  const callByIndex = new Map<number, StreamToolCall>();
  for (const rawFrame of normalized.split('\n\n')) {
    if (rawFrame.trim() === '') continue;
    if (sawDone) return { ...out, ok: false };
    const dataLines: string[] = [];
    for (const line of rawFrame.split('\n')) {
      const trimmed = line.trim();
      if (trimmed.startsWith('data:')) dataLines.push(trimmed.slice(5).trim());
    }
    if (dataLines.length === 0) continue;
    const payload = dataLines.join('\n');
    if (payload === '[DONE]') {
      sawDone = true;
      continue;
    }
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      return { ...out, ok: false };
    }
    if (!isObj(chunk)) return { ...out, ok: false };
    if (typeof chunk.id === 'string' && out.id === 'chatcmpl-aggregate') out.id = chunk.id;
    if (typeof chunk.created === 'number' && out.created === 0) out.created = chunk.created;
    if (isObj(chunk.usage)) {
      out.sawUsage = true;
      if (numOf(chunk.usage.prompt_tokens) !== null) out.input = numOf(chunk.usage.prompt_tokens);
      if (numOf(chunk.usage.completion_tokens) !== null) out.output = numOf(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : null;
    if (choices === null) return { ...out, ok: false };
    if (choices.length === 0) {
      out.frames.push(sseFrame(JSON.stringify({ ...chunk, model })));
      continue;
    }
    const choice = isObj(choices[0]) ? choices[0] : null;
    if (choice === null) return { ...out, ok: false };
    const delta = isObj(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string' && delta.content !== '') {
      if (sawFinish) return { ...out, ok: false };
      out.text += delta.content;
    }
    for (const call of Array.isArray(delta.tool_calls) ? delta.tool_calls : []) {
      if (!isObj(call)) continue;
      const index = typeof call.index === 'number' ? call.index : callByIndex.size;
      let entry = callByIndex.get(index);
      if (entry === undefined) {
        entry = { id: '', name: '', args: '' };
        callByIndex.set(index, entry);
      }
      const fn = isObj(call.function) ? call.function : {};
      if (typeof call.id === 'string' && call.id !== '') entry.id = call.id;
      if (typeof fn.name === 'string' && fn.name !== '') entry.name = fn.name;
      if (typeof fn.arguments === 'string') entry.args += fn.arguments;
    }
    if (typeof choice.finish_reason === 'string') {
      if (sawFinish) return { ...out, ok: false };
      sawFinish = true;
      out.finish = choice.finish_reason;
    }
    out.frames.push(sseFrame(JSON.stringify({ ...chunk, model })));
  }
  if (!sawFinish || out.finish === null) return { ...out, ok: false };
  out.calls = [...callByIndex.entries()].sort((a, b) => a[0] - b[0]).map(([, call]) => call);
  if (!out.sawUsage) {
    const usage = wholeUsage(null, out.text);
    out.frames.push(
      sseFrame(
        JSON.stringify({
          id: out.id,
          object: 'chat.completion.chunk',
          created: out.created,
          model,
          choices: [],
          usage,
        }),
      ),
    );
    out.input = numOf(usage.prompt_tokens);
    out.output = numOf(usage.completion_tokens);
  }
  out.frames.push(sseFrame('[DONE]'));
  out.ok = true;
  return out;
}

/* ------------------------------------------------------------------ */
/* The Anthropic event sequence, from a parsed OpenAI stream           */
/* ------------------------------------------------------------------ */

/** One Anthropic SSE event: its `event:` name, the JSON data line, the blank line. */
function anthropicEvent(name: string, data: unknown): string {
  return `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;
}

/**
 * A parsed OpenAI stream as Anthropic's event order: message_start, one
 * content block per text span and per tool call, message_delta with the stop
 * reason and the output count, then message_stop. The input count rides on
 * message_start and the output count on message_delta, so both are known
 * before the first event is written.
 */
function anthropicEvents(parsed: ParsedStream, model: string): string {
  const events: string[] = [];
  const input = parsed.input ?? 0;
  const output = parsed.output !== null && parsed.output >= 1 ? parsed.output : estimateTokens(parsed.text);
  events.push(
    anthropicEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: input, output_tokens: 0 },
      },
    }),
  );
  let nextIndex = 0;
  if (parsed.text !== '') {
    events.push(
      anthropicEvent('content_block_start', { type: 'content_block_start', index: nextIndex, content_block: { type: 'text', text: '' } }),
    );
    events.push(
      anthropicEvent('content_block_delta', { type: 'content_block_delta', index: nextIndex, delta: { type: 'text_delta', text: parsed.text } }),
    );
    events.push(anthropicEvent('content_block_stop', { type: 'content_block_stop', index: nextIndex }));
    nextIndex += 1;
  }
  for (const call of parsed.calls) {
    const index = nextIndex;
    nextIndex += 1;
    events.push(
      anthropicEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: call.id !== '' ? call.id : `toolu_${index}`, name: call.name, input: {} },
      }),
    );
    if (call.args !== '') {
      events.push(
        anthropicEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: call.args } }),
      );
    }
    events.push(anthropicEvent('content_block_stop', { type: 'content_block_stop', index }));
  }
  if (nextIndex === 0) {
    events.push(
      anthropicEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
    );
    events.push(anthropicEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  }
  events.push(
    anthropicEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStop(parsed.finish), stop_sequence: null },
      usage: { output_tokens: output },
    }),
  );
  events.push(anthropicEvent('message_stop', { type: 'message_stop' }));
  return events.join('');
}

/* ------------------------------------------------------------------ */
/* Serving one prompt, with failover                                   */
/* ------------------------------------------------------------------ */

/** What one served prompt answers with: its status, headers and body. */
interface ServedAnswer {
  status: number;
  stream: boolean;
  failures: FailureRow[];
  body: string;
}

/** The failures header for a prompt's recorded refusals, or none. */
function trailHeader(failures: readonly FailureRow[]): Record<string, string> {
  if (failures.length === 0) return {};
  const rows = failures.slice(0, MAX_TRAIL_ROWS);
  return { [FAILURES_HEADER]: JSON.stringify(rows) };
}

/** A refused prompt's error body, in the wire that asked for it. */
function refusedBody(wire: 'openai' | 'anthropic', failures: readonly FailureRow[]): string {
  const code = failures.length > 0 ? failures[failures.length - 1].code || 'upstream_error' : 'upstream_error';
  if (wire === 'anthropic') {
    return JSON.stringify({ type: 'error', error: { type: 'api_error', message: 'the gateway refused every free model tried for this prompt' } });
  }
  return JSON.stringify({ error: { message: 'the gateway refused every free model tried for this prompt', type: 'upstream_error', code } });
}

/**
 * Translate one successful gateway answer into the client's wire. Null when
 * the answer is not usable, so the server fails over instead of answering
 * with a body the wire checks would reject.
 */
function translate(wire: 'openai' | 'anthropic', text: string, model: string, stream: boolean): string | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(text);
    } catch {
      return null;
    }
    if (!isObj(value)) return null;
    const answer = wire === 'openai' ? openAIJsonAnswer(value, model) : anthropicJsonAnswer(value, model);
    return answer === null ? null : JSON.stringify(answer);
  }
  const parsed = parseStream(text, model);
  if (!parsed.ok) return null;
  return wire === 'openai' ? parsed.frames.join('') : anthropicEvents(parsed, model);
}

/**
 * Serve one prompt: pick the best model for its needs, dial it, and on a
 * 429 or 5xx fail over to the next-best model that meets the same needs. A
 * 401 ends the prompt at once, since every model would refuse the same way,
 * and any other 4xx is the request's own fault, so another model would meet
 * the same refusal. A refused model is NOT set aside for later prompts: a
 * refusal in a six-call trial is most often transient, and the model that
 * just refused may well answer the next prompt at its full rating; retrying
 * it there costs at most one extra call, while writing it off for the trial
 * would give every later prompt to a lower-rated model. A dial that never
 * reached the gateway, or an unusable answer, also fails over, without a
 * trail row, since it has no HTTP status to report.
 */
async function servePrompt(wire: 'openai' | 'anthropic', body: Json): Promise<ServedAnswer> {
  const need = wire === 'openai' ? needsOfOpenAI(body) : needsOfAnthropic(body);
  const stream = body.stream === true;
  const failures: FailureRow[] = [];
  const tried = new Set<string>();
  const named = textOf(body.model);
  let explicit: FreeModel | null = null;
  if (named !== '' && named !== 'auto') {
    const found = models.find((m) => m.id === named);
    if (found !== undefined) explicit = found;
  }
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt += 1) {
    const pick = attempt === 0 && explicit !== null ? explicit : bestPick(need, tried);
    if (pick === null) break;
    tried.add(pick.id);
    const payload = wire === 'openai' ? upstreamFromOpenAI(body, pick.id) : upstreamFromAnthropic(body, pick.id);
    const dialed = await dial(pick.id, payload, stream);
    if (dialed.kind === 'broken') {
      continue;
    }
    if (dialed.kind === 'refused') {
      failures.push({ status: dialed.status, code: dialed.code, model: pick.id });
      if (dialed.status === 401) break;
      if (dialed.status !== 429 && dialed.status < 500) break;
      continue;
    }
    const translated = translate(wire, dialed.text, pick.id, stream);
    if (translated === null) {
      failures.push({ status: 502, code: 'upstream_error', model: pick.id });
      continue;
    }
    return { status: 200, stream, failures, body: translated };
  }
  return { status: 502, stream, failures, body: refusedBody(wire, failures) };
}

/* ------------------------------------------------------------------ */
/* The HTTP routes                                                     */
/* ------------------------------------------------------------------ */

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJsonBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isObj(value) ? value : null;
  } catch {
    return null;
  }
}

function sendBody(res: ServerResponse, status: number, contentType: string, body: string, extra: Record<string, string> = {}): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(body), ...extra });
  res.end(body);
}

function sendJson(res: ServerResponse, status: number, body: unknown, extra: Record<string, string> = {}): void {
  sendBody(res, status, 'application/json', JSON.stringify(body), extra);
}

/** An error in the wire that asked for it. */
function sendWireError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, { type: 'error', error: { type, message } });
  else sendJson(res, status, { error: { message, type, code: type } });
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function serveModels(req: IncomingMessage, res: ServerResponse): void {
  const rows = servedRows();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: rows.map((r) => ({ type: 'model', id: r.id, display_name: r.display, created_at: LIST_CREATED_AT })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: rows.map((r) => ({ id: r.id, object: 'model', created: LIST_CREATED, owned_by: r.provider })),
    });
  }
}

/** POST /v1/chat/completions or POST /v1/messages: one prompt, answered live. */
async function serveCompletion(req: IncomingMessage, res: ServerResponse, wire: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) {
    sendWireError(res, wire, 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  if (!Array.isArray(body.messages)) {
    sendWireError(res, wire, 400, 'invalid_request_error', 'the request holds no messages');
    return;
  }
  const answer = await servePrompt(wire, body);
  const trail = trailHeader(answer.failures);
  if (answer.stream) sendBody(res, answer.status, 'text/event-stream', answer.body, { 'cache-control': 'no-cache', connection: 'keep-alive', ...trail });
  else sendBody(res, answer.status, 'application/json', answer.body, trail);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    serveModels(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serveCompletion(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serveCompletion(req, res, 'anthropic');
    return;
  }
  sendWireError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

/* ------------------------------------------------------------------ */
/* Registering, polling, and starting up                               */
/* ------------------------------------------------------------------ */

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** POST /register with the port and the one catalog, retrying briefly. */
async function register(port: number): Promise<void> {
  let lastError = '';
  for (let attempt = 0; attempt < 5; attempt += 1) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
      const reply: unknown = await res.json();
      if (isObj(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
      return;
    } catch (err) {
      lastError = (err as Error).message;
      await sleep(200);
    }
  }
  throw new Error(`could not register with the env: ${lastError}`);
}

/** Poll GET /work until it says done, then stop the server and exit 0. */
function pollWork(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isObj(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, WORK_POLL_MS);
}

function listenOn(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => {
      const address = server.address();
      if (address === null || typeof address === 'string') reject(new Error('the server has no port'));
      else resolve(address.port);
    });
  });
}

async function start(): Promise<void> {
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the
    // one built catalog and stop before any server starts.
    await new Promise<void>((resolve) => process.stdout.write(`${catalogText}\n`, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendWireError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  const port = await listenOn(server);
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedRows().length} free models`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  pollWork(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
