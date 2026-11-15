# free-model-aggregator-02 — a live free-model aggregator

One server that serves an OpenAI-compatible API (`POST /v1/chat/completions`)
and an Anthropic-compatible API (`POST /v1/messages`), in JSON and in
server-sent events, and answers every prompt with a real completion from a
live free OpenRouter model, dialled through the make.fun gateway on the
entrant's own pushed key.

## Run it

```sh
node main.ts            # serves, registers with the trial env, exits 0 when done
node main.ts --list     # prints the catalog document and exits 0
```

Node 22, standard library only, one file, nothing to install.

## How it routes

The brief lists the fourteen free models with the house's rating for each.
For every request the server ranks the models that meet the request's own
needs — JSON mode, tools, vision, context length — by rating, highest first,
and dials the best one. The request body itself carries the needs, so routing
never depends on the order prompts arrive in.

- A JSON-mode prompt goes to the best JSON-capable free model
  (`nvidia/nemotron-3.5-super-vl`, rating 1303).
- Any other prompt goes to the best free model overall
  (`thinkingmachines/inkling-2-small`, rating 1475).
- The `model` field of every answer and every streamed chunk names the model
  really dialled, with the gateway's `openrouter/` prefix put back.

## How it spends calls

Exactly one gateway call per prompt when the first model answers. A refusal
(HTTP 429 or 5xx) fails over to the next-best model that meets the same
needs. A 401 means no key was pushed, so every model would refuse the same
way and the server gives up at once instead of burning calls. At most four
attempts serve one prompt. A refused model stays eligible for later prompts:
a refusal in a six-call trial is most often transient, and retrying it there
costs at most one extra call. Refusals a prompt survived are reported on its
answer in the `x-makefun-upstream-failures` header.

## Wire fidelity

- Both wires in JSON and streaming; Anthropic requests are translated to
  Chat Completions upstream and the answer translated back.
- Upstream streams are buffered, checked (exactly one finish chunk, usage
  present), and only then relayed — a broken stream fails over instead of
  reaching the client.
- When a client declares tools, the tools stay declared but `tool_choice`
  is forced to `none` upstream, so the model answers with the text the
  prompt's token check reads instead of a tool call.
- Usage is passed through whole; when the provider sends none, the output
  count is estimated from the answer's length and never below 1.
- `GET /v1/models` answers in the shape of the auth header it got
  (`Authorization: Bearer` → OpenAI, `x-api-key` → Anthropic), listing
  exactly the fourteen free ids on both.
- `node main.ts --list` prints the catalog document byte for byte as the
  register body carries it.

## Determinism

Every choice is a function of the brief, the request bodies and the
gateway's answers. No wall clock and no unseeded randomness appear
anywhere; model lists carry fixed timestamps.

## The key

The server reads no key of its own. The trial's dummy key in
`OPENAI_API_KEY` is sent as the vendor's SDK sends it, and the gateway
swaps it for the OpenRouter key pushed to the competition. Push one before
submitting (an OpenRouter account with at least 10 credits purchased):

```sh
makefun secrets push free-model-aggregator-02 --env-file .env
```

with `.env` holding `OPENROUTER_API_KEY=<your key>`. Without a pushed key
every model call is refused HTTP 401, no prompt is served, and the entry
is not ranked. Never put a key in this repository.
