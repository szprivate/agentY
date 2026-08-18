# Slack — a second line into agentY

The ComfyUI side panel stays exactly what it was. This is a **second** way in and
a second place to watch, so work started at the desk can be followed — and
answered, and steered — from a phone.

* Every turn is mirrored to your Slack DM as it runs, **including turns you start
  in the panel**. Queue a video from the canvas, walk away, watch it finish.
* A DM back drives **the same conversation the panel is in**, so Slack is another
  window on one session rather than a second, forked one.
* It is **off by default** and stays off until you turn it on. Nothing is sent
  anywhere until you have created your own Slack app and pasted its tokens in.

---

## What it looks like

**One conversation, one Slack thread** — the panel's chat list, in a DM. A root
message names the conversation, and every turn in it is a reply underneath:

```
DM with the bot
──────────────────────────────────────────
🧵  Samurai references          ← a conversation
 │   working on it… / Rendered 6 refs
 │   🔧 run_research — 4 templates
 │   ⏸️ Which should go on?      ← also shown in the DM
 │
🧵  Kaiju night shots           ← another one
 │   Cut to 5s. Here it is.

you: make a title card          ← top level = a NEW conversation
```

- **Reply inside a thread** → that conversation continues, with its history.
- **Post at the top level** → a fresh conversation, like opening a new chat.

Per turn that is at most three replies: the answer (rewritten as it streams), the
working-out (one message that grows — Slack has only one level of threading, and
the conversation takes it, so a message per tool would bury the answer), and the
transient status line, which is cleared at the end.

| in the panel | in the conversation's thread |
|---|---|
| the message bubble | the turn's answer, edited as it streams |
| collapsible tool / thinking / plan blocks | one working-out message that grows |
| the transient status line | one reply, rewritten, then removed |
| media dropped on the canvas | uploaded into the thread |
| an ask you have to answer | a reply that is **also** shown in the DM, so it pings |

A canvas edit is the one thing a phone genuinely cannot show, so it is described
in words instead ("Collected the outputs into a review node on the canvas").

### One turn at a time

There is one pipeline, so one turn runs at a time. While it does, a reply **in
its own thread** steers it (the same as typing into the panel mid-turn); anything
else — another thread, or a new top-level message — is answered with "busy" and
has to be sent again. Putting it into the running conversation would file it
under a chat it was not written for.

---

## Setting it up

### 1. Create the Slack app

At <https://api.slack.com/apps> → **Create New App** → *From scratch*.

**OAuth & Permissions → Bot Token Scopes**, add:

```
chat:write      post and edit messages
files:write     upload finished images and video
im:history      read your DMs to the bot
im:read         list them
im:write        open the DM it posts into
users:read      resolve your member id
```

**Socket Mode → Enable Socket Mode.** This is what lets the bridge work from a
machine with no public address: the host opens an outbound WebSocket to Slack, so
nothing here has to be reachable from the internet and there is no tunnel to run.
Creating the app-level token it asks for gives you the `xapp-…` token below —
make sure it has the `connections:write` scope.

**Event Subscriptions → Subscribe to bot events**, add `message.im`.

Then **Install to Workspace** and copy the Bot User OAuth Token (`xoxb-…`).

### 2. Find your member id

In Slack: your profile → **⋮** → *Copy member ID*. It looks like `U01ABCDEF`.

### 3. Put the three values in agentY

In the ComfyUI sidebar: **⚙ agentY settings → Authentication (.env)**

```
SLACK_BOT_TOKEN       xoxb-…
SLACK_APP_TOKEN       xapp-…
SLACK_ALLOWED_USERS   U01ABCDEF          (comma-separated for more than one)
```

Then open the **Slack bridge** group in the same dialog and turn `enabled` on.
**Restart the agent host** (`:5000`) — the connection is made at startup.

You should get a `💬 Slack bridge connected` line in the panel, and your next turn
appears in your DM with the bot.

---

## `SLACK_ALLOWED_USERS` is not optional

Anyone who can DM the bot can otherwise run generations, tools and scripts on
this machine, and "whoever found the app in the workspace" is not an access rule.
Leave it empty and the bridge still connects but refuses every message — that is
deliberate, and the log says so.

---

## What a message you send means

Resolved in this order, because getting it wrong is worse than any of them:

1. **The agent asked you something** in that conversation and is holding the turn
   open → your message is the answer.
2. **That conversation's turn is running** → your message is *interjected* into
   it, exactly as typing into the panel mid-turn does.
3. **Some other turn is running** → busy; send it again when that one finishes.
4. **Otherwise** → it runs, in the conversation whose thread you replied in, or
   in a new one if you posted at the top level.

### Sending it something

Attach a file and it arrives as an **input**, not as something the agent is told
about — "make this warmer" with a photo works the way you would expect, and so
does a video with "cut this to five seconds". The agent gets a real path on disk
that it can wire into a loader node, hand to the vision agent, or read.

An image is also *looked at* directly where the orchestrator model can read
images; a video is always passed as a path (no model takes one inline). Either
way the file lands in `output/slack_uploads/`.

A photo with no words is a complete message — it starts a turn on its own.

Up to ten attachments per message, and up to `max_download_mb` (250 MB) each.
Anything refused is named in the DM rather than passed over in silence: from a
phone there is no canvas to look at and no terminal to check.

---

## Asking for a file

Beyond the automatic mirror, the agent can hand you a **file** on purpose —
"send me the shot list as JSON", "send me the third frame", "send me that log".
Any type: image, video, audio, text, JSON, a script.

It goes to the same DM and nowhere else — the tool cannot post to a channel, to
another person, or to a workspace. Files a run *generated* are already mirrored
as they land, so it does not re-send those; this is for the ones nothing else
would send. Up to ten per request: a DM is where you read one thing on a phone,
not a folder to sync into.

Anything too large for Slack, or missing from disk, comes back as a path rather
than as silence.

The tool only exists while the bridge is on — with `enabled` off it is not
offered to the agent at all, so it costs nothing on a machine that does not use
it.

---

## Settings

Under **Slack bridge** in the settings dialog (`[slack]` in
`config/settings.default.toml`):

| key | what it does |
|---|---|
| `enabled` | off by default; takes effect on the next agent start |
| `channel` | blank = the DM with the first allowed user. Set a channel id only if the whole team should see the pipeline |
| `allowed_users` | fallback for `SLACK_ALLOWED_USERS`, if you would rather keep the list in settings |
| `show_tools` | tool calls in the thread |
| `show_thinking` | the agent's reasoning in the thread |
| `max_upload_mb` | files above this are named rather than uploaded (Slack rejects them anyway) |
| `max_download_mb` | ceiling on an attachment coming the other way — a phone video sent to the agent. Larger is refused with a note in the DM |

---

## The canvas, from Slack

The agent can see the workflow you have open, because the host asks the panel for
it: a flag on the health check the panel already polls every five seconds, and the
panel posts the same graph + hooks + selection a typed message carries. A message
you sent in the panel counts as an answer too, so the common case costs nothing.

That means **ComfyUI has to be open** somewhere for a Slack turn to see a canvas.
If nothing answers within a few seconds the turn runs without one, and the canvas
tools say so rather than inventing a graph.

A snapshot older than three minutes is **not** used. A graph handed over as
current when it is minutes out of date is worse than none — the agent would edit
nodes that have moved, or report on a workflow you closed, and nothing on either
side would say so.

## What it does not do

* **DMs only.** Channel mentions are not wired up.
* **No slash commands.** `/qa`, `/switch_model` and friends are panel-side.

---

## When it does not connect

The host logs to `logs/` and prints to the `run_agent.ps1` terminal.

Startup problems are reported in the panel as well, because a bridge that is
misconfigured and a bridge that is switched off look identical from the chair.

| what you see | why |
|---|---|
| nothing at all | `enabled` is off |
| `SLACK_APP_TOKEN holds a xoxb- token` | the two tokens are not interchangeable — see below |
| `Slack rejected SLACK_BOT_TOKEN` | wrong token, or the app was never installed to the workspace |
| `not_allowed_token_type` | `SLACK_APP_TOKEN` is not an app-level token — see below |
| `missing_scope` | the app-level token exists but lacks `connections:write` |
| connects, ignores you | your member id is not in `SLACK_ALLOWED_USERS` |
| connects, posts nowhere | no DM could be opened — check `im:write` |

### The two tokens

This is the one that catches everybody, because the bot token is what every page
of the app config puts in front of you and the other one is somewhere else
entirely:

* **`SLACK_BOT_TOKEN`** — `xoxb-…`, from *OAuth & Permissions*. Used for
  everything the bot says and uploads.
* **`SLACK_APP_TOKEN`** — `xapp-…`, from *Basic Information → App-Level Tokens →
  Generate Token and Scopes*, with the **`connections:write`** scope. It is the
  only thing `apps.connections.open` accepts, which is the call that opens the
  Socket Mode connection.

Put the bot token in both fields and Slack answers `not_allowed_token_type`: a
valid token that simply cannot make that call. agentY checks the prefixes before
connecting and names the field that is wrong.
