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

One message per turn, rewritten as the answer streams in. Under it, a thread with
the detail — the same split the panel makes with collapsible blocks:

| in the panel | in Slack |
|---|---|
| the message bubble | the turn's message, edited as it streams |
| media dropped on the canvas | uploaded to the DM |
| an ask you have to answer | its own message (so it pings) — reply to answer |
| tool calls, thinking, plans, canvas edits | replies in that message's thread |
| the transient status line | one thread reply, rewritten and cleared at the end |

A canvas edit is the one thing a phone genuinely cannot show, so it is described
in words instead ("Collected the outputs into a review node on the canvas").

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

1. **The agent asked you something** and is holding the turn open → your message
   is the answer.
2. **A turn is already running** → your message is *interjected* into it, exactly
   as typing into the panel mid-turn does. It is not a second turn: two turns
   through one pipeline would corrupt both.
3. **Otherwise** → it starts a turn, in whatever conversation the panel last used.

Attach an image and it arrives as an input, not as something the agent is told
about — "make this warmer" with a photo works the way you would expect.

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

---

## What it does not do

* **No canvas context.** A turn started from Slack has no open graph, no hooks and
  no selection — it is a chat turn. Anything that needs the canvas has to be
  started from the panel (and you will still see it in Slack).
* **DMs only.** Channel mentions are not wired up.
* **No slash commands.** `/qa`, `/switch_model` and friends are panel-side.

---

## When it does not connect

The host logs to `logs/` and prints to the `run_agent.ps1` terminal.

| what you see | why |
|---|---|
| nothing at all | `enabled` is off, or the tokens are missing |
| `auth_test failed` | `SLACK_BOT_TOKEN` is wrong, or the app was never installed |
| `could not connect` | Socket Mode is not enabled, or `SLACK_APP_TOKEN` is not an app-level token with `connections:write` |
| connects, ignores you | your member id is not in `SLACK_ALLOWED_USERS` |
| connects, posts nowhere | no DM could be opened — check `im:write` |
