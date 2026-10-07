# Slack — a second line into agentY

Off by default. Turned on, Slack is a second window on the same agent:

- Every turn is mirrored to your Slack DM as it runs, including turns you start in
  the ComfyUI panel.
- A DM back drives the same conversation the panel is in.
- You can send images and video as inputs, and ask the agent to send you files.

The connection is outbound only (Socket Mode), so nothing on your machine has to
be reachable from the internet.

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

**Socket Mode → Enable Socket Mode.** Create the app-level token it asks for, with
the `connections:write` scope. That is the `xapp-…` token.

**Event Subscriptions → Subscribe to bot events**, add `message.im`.

**Install to Workspace** and copy the Bot User OAuth Token (`xoxb-…`).

### 2. Find your member id

In Slack: your profile → **⋮** → *Copy member ID* (looks like `U01ABCDEF`).

### 3. Put the three values in agentY

**agentY settings → Authentication (.env)**:

```
SLACK_BOT_TOKEN       xoxb-…
SLACK_APP_TOKEN       xapp-…
SLACK_ALLOWED_USERS   U01ABCDEF          (comma-separated for more than one)
```

Open the **Slack** group in the same dialog, turn `enabled` on, and **restart the
agent**. The panel shows `💬 Slack bridge connected`, and your next turn appears in
your DM with the bot.

> **`SLACK_ALLOWED_USERS` is required.** Left empty, the bridge connects but
> refuses every message — otherwise anyone who can DM the bot could run tools on
> your machine.

---

## How it works

**One conversation is one Slack thread.**

- **Reply inside a thread** → that conversation continues.
- **Post at the top level** → a new conversation.

A message you send is acknowledged with 👀 at once and ✅ when the turn is done.
Each turn posts its answer (updated as it streams), one message with the
working-out, and a status line that is removed at the end. Generated media is
uploaded into the thread. Canvas edits are described in words.

**What your message means**, in this order:

1. The agent asked you something in that conversation → your message is the answer.
2. That conversation's turn is running → your message goes into the running turn.
3. Another conversation's turn is running → you get "busy"; send it again later.
4. Otherwise → it starts a turn.

**Sending files.** Attach an image or video and it arrives as an input the agent
can use (saved in `output/slack_uploads/`). A photo with no text is a complete
message. Up to ten attachments, 250 MB each (`max_download_mb`).

**Asking for a file.** *"Send me the shot list as JSON"*, *"send me a screenshot of
my workflow"*. Any file type, up to ten per request, only ever to your DM.
Generated media is already mirrored and is not sent twice.

**The canvas.** The agent can see your open workflow only while ComfyUI is open in
a browser. Otherwise the turn runs without a canvas and says so.

**Text commands.** Slack swallows a leading `/`, so reply `undo` or `compact` in
the thread. Other slash commands (`/qa`, `/switch_model`, …) are panel-only.

**Limits.** DMs only; channel mentions are not supported.

---

## Settings

Under **Slack** in the settings dialog (`[slack]` in `config/settings.default.toml`):

| key | what it does |
|---|---|
| `enabled` | off by default; applies at the next agent start |
| `channel` | blank = the DM with the first allowed user; set a channel id to post there instead |
| `allowed_users` | alternative to `SLACK_ALLOWED_USERS` |
| `show_tools` | show tool calls in the thread |
| `show_thinking` | show the agent's reasoning in the thread |
| `max_upload_mb` | larger files are named instead of uploaded |
| `max_download_mb` | largest attachment the agent accepts |

---

## When it does not connect

Startup problems are shown in the panel and in the launcher's terminal.

| what you see | why |
|---|---|
| nothing at all | `enabled` is off |
| `SLACK_APP_TOKEN holds a xoxb- token` | the bot token is in the app-token field |
| `Slack rejected SLACK_BOT_TOKEN` | wrong token, or the app isn't installed to the workspace |
| `not_allowed_token_type` | `SLACK_APP_TOKEN` is not an app-level token |
| `missing_scope` | the app-level token lacks `connections:write` |
| connects, ignores you | your member id isn't in `SLACK_ALLOWED_USERS` |
| connects, posts nowhere | no DM could be opened — check `im:write` |

**The two tokens** are not interchangeable:

- **`SLACK_BOT_TOKEN`** — `xoxb-…`, from *OAuth & Permissions*.
- **`SLACK_APP_TOKEN`** — `xapp-…`, from *Basic Information → App-Level Tokens*,
  with the `connections:write` scope.
