# Slack — a second line into agentY

Off by default. Turned on, Slack is a second window on the same agent:

- Every turn is mirrored to your Slack DM as it runs, including turns you start in
  the ComfyUI panel.
- A DM back drives the same conversation the panel is in.
- You can send images and video as inputs, and ask the agent to send you files.

The connection is outbound only (Socket Mode), so nothing on your machine has to
be reachable from the internet.

This page starts with the personal DM bridge. To let a **team** use agentY from
channels, see [Channels](#channels-several-people-one-agent); to run it on a server, see
[server.md](server.md).

---

## Setting it up

### 1. Create the Slack app

At <https://api.slack.com/apps> → **Create New App** → *From scratch*.

**OAuth & Permissions → Bot Token Scopes**, add:

```
chat:write      post and edit messages
files:write     upload finished images and video
files:read      download what you attach to a message
reactions:write the eyes / check-mark reactions
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

**Limits.** In a DM, one conversation at a time can run (a second is told "busy"); set
`slack.queue_dm_turns` to let DM requests wait in the same line as channel requests.
Channel mentions are supported, see [Channels](#channels-several-people-one-agent).

---

## Channels: several people, one agent

A team can ask agentY from a channel. This is separate from the DM bridge above and
needs a few more things set up; the DM keeps working exactly as before.

**What it does**

- **@mention only.** agentY answers a message that starts with `@agentY`, in a channel
  listed in `slack.channels`. It ignores everything else said there.
- **One thread per request.** The request is the thread: agentY replies under the
  message that mentioned it, and everything about that request (progress, files, questions,
  approvals) stays in that thread. Reply in the thread to continue the same conversation;
  a new `@agentY` at the top level is a new one.
- **It knows who asked.** The person who made the request owns the conversation. Only
  they (and anyone in `slack.approvers` / `SLACK_APPROVERS`, or with the `approver` role in
  `config/users.json`) can answer its questions, say yes to a tool that needs approval, or
  steer the running turn. Anyone else gets "only the requester or an approver may answer
  this".
- **Approvals happen in the thread.** A tool that needs a yes (running a script, installing
  a custom node or an MCP server, switching model) posts a card there that names the
  requester; reply `yes` or `no` (no `@` needed, it is an answer). No answer within
  `slack.approval_timeout_seconds` counts as no. This is asked in Slack even when a ComfyUI
  tab happens to be open on the server, because the person at that screen is not the person
  who asked. Plans and review stops are replies too, so the same rule covers them.
- **A line, not a pile-up.** Up to `server.max_running_turns` requests run at once; the
  rest wait. The requester's thread shows "Queued, number 2 in line", which updates as the
  line moves and disappears when the turn starts. Say `@agentY queue` to see everyone.
  A follow-up message while your request is still waiting is added to it.
- **Files in and out, per request.** Attach images or video to the message that mentions
  agentY (saved under `output/slack_uploads/<channel>_<thread>/`, one folder per request).
  Results, and anything the agent is asked to send, are uploaded into that thread.
- **Memory and cost, per person and per project.** A channel is a project.
  `@agentY remember: I always want 16:9` keeps a note for you;
  `@agentY remember for project: the client is Acme` keeps one for the channel. Both are read
  into every request from that person / channel (`@agentY notes` shows them,
  `@agentY forget my notes` clears yours). Every finished request is added to a ledger
  (`memory/team/usage.jsonl`: who, which channel, tokens, cost); `@agentY usage` or
  `scripts/agenty_users.py usage --by project` sums it up.

**Several turns at once, or a queue?** A queue. The host *can* run conversations in parallel
(each running conversation has a pipeline of its own, with its own buffers, and the ComfyUI
panel relies on that), so two turns do not corrupt each other. What they share is the LLM
providers' rate limits and, for renders, one GPU. So the default is two at a time
(`server.max_running_turns`), the rest in line with their position shown. Raise the number
if your providers allow it; it is capped at `parallel_chats`.

### Create the Slack app (channels)

If you already made the DM app above, add the scopes and events below (then reinstall the
app); otherwise create one from this manifest: **Create New App - From an app manifest**,
paste, create, then follow *Tokens and ids* below.

```yaml
display_information:
  name: agentY
  description: Runs ComfyUI generations and workflows for the team
features:
  app_home:
    home_tab_enabled: false
    messages_tab_enabled: true
    messages_tab_read_only_enabled: false
  bot_user:
    display_name: agentY
    always_online: true
oauth_config:
  scopes:
    bot:
      - app_mentions:read     # hear "@agentY ..." in channels
      - channels:history      # read replies in the request's thread (public channels)
      - groups:history        # ... and in private channels
      - channels:read         # channel names
      - groups:read
      - chat:write            # post and edit messages
      - files:read            # download what you attach
      - files:write           # upload results
      - im:history            # the DM bridge
      - im:read
      - im:write
      - reactions:write       # the eyes / check marks
      - users:read            # who asked (display name)
settings:
  event_subscriptions:
    bot_events:
      - app_mention           # @agentY in a channel
      - message.channels      # replies in a thread (an approval's "yes" needs no @)
      - message.groups
      - message.im            # the DM bridge
  interactivity:
    is_enabled: false
  org_deploy_enabled: false
  socket_mode_enabled: true
  token_rotation_enabled: false
```

`message.channels` / `message.groups` are optional: without them agentY only hears
`@agentY ...`, so an approval has to be answered as `@agentY yes`. Socket Mode means nothing
has to be reachable from the internet.

### Tokens and ids

1. **Basic Information - App-Level Tokens - Generate**, scope `connections:write` - that is
   `SLACK_APP_TOKEN` (`xapp-...`).
2. **Install App**, copy the *Bot User OAuth Token* - `SLACK_BOT_TOKEN` (`xoxb-...`).
3. In the channel: `/invite @agentY` (a bot only hears channels it is in).
4. Channel id: channel name - **View channel details** - the id at the bottom (`C0123ABCD`).
   Member ids: profile - **Copy member ID**.
5. Put them in the server's environment (or **Settings - Authentication (.env)**):

```
SLACK_BOT_TOKEN        xoxb-...
SLACK_APP_TOKEN        xapp-...
SLACK_ALLOWED_USERS    U01ABCDEF,U02GHIJKL        who may make requests
SLACK_ALLOWED_CHANNELS C0123ABCD                  where it answers @mentions
SLACK_APPROVERS        U01ABCDEF                  may approve other people's requests
```

and turn `slack.enabled` on. `slack.channel_members_may_ask = true` lets **any member of an
allowed channel** make requests (then `SLACK_ALLOWED_USERS` only decides who may use DMs);
leave it off to keep to the named people.

Set `security.unattended_tool_policy = "deny"` (the default) so tools that need approval ask
in the thread instead of running unreviewed. For a headless server see
[server.md](server.md).

### What a channel can and cannot do

- Private channels need `groups:*` scopes and the `message.groups` event (in the manifest).
  Group DMs are not handled.
- Canvas features need a ComfyUI tab, which a server does not have: with `server.headless`
  the agent says so and builds and runs workflows directly; results come back as uploads.
- Slack has no buttons here: approvals are `yes` / `no` replies.
- Not live-tested against a real workspace (the tests drive the bridge with a stand-in
  Slack client). Expect to adjust something on the first day; the log line
  `slack: message from ...` in the host's terminal shows what each message was taken to mean.

---

## Settings

Under **Slack** in the settings dialog (`[slack]` in `config/settings.default.toml`):

| key | what it does |
|---|---|
| `enabled` | off by default; applies at the next agent start |
| `channel` | blank = the DM with the first allowed user; set a channel id to post there instead |
| `show_tools` | show tool calls in the thread |
| `show_thinking` | show the agent's reasoning in the thread |
| `max_upload_mb` | larger files are named instead of uploaded |
| `max_download_mb` | largest attachment the agent accepts |
| `channels` | channel ids where an @mention is answered (blank = DMs only); `SLACK_ALLOWED_CHANNELS` adds to it |
| `channel_members_may_ask` | any member of an allowed channel may ask, not only `SLACK_ALLOWED_USERS` |
| `approvers` | may approve other people's requests; `SLACK_APPROVERS` adds to it |
| `approval_timeout_seconds` | how long an approval card waits before it counts as no (600) |
| `queue_dm_turns` | DM requests wait in the shared line instead of "busy" |

Related, under `[server]`: `max_running_turns`, `max_queued_turns`, `headless` - see
[server.md](server.md).

Who may drive the agent is one list: `SLACK_ALLOWED_USERS`, the field in the same Slack section. The old settings key `slack.allowed_users` still works when that variable is empty, but logs a warning and is ignored once the variable is set - move the ids over.

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
| mention in a channel is ignored | the channel id is not in `SLACK_ALLOWED_CHANNELS`, the bot is not invited (`/invite @agentY`), or `app_mention` / `app_mentions:read` is missing |
| "Not authorised" in a channel | the person is not in `SLACK_ALLOWED_USERS` (and `channel_members_may_ask` is off) |
| "only the requester or an approver may answer this" | someone else replied to a request they did not make; add them to `SLACK_APPROVERS` if they should |
| a tool is never approved | the card waits in the request's thread for `yes` / `no`; check `slack.approval_timeout_seconds` |
| attached files do not arrive in a channel | add the `files:read` scope, attach them in the same message as the mention |

**The two tokens** are not interchangeable:

- **`SLACK_BOT_TOKEN`** — `xoxb-…`, from *OAuth & Permissions*.
- **`SLACK_APP_TOKEN`** — `xapp-…`, from *Basic Information → App-Level Tokens*,
  with the `connections:write` scope.
