# Telegram Bot (optional)

This bot exposes the hierarchical QA system over Telegram using long polling.

## Setup
1) Create `.env` at repo root (do not commit it):
```
TELEGRAM_BOT_TOKEN=your_bot_token_here
```

2) Install dependency:
```
pip install python-telegram-bot
```

Optional (voice transcription):
```
pip install openai-whisper
```
Ensure `ffmpeg` is installed and on PATH.

Optional (Gmail sending):
```
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

3) Run the bot:
```
python mcp/telegram_server/bot.py
```

## How it works
- When the bot process is running locally, any Telegram message you send triggers a run of the local graph.
- You do not need to run `app/run_cli.py`; the bot calls the same graph directly.
- Progress UX uses a single status message + typing pulses (no token-per-message spam).
- Per-chat execution is serialized to avoid overlapping runs in the same chat.

## Commands
- `/start` basic instructions
- `/status` show model mapping and latest Telegram log file
- `/contacts` list allowlisted recipients
- `/draft` show current draft status for the active chat
- `/email to=<name> body="..." tone="..."` send an email (confirmation required)
- `/new_chat` start a fresh chat session
- `/chats` list chat IDs with last active time and preview
- `/switch <chat_id>` switch to a previous chat
- `/stop` or `/exit` cancel the current flow without clearing memory

Mailer agent:
- All email requests go through the mailer agent.
- While a draft is pending, normal Q&A still works and the draft can be resumed later.

## Voice messages
- Send a voice note to the bot and it will transcribe it locally using Whisper.
- Set `WHISPER_MODEL` in `.env` to control model size (`tiny`, `base`, `small`, `medium`, `large`). Default is `small`.

## Gmail sending
1) Create `secrets/` at repo root.
2) Download OAuth `credentials.json` to `secrets/credentials.json`.
3) Run `python app/gmail_oauth.py` once to create `secrets/token.json`.
4) Add allowlisted contacts in `data/contacts_allowlist.json`.

## Logs
Each Telegram message writes a JSONL log to `runs/` named `telegram_run_*.jsonl`.

## Async knobs
These are optional `.env` flags:
```
ASYNC_TOOLS=1
ASYNC_PERSIST=1
TELEGRAM_PROGRESS=1
ASYNC_TIMEOUT_WEB_SEC=20
ASYNC_TIMEOUT_PDF_SEC=15
ASYNC_TIMEOUT_EMAIL_SEC=30
```
