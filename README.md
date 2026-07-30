# TDS GA5 Q11 — AI Incident Response Agent

A FastAPI service that reads a noisy incident transcript, uses Gemini to find the root cause, runs minimal diagnostic checks, handles approvals, and returns a full OTLP trace.

---

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v2/incidents` | Submit incident, get dispatches |
| POST | `/v2/incidents/{runId}/receipts` | Post tool outcomes / approvals |
| GET  | `/v2/incidents/{runId}` | Read current / final state |

---

## Deploy on Render

### Step 1 — Get a Gemini API key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click **Create API key**
3. Copy the key — you'll need it in Step 4

### Step 2 — Push this folder to GitHub

```bash
cd "path/to/TDS/GA5/11"
git init
git add .
git commit -m "incident agent"
# create a repo on github.com, then:
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

### Step 3 — Create a new Web Service on Render

1. Log in at [render.com](https://render.com)
2. Click **New → Web Service**
3. Connect your GitHub repo
4. Render will auto-detect `render.yaml` — click **Apply**

   If you prefer manual setup:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Step 4 — Set the environment variable

In your Render service dashboard → **Environment** tab:

| Key | Value |
|-----|-------|
| `GEMINI_API_KEY` | `your-key-from-step-1` |
| `GEMINI_MODEL` | `gemini-1.5-flash` *(default, free tier)* |

Click **Save Changes** → Render redeploys automatically.

### Step 5 — Get your public URL

After deploy completes, Render shows a URL like:

```
https://incident-response-agent-xxxx.onrender.com
```

Paste that URL (no trailing slash) into the assignment grader field.

---

## Local development

```bash
# install deps
pip install -r requirements.txt

# set your key
export GEMINI_API_KEY="your-key-here"

# run
uvicorn main:app --reload --port 8000
```

Test with:

```bash
curl http://localhost:8000/health
```

---

## Free-tier cost note

`gemini-1.5-flash` is free within Google AI Studio quota.  
The grader sends ~6 incidents (~80k tokens total). Each Check/Save run
calls the model once per new `runId`; replays never call the model.
Total cost on paid tier is a few cents at most.

---

## Important grader notes

- The base URL must be **HTTPS with no credentials, query, or fragment**
- Responses are JSON ≤ 768 KiB
- Do **not** redirect grader requests
- Sensitive fields (`accessToken`, `privateNote`) are **never** echoed back
