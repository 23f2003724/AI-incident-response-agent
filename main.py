"""
TDS GA5 Q11 - AI Incident Response Agent
FastAPI + Gemini API + OTLP Tracing
"""
import os, json, hashlib, secrets, time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import google.generativeai as genai

app = FastAPI(title="Incident Response Agent")

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
STORE: Dict[str, dict] = {}          # runId -> state
HASHES: Dict[str, str] = {}          # "incident:{runId}" | "receipt:{id}" -> sha256

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Fast, cheap models — tried in order. First that responds wins.
# gemini-2.5-flash-lite is fastest; fallbacks in case env var is stale.
_PRIMARY = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
MODELS = list(dict.fromkeys([
    _PRIMARY,
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
]))

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()

def _hex(n=8) -> str:
    while True:
        v = secrets.token_hex(n)
        if v != "0"*(n*2): return v

def span_id()  -> str: return _hex(8)
def trace_id() -> str: return _hex(16)
def act_id()   -> str: return "act_" + _hex(8)
def call_id()  -> str: return "call_" + _hex(8)
def appr_id()  -> str: return "appr_" + _hex(8)

def tp(trace: str, span: str) -> str:
    return f"00-{trace}-{span}-01"

def parse_tp(s: str) -> Optional[tuple]:
    p = (s or "").split("-")
    return tuple(p) if len(p) == 4 else None

def arg_digest(args: dict) -> str:
    return hashlib.sha256(
        json.dumps(args, sort_keys=True, separators=(",",":")).encode()
    ).hexdigest()

def strip_priv(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Gemini call — fast model, no thinking, JSON mode, 12s timeout
# ---------------------------------------------------------------------------

def call_model(incident: dict, catalog: list, policy: dict) -> dict:
    allowed   = incident.get("allowedRootCauses", [])
    title     = incident.get("title", "")
    service   = incident.get("service", "")
    severity  = incident.get("severity", "")
    transcript= incident.get("transcript", "")
    max_diag  = policy.get("maximumDiagnostics", 3)
    effect_names = set(policy.get("effectTools", []))

    diag_tools = [t for t in catalog if t["name"] not in effect_names]

    prompt = (
        "You are an incident response AI. Analyze this incident and respond with ONLY "
        "a JSON object (no markdown fences).\n\n"
        f"Allowed root causes: {json.dumps(allowed)}\n"
        f"Title: {title}\nService: {service}\nSeverity: {severity}\n\n"
        f"Transcript (evidence IDs in [brackets]):\n{transcript[:6000]}\n\n"
        f"Diagnostic tools available:\n{json.dumps(diag_tools, separators=(',',':'))[:2000]}\n\n"
        f"Rules:\n"
        f"- Pick exactly ONE rootCause from the allowed list\n"
        f"- Cite exactly 2-4 evidence IDs that support it\n"
        f"- Pick 1 to {max_diag} diagnostic tools (only directly relevant ones)\n"
        f"- For each diagnostic, supply exact incident-specific arguments\n"
        f"- Each diagnostic must cite at least one of your chosen evidence IDs\n\n"
        "JSON shape:\n"
        '{"rootCause":"...","evidence":["ev_...","ev_..."],'
        '"diagnostics":[{"toolName":"...","arguments":{...},"evidence":["ev_..."]}]}'
    )

    cfg = genai.types.GenerationConfig(
        temperature=0.0,
        max_output_tokens=1024,
    )

    last_err = None
    for model_name in MODELS:
        try:
            m = genai.GenerativeModel(model_name)
            resp = m.generate_content(prompt, generation_config=cfg,
                                      request_options={"timeout": 12})
            text = resp.text.strip()
            # strip fences
            if text.startswith("```"):
                lines = text.splitlines()
                text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
            result = json.loads(text.strip())
            result["_model"] = model_name
            break
        except Exception as e:
            last_err = e
            continue
    else:
        raise RuntimeError(f"All models failed. Last: {last_err}")

    # Sanitise
    if result.get("rootCause") not in allowed:
        result["rootCause"] = allowed[0] if allowed else "unknown"
    ev = result.get("evidence", [])
    if len(ev) < 2: ev += [ev[0]] * (2 - len(ev)) if ev else ["ev_unknown", "ev_unknown"]
    result["evidence"] = list(dict.fromkeys(ev))[:4]   # dedup, max 4
    result["diagnostics"] = result.get("diagnostics", [])[:max_diag]
    return result


# ---------------------------------------------------------------------------
# OTLP helpers
# ---------------------------------------------------------------------------
SK_INTERNAL, SK_SERVER, SK_CLIENT = 1, 2, 3
ST_UNSET, ST_OK, ST_ERROR = 0, 1, 2
NS = 1_000_000   # 1 ms in ns

def attr(k, v):
    if isinstance(v, int):   return {"key": k, "value": {"intValue": v}}
    if isinstance(v, float): return {"key": k, "value": {"doubleValue": v}}
    return {"key": k, "value": {"stringValue": str(v)}}

def build_trace(st: dict) -> dict:
    run_id = st["runId"]
    marker = st.get("publicMarker", "")
    tid    = st["_tid"]
    t0     = st["_t0"]
    common = [attr("ga5.run.id", run_id), attr("ga5.public.marker", marker)]
    spans  = []
    t = t0

    def span(sid, parent, name, kind, attrs, t_start, dur, status=ST_UNSET,
             links=None):
        s = {
            "traceId": tid, "spanId": sid,
            "name": name, "kind": kind,
            "startTimeUnixNano": str(t_start),
            "endTimeUnixNano":   str(t_start + dur),
            "attributes": attrs,
            "status": {"code": status},
        }
        if parent: s["parentSpanId"] = parent
        if links:  s["links"] = links
        spans.append(s)

    # SERVER
    srv_id = st["_srv_sid"]
    span(srv_id, None, "POST /v2/incidents", SK_SERVER, common[:], t, NS*100)
    t += NS

    # INTERNAL invoke_agent
    agt_id = st["_agt_sid"]
    span(agt_id, srv_id, "invoke_agent incident-response", SK_INTERNAL, common[:], t, NS*80)
    t += NS

    # CLIENT chat incident-plan (exactly one)
    chat_id  = st["_chat_sid"]
    model_nm = st.get("_model", _PRIMARY)
    span(chat_id, agt_id, "chat incident-plan", SK_CLIENT,
         common + [attr("gen_ai.operation.name","chat"),
                   attr("gen_ai.request.model", model_nm)],
         t, NS*20)
    t += NS*21

    # INTERNAL execute_tool + CLIENT per logical action
    action_log  = st.get("actionLog", [])
    receipt_log = st.get("receiptLog", [])
    exec_sids   = {}   # actionId -> exec span id (for join links)

    # Collect unique logical actions (first entry per actionId)
    seen = set()
    logical = []
    for d in action_log:
        if d["actionId"] not in seen:
            seen.add(d["actionId"])
            logical.append(d)

    for disp in logical:
        aid      = disp["actionId"]
        tname    = disp["toolName"]
        cid_first= disp["callId"]   # first call_id for gen_ai.tool.call.id

        exec_sid = st.get(f"_esid_{aid}", span_id())
        exec_sids[aid] = exec_sid

        span(exec_sid, agt_id, f"execute_tool {tname}", SK_INTERNAL,
             common + [attr("ga5.action.id", aid),
                       attr("gen_ai.tool.name", tname),
                       attr("gen_ai.tool.call.id", cid_first),
                       attr("gen_ai.operation.name","execute_tool")],
             t, NS*15)
        t += NS

        # CLIENT spans — one per attempt
        attempts = [d for d in action_log if d["actionId"] == aid]
        seen_att = set()
        for d in attempts:
            att = d.get("attempt", 1)
            if att in seen_att: continue
            seen_att.add(att)

            csid = st.get(f"_csid_{aid}_{att}", span_id())
            recs = [r for r in receipt_log
                    if r.get("actionId") == aid and r.get("attempt") == att]
            rec  = recs[0] if recs else None
            http_s   = rec.get("status", 0)   if rec else 0
            rec_id   = rec.get("receiptId","") if rec else ""
            nonce    = rec.get("nonce","")     if rec else ""
            err_type = rec.get("errorType","") if rec else ""
            timeout  = (http_s == 0 and err_type == "timeout")
            is503    = (http_s == 503)

            c_attrs = common + [
                attr("ga5.action.id", aid),
                attr("ga5.attempt", att),
                attr("http.request.method", "POST"),
                attr("http.request.resend_count", att - 1),
            ]
            if rec_id: c_attrs.append(attr("ga5.receipt.id",    rec_id))
            if nonce:  c_attrs.append(attr("ga5.receipt.nonce", nonce))

            if timeout:
                c_attrs.append(attr("error.type", "timeout"))
                st_code = ST_ERROR
            elif is503:
                c_attrs.append(attr("error.type", "503"))
                st_code = ST_ERROR
            elif http_s and http_s != 200:
                st_code = ST_ERROR
            else:
                st_code = ST_UNSET

            if http_s:
                c_attrs.append(attr("http.response.status_code", http_s))

            span(csid, exec_sid, f"POST tool/{tname}", SK_CLIENT,
                 c_attrs, t, NS*5, st_code)
            t += NS*6

    # INTERNAL incident.join — only when >1 unique diagnostic
    diag_aids = [d["actionId"] for d in logical if d.get("phase")=="diagnostic"]
    # deduplicate preserving order
    seen2 = set(); uniq_diag_aids = []
    for a in diag_aids:
        if a not in seen2: seen2.add(a); uniq_diag_aids.append(a)

    if len(uniq_diag_aids) > 1:
        join_sid = st.get("_join_sid", span_id())
        links = [{"traceId": tid, "spanId": exec_sids[a]}
                 for a in uniq_diag_aids if a in exec_sids]
        span(join_sid, agt_id, "incident.join", SK_INTERNAL,
             common[:], t, NS*5, links=links)
        t += NS*6

    # INTERNAL approval_gate — if any approval happened
    appr_recs = [r for r in receipt_log if "approvalId" in r]
    if st.get("_had_approval") or appr_recs:
        gate_sid = st.get("_gate_sid", span_id())
        g_attrs  = common[:]
        for r in appr_recs:
            g_attrs.append(attr("ga5.approval.id",    r["approvalId"]))
            g_attrs.append(attr("ga5.approval.nonce", r.get("nonce","")))
        span(gate_sid, agt_id, "approval_gate", SK_INTERNAL, g_attrs, t, NS*5)

    return {"resourceSpans":[{"resource":{"attributes":[attr("service.name","incident-response-agent")]},
            "scopeSpans":[{"scope":{"name":"ga5.incident.agent","version":"1.0.0"},
                           "spans": spans}]}]}


# ---------------------------------------------------------------------------
# Effect argument builder
# ---------------------------------------------------------------------------

def build_effect_args(tool: dict, st: dict) -> dict:
    incident = st.get("_incident", {})
    props    = tool.get("inputSchema", {}).get("properties", {})
    args = {}
    for k, spec in props.items():
        kl = k.lower()
        if "service" in kl:
            args[k] = incident.get("service", "unknown")
        elif "incident" in kl or (kl == "id"):
            args[k] = incident.get("incidentId", "unknown")
        elif spec.get("type") == "integer":
            args[k] = spec.get("default", 2)
        elif spec.get("type") == "boolean":
            args[k] = spec.get("default", True)
        else:
            args[k] = incident.get(k, spec.get("default", "auto"))
    return args

# ---------------------------------------------------------------------------
# Effect planner — called after all diagnostics complete
# ---------------------------------------------------------------------------

def plan_effect(st: dict):
    pending       = st.get("_pending", {})
    action_log    = st.get("actionLog", [])
    effect_tools  = st.get("_effect_tools", [])
    approval_req  = st.get("_approval_req", [])
    catalog       = st.get("_catalog", [])

    if pending: return
    if any(d.get("phase") == "effect" for d in action_log): return
    if st.get("_pending_appr"): return
    if not [d for d in action_log if d.get("phase") == "diagnostic"]: return
    if not effect_tools: return

    catalog_map = {t["name"]: t for t in catalog}
    chosen = next((e for e in effect_tools if e in catalog_map), None)
    if not chosen: return

    tool   = catalog_map[chosen]
    args   = build_effect_args(tool, st)
    tid    = st["_tid"]
    ts     = st.get("_incoming_ts","")

    if chosen in approval_req:
        # Need approval first
        aid  = act_id()
        apid = appr_id()
        dig  = arg_digest(args)
        st["_pending_appr"][apid] = {
            "approvalId": apid, "actionId": aid,
            "toolName": chosen, "arguments": args,
        }
        st["dispatches"] = []
        st["approvals"]  = [{"approvalId": apid, "actionId": aid,
                              "toolName": chosen, "argumentsDigest": dig}]
        st["_had_approval"] = True
        st["_gate_sid"] = span_id()
        return

    # No approval needed — dispatch immediately
    aid  = act_id()
    cid  = call_id()
    csid = span_id()
    esid = span_id()
    d = {
        "actionId": aid, "callId": cid, "phase": "effect",
        "toolName": chosen, "arguments": args,
        "evidence": st["diagnosis"]["evidence"][:2],
        "attempt": 1,
        "traceparent": tp(tid, csid),
    }
    if ts: d["tracestate"] = ts

    st["dispatches"] = [strip_priv(d)]
    st["approvals"]  = []
    st["actionLog"].append(strip_priv(d))
    st["_pending"][aid] = d
    st[f"_esid_{aid}"] = esid
    st[f"_csid_{aid}_1"] = csid


# ---------------------------------------------------------------------------
# Terminal check
# ---------------------------------------------------------------------------

def maybe_terminal(st: dict):
    if st["status"] in ("completed","failed"): return
    if st.get("_pending"): return
    if st.get("_pending_appr"): return

    has_effect = any(d.get("phase")=="effect" for d in st.get("actionLog",[]))
    if not has_effect and st.get("_effect_tools"):
        return  # plan_effect will handle it next

    st["status"] = "completed" if st.get("chosenEffect") else "failed"
    st["dispatches"] = []
    st["approvals"]  = []
    st["otlp"] = build_trace(st)

# ---------------------------------------------------------------------------
# Response serialiser — never echo sensitive fields
# ---------------------------------------------------------------------------

_REDACT = {"sensitive","accessToken","privateNote","authorization",
           "doNotExport","transcript","prompt","prompts"}

def redact(obj):
    if isinstance(obj, dict):
        return {k: redact(v) for k, v in obj.items() if k not in _REDACT}
    if isinstance(obj, list):
        return [redact(i) for i in obj]
    return obj

def make_response(st: dict) -> dict:
    r = {"runId": st["runId"], "status": st["status"],
         "diagnosis": st["diagnosis"]}
    if st["status"] in ("completed","failed"):
        r["chosenEffect"] = st.get("chosenEffect")
        r["suppressed"]   = st.get("suppressed", [])
        r["actionLog"]    = st.get("actionLog", [])
        r["receiptLog"]   = st.get("receiptLog", [])
        r["dispatches"]   = []
        r["approvals"]    = []
        if "otlp" in st: r["otlp"] = st["otlp"]
    else:
        r["dispatches"] = st.get("dispatches", [])
        r["approvals"]  = st.get("approvals",  [])
    return redact(r)


# ---------------------------------------------------------------------------
# POST /v2/incidents
# ---------------------------------------------------------------------------

@app.post("/v2/incidents")
async def create_incident(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    profile = body.get("profile","")
    if profile != "ga5-incident-agent/v2":
        raise HTTPException(400, f"Unsupported profile: {profile}")

    run_id = body.get("runId","")
    if not run_id:
        raise HTTPException(422, "Missing runId")

    bh = _sha(body)

    # ---- replay / 409 ----
    if run_id in STORE:
        if HASHES.get(f"incident:{run_id}","") != bh:
            return JSONResponse(status_code=409,
                content={"error":"conflict","detail":"runId content changed"})
        return JSONResponse(content=make_response(STORE[run_id]))

    # ---- fresh run ----
    incident = body.get("incident",{})
    catalog  = body.get("toolCatalog",[])
    policy   = body.get("policy",{})
    marker   = body.get("publicMarker","")
    agent_nm = body.get("agentName","incident-response")

    hdrs = dict(request.headers)
    inc_tp = hdrs.get("traceparent","")
    inc_ts = hdrs.get("tracestate","")

    parsed = parse_tp(inc_tp)
    tid = parsed[1] if parsed else trace_id()
    if not parsed: inc_tp = ""; inc_ts = ""

    # ---- model call ----
    try:
        plan = call_model(incident, catalog, policy)
    except Exception as e:
        raise HTTPException(500, f"Model error: {e}")

    root_cause  = plan["rootCause"]
    evidence    = plan["evidence"]
    diagnostics = plan.get("diagnostics",[])
    used_model  = plan.get("_model", _PRIMARY)

    effect_tools = policy.get("effectTools",[])
    approval_req = policy.get("approvalRequiredFor",[])

    # ---- build dispatches ----
    dispatches = []
    action_log = []
    esids = {}; csids = {}

    for diag in diagnostics:
        tname   = diag.get("toolName","")
        args    = diag.get("arguments",{})
        d_ev    = diag.get("evidence", evidence[:1])
        # ensure at least 1 evidence from diagnosis set
        if not any(e in evidence for e in d_ev):
            d_ev = evidence[:1]

        aid  = act_id(); cid = call_id()
        csid = span_id(); esid = span_id()
        esids[aid] = esid; csids[(aid,1)] = csid

        d = {
            "actionId": aid, "callId": cid, "phase": "diagnostic",
            "toolName": tname, "arguments": args, "evidence": d_ev,
            "attempt": 1, "traceparent": tp(tid, csid),
        }
        if inc_ts: d["tracestate"] = inc_ts
        dispatches.append(d)
        action_log.append(strip_priv(d))

    # stable trace span IDs
    srv_sid  = span_id(); agt_sid = span_id()
    chat_sid = span_id(); t0 = time.time_ns()

    state = {
        "runId": run_id, "status": "waiting",
        "publicMarker": marker,
        "diagnosis": {"rootCause": root_cause, "evidence": evidence},
        "dispatches": [strip_priv(d) for d in dispatches],
        "approvals":  [],
        "actionLog":  action_log,
        "receiptLog": [],
        "chosenEffect": None,
        "suppressed":  [],
        # trace
        "_tid": tid, "_t0": t0,
        "_srv_sid": srv_sid, "_agt_sid": agt_sid,
        "_chat_sid": chat_sid, "_model": used_model,
        "_incoming_tp": inc_tp, "_incoming_ts": inc_ts,
        # runtime
        "_pending": {d["actionId"]: d for d in dispatches},
        "_pending_appr": {},
        "_had_approval": False,
        "_effect_tools": effect_tools,
        "_approval_req": approval_req,
        "_catalog": catalog,
        "_policy": policy,
        "_incident": incident,
        "_agent_nm": agent_nm,
    }
    # store span IDs
    for (aid,att), csid in csids.items():
        state[f"_csid_{aid}_{att}"] = csid
    for aid, esid in esids.items():
        state[f"_esid_{aid}"] = esid
    if len(dispatches) > 1:
        state["_join_sid"] = span_id()

    STORE[run_id] = state
    HASHES[f"incident:{run_id}"] = bh
    return JSONResponse(content=make_response(state))


# ---------------------------------------------------------------------------
# POST /v2/incidents/{runId}/receipts
# ---------------------------------------------------------------------------

@app.post("/v2/incidents/{run_id}/receipts")
async def post_receipt(run_id: str, request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    if run_id not in STORE:
        raise HTTPException(404, "Run not found")

    st = STORE[run_id]
    rid = body.get("receiptId","")
    if not rid:
        raise HTTPException(422, "Missing receiptId")

    rh = _sha(body)
    hk = f"receipt:{rid}"

    # ---- 409 / replay ----
    if hk in HASHES:
        if HASHES[hk] != rh:
            return JSONResponse(status_code=409,
                content={"error":"conflict","detail":"receiptId content changed"})
        return JSONResponse(content=make_response(st))

    # ---- terminal replay ----
    if st["status"] in ("completed","failed"):
        HASHES[hk] = rh
        return JSONResponse(content=make_response(st))

    HASHES[hk] = rh

    tid = st["_tid"]
    inc_ts = st.get("_incoming_ts","")

    # ---- process approvals ----
    for appr in body.get("approvals",[]):
        apid      = appr.get("approvalId","")
        decision  = appr.get("decision","")
        nonce     = appr.get("nonce","")
        pending_a = st.get("_pending_appr",{})
        if apid not in pending_a: continue

        st["receiptLog"].append({
            "receiptId": rid, "approvalId": apid,
            "decision": decision, "nonce": nonce,
        })

        if decision == "approved":
            info = pending_a[apid]
            aid  = info["actionId"]
            tname= info["toolName"]
            args = info["arguments"]
            cid  = call_id(); csid = span_id(); esid = span_id()

            d = {
                "actionId": aid, "callId": cid, "phase": "effect",
                "toolName": tname, "arguments": args,
                "evidence": st["diagnosis"]["evidence"][:2],
                "attempt": 1, "traceparent": tp(tid, csid),
                "approvalId": apid, "approvalNonce": nonce,
            }
            if inc_ts: d["tracestate"] = inc_ts

            st["dispatches"] = [strip_priv(d)]
            st["approvals"]  = []
            st["actionLog"].append(strip_priv(d))
            st["_pending"][aid] = d
            st[f"_esid_{aid}"] = esid
            st[f"_csid_{aid}_1"] = csid
            del pending_a[apid]

    # ---- process outcomes ----
    for out in body.get("outcomes",[]):
        aid       = out.get("actionId","")
        cid       = out.get("callId","")
        att       = out.get("attempt",1)
        status    = out.get("status",0)
        r_class   = out.get("resultClass","")
        nonce     = out.get("nonce","")
        err_type  = out.get("errorType","")

        pending = st.get("_pending",{})
        if aid not in pending: continue

        rec = {
            "receiptId": rid, "actionId": aid, "callId": cid,
            "attempt": att, "status": status,
            "resultClass": r_class, "nonce": nonce,
        }
        if err_type: rec["errorType"] = err_type
        st["receiptLog"].append(rec)

        disp = pending[aid]

        if status == 503:
            # one retry
            new_att  = att + 1
            new_cid  = call_id()
            new_csid = span_id()
            nd = {**strip_priv(disp),
                  "callId": new_cid, "attempt": new_att,
                  "traceparent": tp(tid, new_csid)}
            nd.pop("approvalId", None); nd.pop("approvalNonce", None)
            if inc_ts: nd["tracestate"] = inc_ts

            st["dispatches"] = [strip_priv(nd)]
            st["approvals"]  = []
            st["actionLog"].append(strip_priv(nd))
            st["_pending"][aid] = nd
            st[f"_csid_{aid}_{new_att}"] = new_csid

        elif status == 0 and err_type == "timeout":
            st["suppressed"].append(aid)
            del pending[aid]

        elif status == 200:
            del pending[aid]
            if disp.get("phase") == "effect":
                st["chosenEffect"] = disp["toolName"]

        else:
            del pending[aid]

    plan_effect(st)
    maybe_terminal(st)
    STORE[run_id] = st
    return JSONResponse(content=make_response(st))


# ---------------------------------------------------------------------------
# GET /v2/incidents/{runId}
# ---------------------------------------------------------------------------

@app.get("/v2/incidents/{run_id}")
async def get_incident(run_id: str):
    if run_id not in STORE:
        raise HTTPException(404, "Run not found")
    return JSONResponse(content=make_response(STORE[run_id]))


# ---------------------------------------------------------------------------
# Health / debug
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"status": "ok", "service": "incident-response-agent"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/debug/models")
async def list_models():
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY not set"}
    try:
        return {"models": [m.name for m in genai.list_models()
                           if "generateContent" in m.supported_generation_methods]}
    except Exception as e:
        return {"error": str(e)}
