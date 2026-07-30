"""
TDS GA5 Q11 - AI Incident Response Agent
FastAPI + Gemini API + OTLP Tracing
"""
import os
import json
import hashlib
import secrets
import time
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import google.generativeai as genai

app = FastAPI(title="Incident Response Agent")

# ---------------------------------------------------------------------------
# In-memory store  {runId -> state_dict}
# ---------------------------------------------------------------------------
STORE: Dict[str, dict] = {}
# Content hashes for 409 detection  {runId -> hash, receiptId -> hash}
CONTENT_HASHES: Dict[str, str] = {}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _content_hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def _new_hex(n_bytes: int = 8) -> str:
    """Return n_bytes of nonzero random lowercase hex."""
    while True:
        val = secrets.token_hex(n_bytes)
        if val != "0" * (n_bytes * 2):
            return val


def _new_span_id() -> str:
    return _new_hex(8)


def _new_trace_id() -> str:
    return _new_hex(16)


def _make_traceparent(trace_id: str, span_id: str) -> str:
    return f"00-{trace_id}-{span_id}-01"


def _parse_traceparent(tp: str) -> Optional[tuple]:
    """Return (version, trace_id, span_id, flags) or None."""
    if not tp:
        return None
    parts = tp.split("-")
    if len(parts) != 4:
        return None
    return parts[0], parts[1], parts[2], parts[3]


def _arguments_digest(arguments: dict) -> str:
    """SHA-256 over recursively key-sorted compact JSON."""
    sorted_json = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(sorted_json.encode()).hexdigest()


def _new_action_id() -> str:
    return "act_" + _new_hex(8)


def _new_call_id() -> str:
    return "call_" + _new_hex(8)


def _new_approval_id() -> str:
    return "appr_" + _new_hex(8)


# ---------------------------------------------------------------------------
# Gemini planning call
# ---------------------------------------------------------------------------

def _call_gemini(incident: dict, tool_catalog: list, policy: dict) -> dict:
    """
    Ask Gemini to:
      1. Choose rootCause from allowedRootCauses
      2. Cite 2-4 evidence IDs
      3. Choose 1-3 diagnostic tool calls (name + arguments + evidence)
    Returns dict with keys: rootCause, evidence, diagnostics
    """
    model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

    allowed = incident.get("allowedRootCauses", [])
    transcript = incident.get("transcript", "")
    max_diag = policy.get("maximumDiagnostics", 3)
    effect_tools = policy.get("effectTools", [])

    # Only pass non-sensitive info to model
    diagnostic_tools = [
        t for t in tool_catalog
        if t["name"] not in effect_tools
    ]

    prompt = f"""You are an expert incident response AI. Analyze the following incident transcript and:

1. Choose exactly ONE root cause from this list: {json.dumps(allowed)}
2. Cite exactly 2-4 evidence IDs from the transcript (they look like [ev_xxx] at line starts)
3. Choose 1 to {max_diag} diagnostic tool calls from the catalog to confirm the root cause.
   - Only choose tools that are directly relevant
   - Use exact incident-specific arguments
   - Each diagnostic must cite at least one of your chosen evidence IDs

INCIDENT TITLE: {incident.get('title','')}
SERVICE: {incident.get('service','')}
SEVERITY: {incident.get('severity','')}

TRANSCRIPT:
{transcript[:8000]}

AVAILABLE DIAGNOSTIC TOOLS:
{json.dumps(diagnostic_tools, indent=2)[:3000]}

Respond with ONLY valid JSON in this exact shape (no markdown):
{{
  "rootCause": "one value from allowedRootCauses",
  "evidence": ["ev_xxx", "ev_yyy"],
  "diagnostics": [
    {{
      "toolName": "tool_name",
      "arguments": {{"key": "value"}},
      "evidence": ["ev_xxx"]
    }}
  ]
}}"""

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    text = response.text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    result = json.loads(text)

    # Validate rootCause
    if result.get("rootCause") not in allowed:
        result["rootCause"] = allowed[0]

    # Clamp evidence to 2-4
    ev = result.get("evidence", [])
    if len(ev) < 2:
        ev = ev + ["ev_unknown"] * (2 - len(ev))
    result["evidence"] = ev[:4]

    # Clamp diagnostics
    result["diagnostics"] = result.get("diagnostics", [])[:max_diag]
    return result


# ---------------------------------------------------------------------------
# OTLP Trace builder
# ---------------------------------------------------------------------------

SPAN_KIND_INTERNAL = 1
SPAN_KIND_SERVER = 2
SPAN_KIND_CLIENT = 3
OTLP_STATUS_UNSET = 0
OTLP_STATUS_OK = 1
OTLP_STATUS_ERROR = 2


def _ts_ns() -> int:
    return int(time.time_ns())


def _make_attr(key: str, value) -> dict:
    if isinstance(value, str):
        return {"key": key, "value": {"stringValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": value}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": str(value)}}


def build_otlp_trace(state: dict) -> dict:
    """
    Build the required OTLP trace from stored state.

    Structure:
      SERVER   POST /v2/incidents
      └─ INTERNAL invoke_agent incident-response
         ├─ CLIENT   chat incident-plan
         ├─ INTERNAL execute_tool <toolName>  (one per logical action)
         │  └─ CLIENT POST tool/<toolName>    (one per attempt)
         ├─ INTERNAL incident.join            (if fan-out > 1 diagnostic)
         └─ INTERNAL approval_gate           (if approval happened)
    """
    run_id = state["runId"]
    public_marker = state.get("publicMarker", "")
    trace_id = state.get("_trace_id", _new_trace_id())
    base_time = state.get("_trace_start_ns", _ts_ns())
    dt = 1_000_000  # 1ms in ns

    common_attrs = [
        _make_attr("ga5.run.id", run_id),
        _make_attr("ga5.public.marker", public_marker),
    ]

    spans = []
    t = base_time

    # --- SERVER span: POST /v2/incidents ---
    server_span_id = state.get("_server_span_id", _new_span_id())
    server_span = {
        "traceId": trace_id,
        "spanId": server_span_id,
        "name": "POST /v2/incidents",
        "kind": SPAN_KIND_SERVER,
        "startTimeUnixNano": str(t),
        "endTimeUnixNano": str(t + dt * 100),
        "attributes": common_attrs[:],
        "status": {"code": OTLP_STATUS_UNSET},
    }
    # Propagate incoming traceparent if present
    incoming_tp = state.get("_incoming_traceparent")
    incoming_ts = state.get("_incoming_tracestate")
    if incoming_tp:
        parsed = _parse_traceparent(incoming_tp)
        if parsed:
            server_span["traceId"] = parsed[1]
    spans.append(server_span)
    t += dt

    # --- INTERNAL invoke_agent span ---
    agent_span_id = state.get("_agent_span_id", _new_span_id())
    agent_span = {
        "traceId": trace_id,
        "spanId": agent_span_id,
        "parentSpanId": server_span_id,
        "name": "invoke_agent incident-response",
        "kind": SPAN_KIND_INTERNAL,
        "startTimeUnixNano": str(t),
        "endTimeUnixNano": str(t + dt * 90),
        "attributes": common_attrs[:],
        "status": {"code": OTLP_STATUS_UNSET},
    }
    spans.append(agent_span)
    t += dt

    # --- CLIENT chat incident-plan span ---
    chat_span_id = state.get("_chat_span_id", _new_span_id())
    model_name = state.get("_model_name", os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"))
    chat_span = {
        "traceId": trace_id,
        "spanId": chat_span_id,
        "parentSpanId": agent_span_id,
        "name": "chat incident-plan",
        "kind": SPAN_KIND_CLIENT,
        "startTimeUnixNano": str(t),
        "endTimeUnixNano": str(t + dt * 30),
        "attributes": common_attrs + [
            _make_attr("gen_ai.operation.name", "chat"),
            _make_attr("gen_ai.request.model", model_name),
        ],
        "status": {"code": OTLP_STATUS_UNSET},
    }
    spans.append(chat_span)
    t += dt * 31

    # --- INTERNAL execute_tool + CLIENT spans per action ---
    action_log = state.get("actionLog", [])
    receipt_log = state.get("receiptLog", [])
    tool_span_ids = {}  # actionId -> execute_tool span id (for join links)

    for dispatch in action_log:
        action_id = dispatch["actionId"]
        call_id = dispatch["callId"]
        tool_name = dispatch["toolName"]
        attempt = dispatch.get("attempt", 1)

        # Find all receipts for this action
        receipts_for_action = [
            r for r in receipt_log
            if r.get("actionId") == action_id
        ]

        # INTERNAL execute_tool span
        exec_span_id = state.get(f"_exec_span_{action_id}", _new_span_id())
        tool_span_ids[action_id] = exec_span_id

        exec_attrs = common_attrs + [
            _make_attr("ga5.action.id", action_id),
            _make_attr("gen_ai.tool.name", tool_name),
            _make_attr("gen_ai.tool.call.id", call_id),
            _make_attr("gen_ai.operation.name", "execute_tool"),
        ]
        exec_span = {
            "traceId": trace_id,
            "spanId": exec_span_id,
            "parentSpanId": agent_span_id,
            "name": f"execute_tool {tool_name}",
            "kind": SPAN_KIND_INTERNAL,
            "startTimeUnixNano": str(t),
            "endTimeUnixNano": str(t + dt * 20),
            "attributes": exec_attrs,
            "status": {"code": OTLP_STATUS_UNSET},
        }
        spans.append(exec_span)
        t += dt

        # CLIENT POST tool/<toolName> spans per attempt
        # Each attempt is stored in action log separately (retries increment attempt)
        attempts_for_action = [
            d for d in action_log
            if d.get("actionId") == action_id
        ]
        for disp in attempts_for_action:
            att = disp.get("attempt", 1)
            client_span_id = disp.get("_client_span_id", _new_span_id())
            # Find receipt for this specific attempt
            rec = next(
                (r for r in receipts_for_action if r.get("attempt") == att), None
            )
            http_status = rec.get("status", 0) if rec else 0
            receipt_id = rec.get("receiptId", "") if rec else ""
            nonce = rec.get("nonce", "") if rec else ""
            is_timeout = (rec and rec.get("status") == 0 and
                          rec.get("errorType") == "timeout") if rec else False
            is_503 = http_status == 503

            client_attrs = common_attrs + [
                _make_attr("ga5.action.id", action_id),
                _make_attr("ga5.attempt", att),
                _make_attr("http.request.method", "POST"),
                _make_attr("http.request.resend_count", att - 1),
            ]
            if receipt_id:
                client_attrs.append(_make_attr("ga5.receipt.id", receipt_id))
            if nonce:
                client_attrs.append(_make_attr("ga5.receipt.nonce", nonce))

            span_status = {"code": OTLP_STATUS_UNSET}
            if is_timeout:
                client_attrs.append(_make_attr("error.type", "timeout"))
                span_status = {"code": OTLP_STATUS_ERROR}
            elif is_503:
                client_attrs.append(_make_attr("error.type", "503"))
                span_status = {"code": OTLP_STATUS_ERROR}
            elif http_status and http_status != 200:
                span_status = {"code": OTLP_STATUS_ERROR}

            if http_status:
                client_attrs.append(_make_attr("http.response.status_code", http_status))

            client_span = {
                "traceId": trace_id,
                "spanId": client_span_id,
                "parentSpanId": exec_span_id,
                "name": f"POST tool/{tool_name}",
                "kind": SPAN_KIND_CLIENT,
                "startTimeUnixNano": str(t),
                "endTimeUnixNano": str(t + dt * 5),
                "attributes": client_attrs,
                "status": span_status,
            }
            spans.append(client_span)
            t += dt * 6

    # --- INTERNAL incident.join (if >1 diagnostic dispatch) ---
    diag_dispatches = [d for d in action_log if d.get("phase") == "diagnostic"]
    unique_diag_actions = list({d["actionId"]: d for d in diag_dispatches}.values())
    if len(unique_diag_actions) > 1:
        join_span_id = state.get("_join_span_id", _new_span_id())
        links = [
            {"traceId": trace_id, "spanId": tool_span_ids.get(d["actionId"], "")}
            for d in unique_diag_actions
            if tool_span_ids.get(d["actionId"])
        ]
        join_span = {
            "traceId": trace_id,
            "spanId": join_span_id,
            "parentSpanId": agent_span_id,
            "name": "incident.join",
            "kind": SPAN_KIND_INTERNAL,
            "startTimeUnixNano": str(t),
            "endTimeUnixNano": str(t + dt * 5),
            "attributes": common_attrs[:],
            "links": links,
            "status": {"code": OTLP_STATUS_UNSET},
        }
        spans.append(join_span)
        t += dt * 6

    # --- INTERNAL approval_gate (if approval happened) ---
    approvals_in_receipts = [r for r in receipt_log if "approvalId" in r]
    pending_approvals = state.get("_pending_approvals", {})
    all_approvals = list(pending_approvals.values()) + [
        {"approvalId": r["approvalId"]} for r in approvals_in_receipts
    ]

    if all_approvals or state.get("_had_approval"):
        gate_span_id = state.get("_gate_span_id", _new_span_id())
        gate_attrs = common_attrs[:]
        # Add approval metadata from receipt
        for r in approvals_in_receipts:
            gate_attrs.append(_make_attr("ga5.approval.id", r["approvalId"]))
            gate_attrs.append(_make_attr("ga5.approval.nonce", r.get("nonce", "")))
        gate_span = {
            "traceId": trace_id,
            "spanId": gate_span_id,
            "parentSpanId": agent_span_id,
            "name": "approval_gate",
            "kind": SPAN_KIND_INTERNAL,
            "startTimeUnixNano": str(t),
            "endTimeUnixNano": str(t + dt * 5),
            "attributes": gate_attrs,
            "status": {"code": OTLP_STATUS_UNSET},
        }
        spans.append(gate_span)

    return {
        "resourceSpans": [{
            "resource": {
                "attributes": [
                    _make_attr("service.name", "incident-response-agent"),
                ]
            },
            "scopeSpans": [{
                "scope": {"name": "ga5.incident.agent", "version": "1.0.0"},
                "spans": spans,
            }]
        }]
    }


# ---------------------------------------------------------------------------
# POST /v2/incidents
# ---------------------------------------------------------------------------

@app.post("/v2/incidents")
async def create_incident(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Validate profile
    profile = body.get("profile", "")
    if profile != "ga5-incident-agent/v2":
        raise HTTPException(status_code=400, detail=f"Unsupported profile: {profile}")

    run_id = body.get("runId", "")
    if not run_id:
        raise HTTPException(status_code=422, detail="Missing runId")

    body_hash = _content_hash(body)

    # Replay: same runId + same content -> return stored state without model call
    if run_id in STORE:
        stored = STORE[run_id]
        stored_hash = CONTENT_HASHES.get(f"incident:{run_id}", "")
        if stored_hash and stored_hash != body_hash:
            return JSONResponse(status_code=409, content={
                "error": "conflict",
                "detail": "runId exists with different content"
            })
        # Replay: return current stored response
        return JSONResponse(content=_build_response(stored))

    # Extract fields (never send sensitive to model)
    incident = body.get("incident", {})
    tool_catalog = body.get("toolCatalog", [])
    policy = body.get("policy", {})
    public_marker = body.get("publicMarker", "")
    agent_name = body.get("agentName", "incident-response")

    # Parse incoming traceparent / tracestate
    headers = dict(request.headers)
    incoming_tp = headers.get("traceparent", "")
    incoming_ts = headers.get("tracestate", "")

    # Set up trace IDs
    parsed = _parse_traceparent(incoming_tp) if incoming_tp else None
    if parsed:
        trace_id = parsed[1]
    else:
        trace_id = _new_trace_id()
        incoming_tp = ""
        incoming_ts = ""

    model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

    # Call Gemini for planning
    try:
        plan = _call_gemini(incident, tool_catalog, policy)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

    root_cause = plan["rootCause"]
    evidence = plan["evidence"]
    diagnostics = plan.get("diagnostics", [])

    effect_tools = policy.get("effectTools", [])
    approval_required = policy.get("approvalRequiredFor", [])

    # Build dispatches for diagnostic tools
    dispatches = []
    action_log = []
    span_meta = {}  # actionId -> {exec_span_id, client_span_id}

    for diag in diagnostics:
        tool_name = diag.get("toolName", "")
        arguments = diag.get("arguments", {})
        diag_evidence = diag.get("evidence", evidence[:1])

        action_id = _new_action_id()
        call_id = _new_call_id()
        client_span_id = _new_span_id()
        exec_span_id = _new_span_id()

        traceparent = _make_traceparent(trace_id, client_span_id)

        dispatch = {
            "actionId": action_id,
            "callId": call_id,
            "phase": "diagnostic",
            "toolName": tool_name,
            "arguments": arguments,
            "evidence": diag_evidence,
            "attempt": 1,
            "traceparent": traceparent,
        }
        if incoming_ts:
            dispatch["tracestate"] = incoming_ts

        dispatches.append(dispatch)
        action_log.append({**dispatch, "_client_span_id": client_span_id})
        span_meta[action_id] = {
            "exec_span_id": exec_span_id,
            "client_span_id": client_span_id,
        }

    # Stable span IDs for trace
    server_span_id = _new_span_id()
    agent_span_id = _new_span_id()
    chat_span_id = _new_span_id()
    trace_start_ns = _ts_ns()

    # Build span ID registry
    exec_span_ids = {aid: m["exec_span_id"] for aid, m in span_meta.items()}
    for aid, m in span_meta.items():
        exec_span_ids[aid] = m["exec_span_id"]

    # State to persist
    state = {
        "runId": run_id,
        "status": "waiting",
        "publicMarker": public_marker,
        "diagnosis": {"rootCause": root_cause, "evidence": evidence},
        "dispatches": [_strip_internal(d) for d in dispatches],
        "approvals": [],
        "actionLog": [_strip_internal(d) for d in action_log],
        "receiptLog": [],
        "chosenEffect": None,
        "suppressed": [],
        # Internal trace state
        "_trace_id": trace_id,
        "_trace_start_ns": trace_start_ns,
        "_server_span_id": server_span_id,
        "_agent_span_id": agent_span_id,
        "_chat_span_id": chat_span_id,
        "_model_name": model_name,
        "_incoming_traceparent": incoming_tp,
        "_incoming_tracestate": incoming_ts,
        "_pending_actions": {
            d["actionId"]: d for d in dispatches
        },
        "_pending_approvals": {},
        "_had_approval": False,
        "_effect_tools": effect_tools,
        "_approval_required": approval_required,
        "_tool_catalog": tool_catalog,
        "_policy": policy,
        "_plan": plan,
        "_incident": incident,
        "_agent_name": agent_name,
        # Store exec span IDs for trace building
        **{f"_exec_span_{aid}": m["exec_span_id"] for aid, m in span_meta.items()},
        **{f"_client_span_{aid}_1": m["client_span_id"] for aid, m in span_meta.items()},
    }

    STORE[run_id] = state
    CONTENT_HASHES[f"incident:{run_id}"] = body_hash

    return JSONResponse(content=_build_response(state))


def _strip_internal(d: dict) -> dict:
    """Remove _internal keys from a dispatch dict."""
    return {k: v for k, v in d.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# POST /v2/incidents/{runId}/receipts
# ---------------------------------------------------------------------------

@app.post("/v2/incidents/{run_id}/receipts")
async def post_receipt(run_id: str, request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if run_id not in STORE:
        raise HTTPException(status_code=404, detail="Run not found")

    state = STORE[run_id]

    # Completed runs: replay only
    if state["status"] in ("completed", "failed"):
        receipt_id = body.get("receiptId", "")
        receipt_hash = _content_hash(body)
        stored_hash = CONTENT_HASHES.get(f"receipt:{receipt_id}", "")
        if stored_hash and stored_hash != receipt_hash:
            return JSONResponse(status_code=409, content={
                "error": "conflict", "detail": "receiptId exists with different content"
            })
        return JSONResponse(content=_build_response(state))

    receipt_id = body.get("receiptId", "")
    if not receipt_id:
        raise HTTPException(status_code=422, detail="Missing receiptId")

    receipt_hash = _content_hash(body)

    # 409 if same receiptId with changed content
    if receipt_id in CONTENT_HASHES:
        if CONTENT_HASHES[f"receipt:{receipt_id}"] != receipt_hash:
            return JSONResponse(status_code=409, content={
                "error": "conflict", "detail": "receiptId exists with different content"
            })
        # Replay
        return JSONResponse(content=_build_response(state))

    CONTENT_HASHES[f"receipt:{receipt_id}"] = receipt_hash

    # Handle approval receipts
    approvals_in_body = body.get("approvals", [])
    outcomes_in_body = body.get("outcomes", [])

    for appr in approvals_in_body:
        appr_id = appr.get("approvalId", "")
        decision = appr.get("decision", "")
        nonce = appr.get("nonce", "")

        pending = state.get("_pending_approvals", {})
        if appr_id not in pending:
            continue  # ignore unknown approval

        # Record approval receipt
        state["receiptLog"].append({
            "receiptId": receipt_id,
            "approvalId": appr_id,
            "decision": decision,
            "nonce": nonce,
        })

        if decision == "approved":
            # Dispatch the effect
            appr_info = pending[appr_id]
            tool_name = appr_info["toolName"]
            arguments = appr_info["arguments"]
            action_id = appr_info["actionId"]
            call_id = _new_call_id()
            client_span_id = _new_span_id()
            exec_span_id = _new_span_id()
            trace_id = state["_trace_id"]
            incoming_ts = state.get("_incoming_tracestate", "")

            dispatch = {
                "actionId": action_id,
                "callId": call_id,
                "phase": "effect",
                "toolName": tool_name,
                "arguments": arguments,
                "evidence": state["diagnosis"]["evidence"][:2],
                "attempt": 1,
                "traceparent": _make_traceparent(trace_id, client_span_id),
                "approvalId": appr_id,
                "approvalNonce": nonce,
            }
            if incoming_ts:
                dispatch["tracestate"] = incoming_ts

            state["dispatches"] = [_strip_internal(dispatch)]
            state["approvals"] = []
            state["actionLog"].append(_strip_internal(dispatch))
            state["_pending_actions"][action_id] = dispatch
            state[f"_exec_span_{action_id}"] = exec_span_id
            state[f"_client_span_{action_id}_1"] = client_span_id
            del pending[appr_id]
            state["_had_approval"] = True

    # Handle tool outcome receipts
    for outcome in outcomes_in_body:
        action_id = outcome.get("actionId", "")
        call_id = outcome.get("callId", "")
        attempt = outcome.get("attempt", 1)
        status = outcome.get("status", 0)
        result_class = outcome.get("resultClass", "")
        nonce = outcome.get("nonce", "")
        error_type = outcome.get("errorType", "")

        # Only accept pending calls
        pending = state.get("_pending_actions", {})
        if action_id not in pending:
            continue

        # Record receipt
        rec = {
            "receiptId": receipt_id,
            "actionId": action_id,
            "callId": call_id,
            "attempt": attempt,
            "status": status,
            "resultClass": result_class,
            "nonce": nonce,
        }
        if error_type:
            rec["errorType"] = error_type
        state["receiptLog"].append(rec)

        # Update CLIENT span with receipt info
        state[f"_receipt_status_{action_id}_{attempt}"] = status
        state[f"_receipt_nonce_{action_id}_{attempt}"] = nonce
        state[f"_receipt_id_{action_id}_{attempt}"] = receipt_id

        dispatch = pending[action_id]

        if status == 503:
            # Retry once
            new_attempt = attempt + 1
            new_call_id = _new_call_id()
            client_span_id = _new_span_id()
            trace_id = state["_trace_id"]
            incoming_ts = state.get("_incoming_tracestate", "")

            new_dispatch = {**_strip_internal(dispatch),
                            "callId": new_call_id,
                            "attempt": new_attempt,
                            "traceparent": _make_traceparent(trace_id, client_span_id),
                            "_client_span_id": client_span_id,
                            }
            if incoming_ts:
                new_dispatch["tracestate"] = incoming_ts
            # Remove approval from retry
            new_dispatch.pop("approvalId", None)
            new_dispatch.pop("approvalNonce", None)

            state["dispatches"] = [_strip_internal(new_dispatch)]
            state["approvals"] = []
            state["actionLog"].append(_strip_internal(new_dispatch))
            state["_pending_actions"][action_id] = new_dispatch
            state[f"_client_span_{action_id}_{new_attempt}"] = client_span_id
            continue

        elif status == 0 and error_type == "timeout":
            # Timeout: suppress this action's dependent effect
            state["suppressed"].append(action_id)
            del pending[action_id]

        elif status == 200:
            # Success
            del pending[action_id]
            phase = dispatch.get("phase", "diagnostic")
            if phase == "effect":
                state["chosenEffect"] = dispatch["toolName"]

        else:
            # Other error: mark failed
            del pending[action_id]

    # Check if all diagnostics are done -> plan effect
    _maybe_plan_effect(state)

    # Check if terminal
    _check_terminal(state)

    STORE[run_id] = state
    return JSONResponse(content=_build_response(state))


# ---------------------------------------------------------------------------
# Effect planning helper
# ---------------------------------------------------------------------------

def _maybe_plan_effect(state: dict):
    """
    After all diagnostics complete (no pending diagnostic actions),
    choose and dispatch exactly one effect.
    """
    pending = state.get("_pending_actions", {})
    action_log = state.get("actionLog", [])
    receipt_log = state.get("receiptLog", [])
    effect_tools = state.get("_effect_tools", [])
    approval_required = state.get("_approval_required", [])
    tool_catalog = state.get("_tool_catalog", [])

    # Still have pending actions?
    if pending:
        return

    # Already have an effect dispatch or chosen?
    effect_dispatched = any(d.get("phase") == "effect" for d in action_log)
    if effect_dispatched:
        return

    # Already waiting for approval?
    if state.get("_pending_approvals"):
        return

    # All diagnostics done (success or suppressed)
    diag_dispatches = [d for d in action_log if d.get("phase") == "diagnostic"]
    if not diag_dispatches:
        return

    # Check if any timeout suppression blocks all - still proceed with effect
    # Choose effect tool from catalog
    if not effect_tools:
        return

    # Find the effect tool in catalog
    plan = state.get("_plan", {})
    chosen_tool_name = None

    # Try to use Gemini plan if it included an effect, else pick first effect tool
    # We pick the first available effect tool that's in the catalog
    catalog_names = {t["name"]: t for t in tool_catalog}
    for et in effect_tools:
        if et in catalog_names:
            chosen_tool_name = et
            break

    if not chosen_tool_name:
        return

    chosen_tool = catalog_names[chosen_tool_name]

    # Build arguments: use incident context intelligently
    # For simplicity, use empty args or defaults from schema
    arguments = _build_effect_arguments(chosen_tool, state)

    trace_id = state["_trace_id"]
    incoming_ts = state.get("_incoming_tracestate", "")

    # Needs approval?
    if chosen_tool_name in approval_required:
        action_id = _new_action_id()
        approval_id = _new_approval_id()
        digest = _arguments_digest(arguments)

        state["_pending_approvals"][approval_id] = {
            "approvalId": approval_id,
            "actionId": action_id,
            "toolName": chosen_tool_name,
            "arguments": arguments,
        }
        state["dispatches"] = []
        state["approvals"] = [{
            "approvalId": approval_id,
            "actionId": action_id,
            "toolName": chosen_tool_name,
            "argumentsDigest": digest,
        }]
        state["_had_approval"] = True
        return

    # No approval needed
    action_id = _new_action_id()
    call_id = _new_call_id()
    client_span_id = _new_span_id()
    exec_span_id = _new_span_id()

    dispatch = {
        "actionId": action_id,
        "callId": call_id,
        "phase": "effect",
        "toolName": chosen_tool_name,
        "arguments": arguments,
        "evidence": state["diagnosis"]["evidence"][:2],
        "attempt": 1,
        "traceparent": _make_traceparent(trace_id, client_span_id),
    }
    if incoming_ts:
        dispatch["tracestate"] = incoming_ts

    state["dispatches"] = [_strip_internal(dispatch)]
    state["approvals"] = []
    state["actionLog"].append(_strip_internal(dispatch))
    state["_pending_actions"][action_id] = dispatch
    state[f"_exec_span_{action_id}"] = exec_span_id
    state[f"_client_span_{action_id}_1"] = client_span_id


def _build_effect_arguments(tool: dict, state: dict) -> dict:
    """Build minimal valid arguments for an effect tool."""
    schema = tool.get("inputSchema", {})
    props = schema.get("properties", {})
    incident = state.get("_incident", {})
    args = {}
    for key, spec in props.items():
        # Try to fill from incident fields
        if "service" in key.lower():
            args[key] = incident.get("service", "unknown-service")
        elif "incident" in key.lower() or "id" in key.lower():
            args[key] = incident.get("incidentId", "unknown")
        elif spec.get("type") == "string":
            args[key] = incident.get(key, spec.get("default", "auto"))
        elif spec.get("type") == "integer":
            args[key] = spec.get("default", 1)
        elif spec.get("type") == "boolean":
            args[key] = spec.get("default", True)
    return args


def _check_terminal(state: dict):
    """If no pending work, mark completed/failed."""
    pending = state.get("_pending_actions", {})
    pending_approvals = state.get("_pending_approvals", {})
    if pending or pending_approvals:
        return
    if state["status"] in ("completed", "failed"):
        return

    # Still waiting for effects?
    effect_dispatched = any(d.get("phase") == "effect" for d in state.get("actionLog", []))
    if not effect_dispatched and state.get("_effect_tools"):
        return  # will plan effect next

    # All done
    state["status"] = "completed" if state.get("chosenEffect") else "failed"
    state["dispatches"] = []
    state["approvals"] = []
    # Build final OTLP
    state["otlp"] = build_otlp_trace(state)


# ---------------------------------------------------------------------------
# GET /v2/incidents/{runId}
# ---------------------------------------------------------------------------

@app.get("/v2/incidents/{run_id}")
async def get_incident(run_id: str):
    if run_id not in STORE:
        raise HTTPException(status_code=404, detail="Run not found")
    return JSONResponse(content=_build_response(STORE[run_id]))


# ---------------------------------------------------------------------------
# Response builder
# ---------------------------------------------------------------------------

def _build_response(state: dict) -> dict:
    """Build the public-facing response from internal state."""
    resp = {
        "runId": state["runId"],
        "status": state["status"],
        "diagnosis": state["diagnosis"],
    }

    if state["status"] in ("completed", "failed"):
        resp["chosenEffect"] = state.get("chosenEffect")
        resp["suppressed"] = state.get("suppressed", [])
        resp["actionLog"] = state.get("actionLog", [])
        resp["receiptLog"] = state.get("receiptLog", [])
        if "otlp" in state:
            resp["otlp"] = state["otlp"]
        # Ensure no dispatches/approvals in terminal
        resp["dispatches"] = []
        resp["approvals"] = []
    else:
        # Waiting: include pending dispatches and approvals
        resp["dispatches"] = state.get("dispatches", [])
        resp["approvals"] = state.get("approvals", [])

    # Sanitize: never echo sensitive fields
    return _redact_response(resp)


def _redact_response(obj):
    """Recursively remove any sensitive keys."""
    SENSITIVE_KEYS = {
        "sensitive", "accessToken", "privateNote", "authorization",
        "doNotExport", "transcript", "prompt", "prompts",
    }
    if isinstance(obj, dict):
        return {
            k: _redact_response(v)
            for k, v in obj.items()
            if k not in SENSITIVE_KEYS
        }
    if isinstance(obj, list):
        return [_redact_response(i) for i in obj]
    return obj


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/")
async def health():
    return {"status": "ok", "service": "incident-response-agent"}


@app.get("/health")
async def healthz():
    return {"status": "ok"}

