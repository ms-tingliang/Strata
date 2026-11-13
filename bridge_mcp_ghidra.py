# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "requests>=2,<3",
#     "mcp>=2,<3",
# ]
# ///

import argparse
import contextvars
import functools
import inspect
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import requests

from mcp.server.mcpserver import MCPServer

DEFAULT_GHIDRA_SERVER = "http://127.0.0.1:8080/"
DEFAULT_MCP_PORT = 8081
DEFAULT_TIMEOUT = 180
DEFAULT_PAGE_SIZE = 5000
DEFAULT_MAX_ROWS = 200000
DEFAULT_MAX_INLINE = 200
DEFAULT_DUMP_DIR = "re/ghidra"

# Endpoint census taken from the installed plugin constant pool
# (GhidraMCP.jar -> com/lauriewired/GhidraMCPPlugin.class, createContext paths) and
# confirmed against the live server. This is the whole native surface: nothing here needs
# to be probed over the Ghidra HTTP port directly.
READ_ENDPOINTS = {
    "list_functions": (),
    "methods": ("offset", "limit"),
    "classes": ("offset", "limit"),
    "namespaces": ("offset", "limit"),
    "segments": ("offset", "limit"),
    "imports": ("offset", "limit"),
    "exports": ("offset", "limit"),
    "data": ("offset", "limit"),
    "strings": ("offset", "limit", "filter"),
    "searchFunctions": ("query", "offset", "limit"),
    "get_function_by_address": ("address",),
    "get_current_address": (),
    "get_current_function": (),
    "decompile_function": ("address",),
    "disassemble_function": ("address",),
    "xrefs_to": ("address", "offset", "limit"),
    "xrefs_from": ("address", "offset", "limit"),
    "function_xrefs": ("name", "offset", "limit"),
}

# Endpoints that honour offset/limit. /list_functions ignores them and answers with the
# whole program, so it must not be paged.
PAGINATED_ENDPOINTS = {
    "methods", "classes", "namespaces", "segments", "imports", "exports", "data",
    "strings", "searchFunctions", "xrefs_to", "xrefs_from", "function_xrefs",
}

# Native endpoints that are not a GET-with-params call, so fetch_all and fetch_batch
# cannot drive them. /decompile decompiles by name and takes the bare function name as
# the raw POST body; GET with address=, name= or no params answers "Function not found".
# The bridge drives it through _post_body for decompile_function and decompile_batch.
# Evidence: GhidraMCP.jar constant pool entry "/decompile" and live POST probes on
# 2026-10-16.
POST_BODY_ENDPOINTS = {"decompile": "the bare function name as the raw POST body"}

UPDATE_ENDPOINTS = {
    "renameFunction": ("oldName", "newName"),
    "rename_function_by_address": ("function_address", "new_name"),
    "renameData": ("address", "newName"),
    "renameVariable": ("functionName", "oldName", "newName"),
    "set_decompiler_comment": ("address", "comment"),
    "set_disassembly_comment": ("address", "comment"),
    "set_function_prototype": ("function_address", "prototype"),
    "set_local_variable_type": ("function_address", "variable_name", "new_type"),
}

# Friendly inventory names -> the endpoint that answers them.
INVENTORY_KINDS = {
    "functions": "list_functions",
    "methods": "methods",
    "classes": "classes",
    "namespaces": "namespaces",
    "segments": "segments",
    "imports": "imports",
    "exports": "exports",
    "data": "data",
    "strings": "strings",
}

FAILURE_PATTERN = re.compile("(?i)(failed|failure|fails|error|not found|no function found|unable to|invalid|no such)")
ADDRESS_PATTERN = re.compile(r"^(0x)?[0-9a-fA-F]{4,16}$")

# The plugin answers HTTP 200 with a one-line complaint when its arguments are unusable
# ("Error getting references to address: null", "Function not found", "No function found at
# or containing address ..."). Relaying that as a data row makes a failed read look like a
# result, so a single-line body that reads as a complaint is reported as a failure. The
# pattern is anchored and deliberately narrow: a real row starts with an address or a name,
# and "No functions matching 'x'" is a legitimate empty search, not a complaint.
COMPLAINT_PATTERN = re.compile(
    r"^(Error\b|Request failed\b|Function not found|No function found|Could not\b"
    r"|Unable to\b|Invalid\b|Address is required|Function name is required"
    r"|Search term is required)")

# Replies that do not prove an edit landed. renameData answers "Rename data attempted" for
# an address that holds no label, and the comment endpoints answer "Comment set
# successfully" for an address outside every segment, so neither reply on its own is
# evidence that anything changed.
UNCONFIRMED_REPLY_PATTERN = re.compile(r"(?i)\battempted\b")

# The argument names that carry the address an update edits.
ADDRESS_ARGS = ("address", "function_address")

# The plugin drops the connection on a negative limit and answers an unparsable one with
# its default page, so a page request is clamped before it is sent.
MAX_PAGE_LIMIT = 100000

# Set up logging
log_level = logging.INFO
logger = logging.getLogger(__name__)

# The tool call being served, for the log. Context-local: concurrent calls cannot see
# each other's value, and nothing survives the call.
_CALL = contextvars.ContextVar("bridge_call", default=None)
logger.setLevel(log_level)

mcp = MCPServer("ghidra-mcp", log_level=logging.getLevelName(logger.getEffectiveLevel()))

# No mutable module state: a tool call never reads or writes a process-wide setting, so
# one call cannot observe another's leftovers and a restart is never needed to pick up a
# new value. Configuration is resolved fresh for every call from the environment, which
# main() seeds once from the command line.
ENV_SERVER = "GHIDRA_MCP_SERVER"
ENV_TIMEOUT = "GHIDRA_MCP_TIMEOUT"
ENV_PAGE_SIZE = "GHIDRA_MCP_PAGE_SIZE"
ENV_MAX_ROWS = "GHIDRA_MCP_MAX_ROWS"
ENV_WORKSPACE = "GHIDRA_MCP_WORKSPACE"
ENV_DUMP_DIR = "GHIDRA_MCP_DUMP_DIR"
ENV_LOG_LEVEL = "GHIDRA_MCP_LOG_LEVEL"
ENV_STATELESS = "GHIDRA_MCP_STATELESS"


@dataclass(frozen=True)
class Config:
    """Immutable view of one call's settings."""

    server: str
    timeout: int
    page_size: int
    max_rows: int
    workspace: Path
    dump_dir: str
    log_level: str
    stateless: bool


def _config() -> Config:
    """Build this call's configuration from the environment; nothing is remembered."""
    env = os.environ
    return Config(
        server=env.get(ENV_SERVER) or DEFAULT_GHIDRA_SERVER,
        timeout=max(1, _as_int(env.get(ENV_TIMEOUT), DEFAULT_TIMEOUT)),
        page_size=max(1, _as_int(env.get(ENV_PAGE_SIZE), DEFAULT_PAGE_SIZE)),
        max_rows=max(1, _as_int(env.get(ENV_MAX_ROWS), DEFAULT_MAX_ROWS)),
        workspace=Path(env.get(ENV_WORKSPACE) or Path.cwd()).resolve(),
        dump_dir=env.get(ENV_DUMP_DIR) or DEFAULT_DUMP_DIR,
        log_level=_log_level(env.get(ENV_LOG_LEVEL)),
        stateless=str(env.get(ENV_STATELESS, "1")).lower() not in ("0", "false", "no"),
    )

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _log_level(value) -> str:
    """Accept only a level the logging module knows."""
    text = str(value or "").strip().upper()
    return text if text in LOG_LEVELS else "INFO"


def _configure_logging(level: str) -> None:
    """Put one line per tool call and per native call on stderr."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    try:
        mcp.settings.log_level = level
    except Exception:
        pass


def _brief(value, limit: int = 200) -> str:
    """One-line, length-capped rendering of an argument or reply for the log."""
    text = " ".join(str(value).split())
    if len(text) > limit:
        text = text[:limit] + f"... (+{len(text) - limit} chars)"
    return text


def _call_id() -> str:
    call = _CALL.get()
    return call["id"] if call else "-"


def _call_tool() -> str:
    call = _CALL.get()
    return call["tool"] if call else "startup"


def _begin_call(function, args: tuple, kwargs: dict) -> dict:
    """Open a log context naming the bridged tool and the parameters it received."""
    names = list(inspect.signature(function).parameters)
    arguments = dict(zip(names, args))
    arguments.update(kwargs)
    call = {"tool": function.__name__, "id": uuid.uuid4().hex[:8],
            "started": time.perf_counter()}
    _CALL.set(call)
    logged = ", ".join(f"{key}={_brief(value, 80)}" for key, value in arguments.items())
    logger.info("[%s] TOOL %s(%s)", call["id"], call["tool"], logged)
    return call


def _end_call(call: dict, result) -> None:
    """Close the log context; nothing is carried into the next call."""
    elapsed = int((time.perf_counter() - call["started"]) * 1000)
    count = len(result) if isinstance(result, (list, tuple)) else len(str(result))
    unit = "lines" if isinstance(result, (list, tuple)) else "chars"
    shape = f"{count} {unit if count != 1 else unit[:-1]}"
    logger.info("[%s] TOOL %s done: %s in %dms", call["id"], call["tool"], shape, elapsed)
    _CALL.set(None)


def _request(method: str, endpoint: str, *, params: dict | None = None,
             data: dict | str | None = None) -> tuple[bool, str]:
    """
    The one place a native endpoint is touched. Every call is logged with the bridged
    tool that asked for it, the endpoint, the parameters, the HTTP status and the time
    taken, so a bad rename is diagnosable from the server log alone.
    """
    cfg = _config()
    url = urljoin(cfg.server, endpoint)
    body = data.encode("utf-8") if isinstance(data, str) else data
    target = params if params else (data if data is not None else "")
    call_id = _call_id()
    tool = _call_tool()
    started = time.perf_counter()
    try:
        response = requests.request(method, url, params=params, data=body,
                                    timeout=cfg.timeout)
        response.encoding = "utf-8"
    except Exception as exc:
        logger.warning("[%s] %s %s /%s %s -> %s: %s (%dms)", call_id, tool, method,
                       endpoint, _brief(target), type(exc).__name__, exc,
                       int((time.perf_counter() - started) * 1000))
        return False, f"Request failed: {str(exc)}"
    elapsed = int((time.perf_counter() - started) * 1000)
    text = response.text.strip()
    if not response.ok:
        logger.warning("[%s] %s %s /%s %s -> HTTP %s: %s (%dms)", call_id, tool, method,
                       endpoint, _brief(target), response.status_code, _brief(text), elapsed)
        return False, f"Error {response.status_code}: {text}"
    logger.info("[%s] %s %s /%s %s -> HTTP %s, %d chars (%dms)", call_id, tool, method,
                endpoint, _brief(target), response.status_code, len(text), elapsed)
    return True, text


def safe_get(endpoint: str, params: dict = None) -> list:
    """
    Perform a GET request with optional query parameters. An empty answer and an upstream
    complaint both come back as one explanatory line, never as an empty reply. Anything else
    is relayed line for line, blank lines included, because pseudocode and disassembly use
    them as separators and a caller cannot put them back.
    """
    ok, text = _request("GET", endpoint, params=params or {})
    if not ok:
        return [text]
    rows = _rows(text)
    complaint = _complaint(rows)
    if complaint:
        return [f"Error: {endpoint} rejected the request: {complaint}"]
    if not rows:
        return [f"0 rows: /{endpoint} answered with an empty result"]
    return text.splitlines()


def safe_post(endpoint: str, data: dict | str) -> str:
    """POST to a native endpoint and hand back the reply text."""
    ok, text = _request("POST", endpoint, data=data)
    return text


def _rows(text: str) -> list:
    """Split an upstream body into data rows, dropping blank lines."""
    return [line for line in text.splitlines() if line.strip()]


def _page_params(params: dict) -> str:
    """
    Normalise offset/limit before they reach the plugin, in place. The plugin ignores a
    value it cannot parse and answers its default page, which the bridge would then relay
    as real data, and a negative limit kills the connection outright. So an unparsable
    value is an error here and an out-of-range one is clamped.
    """
    for key in ("offset", "limit"):
        value = params.get(key)
        if value is None or isinstance(value, bool):
            params.pop(key, None)
            continue
        if isinstance(value, int):
            number = value
        elif isinstance(value, str) and value.strip().lstrip("-").isdigit():
            number = int(value.strip())
        elif isinstance(value, float) and value.is_integer():
            number = int(value)
        else:
            return f"Error: {key} must be an integer, got {_brief(value, 40)}"
        params[key] = max(0, number) if key == "offset" else min(max(1, number), MAX_PAGE_LIMIT)
    return ""


def _complaint(row_list: list) -> str:
    """The one-line complaint the plugin answers instead of rows, or "" when these are rows."""
    if len(row_list) == 1 and COMPLAINT_PATTERN.match(row_list[0]):
        return row_list[0]
    return ""


def _get_rows(endpoint: str, params: dict | None = None) -> tuple[bool, list]:
    """
    GET a read endpoint. Returns (ok, rows); rows carry the error when ok is False. An
    upstream complaint counts as a failure even though the plugin answers HTTP 200.
    """
    query = dict(params) if isinstance(params, dict) else {}
    problem = _page_params(query)
    if problem:
        return False, [problem]
    ok, text = _request("GET", endpoint, params=query)
    if not ok:
        return False, [text]
    rows = _rows(text)
    complaint = _complaint(rows)
    if complaint:
        return False, [complaint]
    return True, rows


def _get_text(endpoint: str, params: dict | None = None) -> tuple[bool, str]:
    """GET an endpoint whose body is text to keep as-is (decompile, disassemble)."""
    return _request("GET", endpoint, params=params or {})


def _post_body(endpoint: str, body: str) -> tuple[bool, str]:
    """POST a raw-text body to a read-style endpoint (the plugin's /decompile)."""
    return _request("POST", endpoint, data=body)


def _mapped_ranges() -> list:
    """
    The address ranges the program actually maps, read from /segments. Cached on the log
    context of the call that read it, so it dies with that call and no call can see another
    call's view of the program.
    """
    call = _CALL.get()
    if call is not None:
        cached = call.get("segments")
        if cached is not None:
            return cached
    ok, rows = _get_rows("segments")
    if not ok:
        return []
    ranges = []
    for row in rows:
        match = re.search(r"([0-9a-fA-F]{4,16})\s*-\s*([0-9a-fA-F]{4,16})", row)
        if match:
            ranges.append((int(match.group(1), 16), int(match.group(2), 16)))
    if call is not None and ranges:
        call["segments"] = ranges
    return ranges


def _address_is_mapped(address: str):
    """True or False when the segment map is known, None when it could not be read."""
    value = str(address).strip().lower()
    if value.startswith("0x"):
        value = value[2:]
    if not re.fullmatch(r"[0-9a-f]{1,16}", value):
        return False
    ranges = _mapped_ranges()
    if not ranges:
        return None
    number = int(value, 16)
    return any(low <= number <= high for low, high in ranges)


def _verdict(endpoint: str, args: dict, text: str) -> str:
    """
    OK, FAIL or UNKNOWN for an update. The plugin answers 200 for edits that never landed,
    so a reply is only trusted when it commits to the result and the address it names lies
    inside a mapped segment. UNKNOWN is reported to the agent and not counted as applied.
    """
    if FAILURE_PATTERN.search(text):
        return "FAIL"
    address = next((args[key] for key in ADDRESS_ARGS if args.get(key)), "")
    if address and _address_is_mapped(address) is False:
        return "UNKNOWN"
    if UNCONFIRMED_REPLY_PATTERN.search(text):
        return "UNKNOWN"
    return "OK"


def _post_update(endpoint: str, data: dict) -> tuple[str, str]:
    """
    POST an update endpoint. The plugin answers 200 even when the edit itself failed
    ("Rename failed", "Function not found") and answers a success-looking reply when nothing
    changed, so the verdict is the reply text read against the address it names, and it is
    logged next to the arguments that produced it.
    """
    ok, text = _request("POST", endpoint, data=data)
    if not ok:
        return "FAIL", text
    verdict = _verdict(endpoint, data, text)
    logger.info("[%s] %s VERDICT /%s %s -> %s: %s", _call_id(), _call_tool(), endpoint,
                _brief(data, 160), verdict, _brief(text, 120))
    return verdict, text
def _as_int(value, default: int = 0) -> int:
    """Query values arrive as JSON, so a bad offset or limit must not take the tool down."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _page_args(offset, limit, default_limit: int = 100) -> dict:
    """
    Clamp a page request for the legacy readers. The plugin closes the connection on a
    negative limit, so the caller's numbers cannot be passed through untouched.
    """
    return {"offset": max(0, _as_int(offset, 0)),
            "limit": min(max(1, _as_int(limit, default_limit)), MAX_PAGE_LIMIT)}


def _safe_tool(function):
    """
    A tool must never raise. An unexpected exception becomes a report line the agent can
    read; the traceback stays at debug level so the server log stays readable. Each call
    gets its own log context, so the log names the tool and the parameters it received.
    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        call = _begin_call(function, args, kwargs)
        try:
            result = function(*args, **kwargs)
        except Exception as exc:
            logger.error("[%s] TOOL %s raised %s: %s", call["id"], function.__name__,
                         type(exc).__name__, exc)
            logger.debug("[%s] TOOL %s traceback", call["id"], function.__name__,
                         exc_info=True)
            return [f"Error in {function.__name__}: {type(exc).__name__}: {exc}"]
        _end_call(call, result)
        return result
    return wrapper


def _logged_tool(function):
    """Log a legacy tool call with its parameters without changing its error path."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        call = _begin_call(function, args, kwargs)
        result = function(*args, **kwargs)
        _end_call(call, result)
        return result
    return wrapper


def _endpoint_hint(endpoint: str) -> str:
    """Explain a native endpoint that the paging readers cannot drive directly."""
    if endpoint in POST_BODY_ENDPOINTS:
        return (f" (native POST endpoint taking"
                f" {POST_BODY_ENDPOINTS[endpoint]}; use the decompile_function or"
                f" decompile_batch tool)")
    return ""


def _fetch_all_rows(endpoint: str, params: dict | None = None, page_size: int = 0,
                    max_rows: int = 0) -> tuple[list, str]:
    """
    Page through a paginated read endpoint until a short page comes back, so a whole
    inventory arrives in one MCP call instead of one call per page.
    """
    base = dict(params) if isinstance(params, dict) else {}
    problem = _page_params(base)
    if problem:
        return [], problem
    offset = max(0, _as_int(base.pop("offset", 0), 0))
    requested_limit = max(0, _as_int(base.pop("limit", 0), 0))
    page = max(1, _as_int(page_size, 0) or _config().page_size)
    if requested_limit:
        page = min(page, requested_limit)
    requested_cap = _as_int(max_rows, 0)
    cap = requested_cap if requested_cap > 0 else _config().max_rows

    if endpoint not in PAGINATED_ENDPOINTS:
        ok, rows = _get_rows(endpoint, base)
        if not ok:
            return [], (rows[0] if rows else "request failed")
        return rows[:cap], ""

    rows: list[str] = []
    first_row = ""
    while True:
        ok, page_rows = _get_rows(endpoint, {**base, "offset": offset, "limit": page})
        if not ok:
            return rows, (page_rows[0] if page_rows else "request failed")
        if page_rows and page_rows[0] == first_row:
            # The endpoint ignored the offset, so paging again would only repeat forever.
            return rows, ""
        if not first_row and page_rows:
            first_row = page_rows[0]
        rows.extend(page_rows)
        if len(rows) >= cap:
            # max_rows is a cap, so a page that runs past it is trimmed, not relayed whole.
            return rows[:cap], ""
        if len(page_rows) < page:
            return rows, ""
        offset += page


def _dump_path(name: str) -> Path:
    """
    Resolve a dump file inside the workspace. The installed game and its data are
    read-only (ARTIFACT_SAFETY.md), so nothing may be written outside the workspace.
    """
    candidate = Path(name)
    if candidate.is_absolute():
        raise ValueError("save paths must be relative to the dump directory")
    cfg = _config()
    root = (cfg.workspace / cfg.dump_dir).resolve()
    target = (root / candidate).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError(f"refusing to write outside {root}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _report(rows: list, save: str | None, max_inline: int, auto_name: str) -> list:
    """
    Shape a bulk result: small results come back inline, large ones are written to the
    dump directory and only summarised, so an 8k-row inventory cannot flood the context.
    """
    if not rows:
        logger.info("[%s] %s returned 0 rows", _call_id(), _call_tool())
        return ["0 rows: the endpoint answered with an empty result"]
    if save is None and len(rows) > max_inline:
        save = auto_name
    if save:
        try:
            path = _dump_path(save)
            path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        except (ValueError, OSError) as e:
            return [f"Error: {str(e)}"]
        logger.info("[%s] %s wrote %d rows to %s", _call_id(), _call_tool(), len(rows), path)
        preview = rows[:min(max_inline, 10)]

        return [f"wrote {len(rows)} rows to {path}"] + preview
    if len(rows) <= max_inline:
        return list(rows)
    return list(rows[:max_inline]) + [f"... {len(rows) - max_inline} more rows not shown"]


def _function_name_at(address: str) -> str:
    ok, text = _get_text("get_function_by_address", {"address": address})
    if not ok:
        return ""
    match = re.match(r"Function:\s*(\S+)", text)
    return match.group(1) if match else ""


def _decompile_failed(text: str) -> bool:
    """
    The plugin answers 200 with a one-line complaint ("Function not found", "No function
    found at or containing address ...") when decompilation fails. Real pseudocode always
    carries a brace, so only brace-less complaints count as failures.
    """
    return "{" not in text and bool(FAILURE_PATTERN.search(text))


def _safe_file_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", text).strip("_") or "function"


def _apply_operations(operations: list, dry_run: bool) -> list:
    """
    Run a list of {"tool": ..., "args": ...} updates in one call and report each verdict.
    A plain string in the list is a SKIP line a batch helper worked out for itself, so the
    reason reaches the agent without being dressed up as a tool name. The declared schema
    already rejects an item that is not an object, so no guard for that lives here.
    """
    report: list[str] = []
    applied = 0
    unconfirmed = 0
    for index, operation in enumerate(operations, start=1):
        if isinstance(operation, str):
            report.append(f"[{index}] {operation}")
            continue
        tool = str(operation.get("tool", ""))
        args = operation.get("args")
        if not isinstance(args, dict):
            report.append(f"[{index}] SKIP {tool or 'unknown tool'}: args must be an object")
            continue
        if tool not in UPDATE_ENDPOINTS:
            report.append(f"[{index}] SKIP unknown tool '{tool}' - see bridge_info()")
            continue
        required = UPDATE_ENDPOINTS[tool]
        missing = [key for key in required if not args.get(key)]
        if missing:
            report.append(f"[{index}] SKIP {tool}: missing args {missing}")
            continue
        payload = {key: args[key] for key in required}
        if dry_run:
            report.append(f"[{index}] DRY RUN {tool} {payload}")
            continue
        verdict, text = _post_update(tool, payload)
        if verdict == "OK":
            applied += 1
        elif verdict == "UNKNOWN":
            unconfirmed += 1
        report.append(f"[{index}] {verdict} {tool} {payload} -> {text}")
    tally = f"applied {applied} of {len(operations)} operations"
    if unconfirmed:
        tally += f", {unconfirmed} not confirmed by the plugin"
    report.append(tally + (" (dry run, nothing applied)" if dry_run else ""))
    return report

@mcp.tool()
@_logged_tool
def list_methods(offset: int = 0, limit: int = 100) -> list:
    """
    List all function names in the program with pagination.
    """
    return safe_get("methods", _page_args(offset, limit))

@mcp.tool()
@_logged_tool
def list_classes(offset: int = 0, limit: int = 100) -> list:
    """
    List all namespace/class names in the program with pagination.
    """
    return safe_get("classes", _page_args(offset, limit))

@mcp.tool()
@_logged_tool
def decompile_function(name: str) -> str:
    """
    Decompile a specific function by name and return the decompiled C code.
    """
    return safe_post("decompile", name)

@mcp.tool()
@_logged_tool
def rename_function(old_name: str, new_name: str) -> str:
    """
    Rename a function by its current name to a new user-defined name.
    """
    return safe_post("renameFunction", {"oldName": old_name, "newName": new_name})

@mcp.tool()
@_logged_tool
def rename_data(address: str, new_name: str) -> str:
    """
    Rename a data label at the specified address.
    """
    return safe_post("renameData", {"address": address, "newName": new_name})

@mcp.tool()
@_logged_tool
def list_segments(offset: int = 0, limit: int = 100) -> list:
    """
    List all memory segments in the program with pagination.
    """
    return safe_get("segments", _page_args(offset, limit))

@mcp.tool()
@_logged_tool
def list_imports(offset: int = 0, limit: int = 100) -> list:
    """
    List imported symbols in the program with pagination.
    """
    return safe_get("imports", _page_args(offset, limit))

@mcp.tool()
@_logged_tool
def list_exports(offset: int = 0, limit: int = 100) -> list:
    """
    List exported functions/symbols with pagination.
    """
    return safe_get("exports", _page_args(offset, limit))

@mcp.tool()
@_logged_tool
def list_namespaces(offset: int = 0, limit: int = 100) -> list:
    """
    List all non-global namespaces in the program with pagination.
    """
    return safe_get("namespaces", _page_args(offset, limit))

@mcp.tool()
@_logged_tool
def list_data_items(offset: int = 0, limit: int = 100) -> list:
    """
    List defined data labels and their values with pagination.
    """
    return safe_get("data", _page_args(offset, limit))

@mcp.tool()
@_logged_tool
def search_functions_by_name(query: str, offset: int = 0, limit: int = 100) -> list:
    """
    Search for functions whose name contains the given substring.
    """
    if not query:
        return ["Error: query string is required"]
    return safe_get("searchFunctions", {"query": query, **_page_args(offset, limit)})

@mcp.tool()
@_logged_tool
def rename_variable(function_name: str, old_name: str, new_name: str) -> str:
    """
    Rename a local variable within a function.
    """
    return safe_post("renameVariable", {
        "functionName": function_name,
        "oldName": old_name,
        "newName": new_name
    })

@mcp.tool()
@_logged_tool
def get_function_by_address(address: str) -> str:
    """
    Get a function by its address.
    """
    return "\n".join(safe_get("get_function_by_address", {"address": address}))

@mcp.tool()
@_logged_tool
def get_current_address() -> str:
    """
    Get the address currently selected by the user.
    """
    return "\n".join(safe_get("get_current_address"))

@mcp.tool()
@_logged_tool
def get_current_function() -> str:
    """
    Get the function currently selected by the user.
    """
    return "\n".join(safe_get("get_current_function"))

@mcp.tool()
@_logged_tool
def list_functions() -> list:
    """
    List all functions in the database.
    """
    return safe_get("list_functions")

@mcp.tool()
@_logged_tool
def decompile_function_by_address(address: str) -> str:
    """
    Decompile a function at the given address.
    """
    return "\n".join(safe_get("decompile_function", {"address": address}))

@mcp.tool()
@_logged_tool
def disassemble_function(address: str) -> list:
    """
    Get assembly code (address: instruction; comment) for a function.
    """
    return safe_get("disassemble_function", {"address": address})

@mcp.tool()
@_logged_tool
def set_decompiler_comment(address: str, comment: str) -> str:
    """
    Set a comment for a given address in the function pseudocode.
    """
    return safe_post("set_decompiler_comment", {"address": address, "comment": comment})

@mcp.tool()
@_logged_tool
def set_disassembly_comment(address: str, comment: str) -> str:
    """
    Set a comment for a given address in the function disassembly.
    """
    return safe_post("set_disassembly_comment", {"address": address, "comment": comment})

@mcp.tool()
@_logged_tool
def rename_function_by_address(function_address: str, new_name: str) -> str:
    """
    Rename a function by its address.
    """
    return safe_post("rename_function_by_address", {"function_address": function_address, "new_name": new_name})

@mcp.tool()
@_logged_tool
def set_function_prototype(function_address: str, prototype: str) -> str:
    """
    Set a function's prototype.
    """
    return safe_post("set_function_prototype", {"function_address": function_address, "prototype": prototype})

@mcp.tool()
@_logged_tool
def set_local_variable_type(function_address: str, variable_name: str, new_type: str) -> str:
    """
    Set a local variable's type.
    """
    return safe_post("set_local_variable_type", {"function_address": function_address, "variable_name": variable_name, "new_type": new_type})

@mcp.tool()
@_logged_tool
def get_xrefs_to(address: str, offset: int = 0, limit: int = 100) -> list:
    """
    Get all references to the specified address (xref to).
    
    Args:
        address: Target address in hex format (e.g. "0x1400010a0")
        offset: Pagination offset (default: 0)
        limit: Maximum number of references to return (default: 100)
        
    Returns:
        List of references to the specified address
    """
    return safe_get("xrefs_to", {"address": address, **_page_args(offset, limit)})

@mcp.tool()
@_logged_tool
def get_xrefs_from(address: str, offset: int = 0, limit: int = 100) -> list:
    """
    Get all references from the specified address (xref from).
    
    Args:
        address: Source address in hex format (e.g. "0x1400010a0")
        offset: Pagination offset (default: 0)
        limit: Maximum number of references to return (default: 100)
        
    Returns:
        List of references from the specified address
    """
    return safe_get("xrefs_from", {"address": address, **_page_args(offset, limit)})

@mcp.tool()
@_logged_tool
def get_function_xrefs(name: str, offset: int = 0, limit: int = 100) -> list:
    """
    Get all references to the specified function by name.
    
    Args:
        name: Function name to search for
        offset: Pagination offset (default: 0)
        limit: Maximum number of references to return (default: 100)
        
    Returns:
        List of references to the specified function
    """
    return safe_get("function_xrefs", {"name": name, **_page_args(offset, limit)})

@mcp.tool()
@_logged_tool
def list_strings(offset: int = 0, limit: int = 2000, filter: str = None) -> list:
    """
    List all defined strings in the program with their addresses.
    
    Args:
        offset: Pagination offset (default: 0)
        limit: Maximum number of strings to return (default: 2000)
        filter: Optional filter to match within string content
        
    Returns:
        List of strings with their addresses
    """
    params = _page_args(offset, limit, 2000)
    if filter:
        params["filter"] = filter
    return safe_get("strings", params)


@mcp.tool()
@_safe_tool
def bridge_info() -> list:
    """
    Report the native Ghidra HTTP surface this bridge wraps - endpoint names, exact
    argument names and which endpoints paginate - plus the dump directory and the
    write guard. Read this instead of probing the Ghidra port yourself.
    """
    cfg = _config()
    out = [
        f"ghidra server: {cfg.server}",
        f"workspace: {cfg.workspace}",
        f"dump dir: {(cfg.workspace / cfg.dump_dir).resolve()}",
        f"page size: {cfg.page_size} rows, max rows: {cfg.max_rows}, timeout: {cfg.timeout}s",
        f"transport: {'stateless' if cfg.stateless else 'session'}-http, log level {cfg.log_level}",
        "",
        "read endpoints (GET):",
    ]
    for name in sorted(READ_ENDPOINTS):
        shape = "paginated" if name in PAGINATED_ENDPOINTS else "single response"
        out.append(f"  {name} {' '.join(READ_ENDPOINTS[name])} [{shape}]".rstrip())
    out.append("")
    out.append("update endpoints (POST, argument names are exact):")
    for name in sorted(UPDATE_ENDPOINTS):
        out.append(f"  {name} {' '.join(UPDATE_ENDPOINTS[name])}")
    out.append("")
    out.append("native POST-body endpoints (fetch_all/fetch_batch cannot drive these):")
    for name in sorted(POST_BODY_ENDPOINTS):
        out.append(f"  {name} -> {POST_BODY_ENDPOINTS[name]}; use decompile_function or"
                   f" decompile_batch")
    out.append("")
    out.append("bulk read tools: fetch_all, fetch_batch, list_all_functions, list_all_strings,")
    out.append("  search_functions_all, xrefs_all, decompile_batch, dump_inventory")
    out.append("bulk update tools: update_batch, rename_functions_batch, set_comments_batch,")
    out.append("  set_prototypes_batch")
    out.append("")
    out.append("update verdicts: FAIL means the plugin refused; OK means the reply commits to the")
    out.append("  change and the address lies inside a mapped segment; UNKNOWN means the reply does")
    out.append("  not prove the edit landed, and it is never counted in the applied tally.")
    out.append("read verdicts: an empty answer comes back as one '0 rows' line, a one-line plugin")
    out.append("  complaint is reported as an error rather than relayed as a data row, and offset/")
    out.append("  limit are clamped to 0..100000 because the plugin drops the connection on a")
    out.append("  negative limit and answers an unparsable one with its default page.")
    return out


@mcp.tool()
@_safe_tool
def fetch_all(endpoint: str, params: dict | None = None, page_size: int = 0,
              max_rows: int = 0, save: str | None = None,
              max_inline: int = DEFAULT_MAX_INLINE) -> list:
    """
    Page through one read endpoint and deliver the whole result in a single call.

    Args:
        endpoint: read endpoint name, see bridge_info()
        params: that endpoint's query params (query, filter, address, name)
        page_size: rows per upstream request; 0 uses the server default
        max_rows: stop after this many rows; 0 uses the server default
        save: file name under the dump dir; when omitted the result is written to a file
              only if it is larger than max_inline
        max_inline: rows allowed to come back inline before a file is written instead
    """
    if endpoint not in READ_ENDPOINTS:
        return [f"Error: unknown read endpoint '{endpoint}'{_endpoint_hint(endpoint)}"
                " - see bridge_info()"]
    if params is not None and not isinstance(params, dict):
        return ["Error: params must be an object of query parameters"]
    rows, err = _fetch_all_rows(endpoint, params, page_size, max_rows)
    if err:
        return [err] if not rows else [f"{len(rows)} rows collected before: {err}"]
    return _report(rows, save, max_inline, f"{endpoint}.txt")


@mcp.tool()
@_safe_tool
def fetch_batch(calls: list[dict], max_inline: int = DEFAULT_MAX_INLINE) -> list:
    """
    Run many read calls in one round trip. Each item is
    {"endpoint": "<read endpoint>", "params": {...}}; rows are labelled per call.
    Use this for orientation passes, fetch_all for whole inventories.
    """
    out: list[str] = []
    for index, call in enumerate(calls, start=1):
        # The declared list[dict] schema rejects an item that is not an object, so only the
        # nested params dict - which the schema cannot type - is guarded here.
        endpoint = str(call.get("endpoint", ""))
        params = call.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            out.append(f"[{index}] SKIP {endpoint or 'unknown read endpoint'}: params must be an object")
            continue
        if endpoint not in READ_ENDPOINTS:
            out.append(f"[{index}] SKIP unknown read endpoint '{endpoint}'"
                       f"{_endpoint_hint(endpoint)} - see bridge_info()")
            continue
        ok, rows = _get_rows(endpoint, params)
        label = f"[{index}] GET {endpoint} {params}"
        if not ok:
            out.append(f"{label} -> {rows[0] if rows else 'request failed'}")
            continue
        out.append(f"{label} -> {len(rows)} rows")
        out.extend(rows[:max_inline])
        if len(rows) > max_inline:
            out.append(f"... {len(rows) - max_inline} more rows - use fetch_all to collect them")
    return out


@mcp.tool()
@_safe_tool
def list_all_functions(save: str | None = None,
                       max_inline: int = DEFAULT_MAX_INLINE) -> list:
    """
    Every function in the program as "name at address". This program holds thousands of
    them, so the result lands in the dump directory unless it fits in max_inline.
    """
    ok, rows = _get_rows("list_functions")
    if not ok:
        return rows
    return _report(rows, save, max_inline, "functions.txt")


@mcp.tool()
@_safe_tool
def list_all_strings(filter: str | None = None, save: str | None = None,
                     max_inline: int = DEFAULT_MAX_INLINE) -> list:
    """
    Every defined string as 'address: "text"', optionally substring filtered, collected
    across pages in one call.
    """
    rows, err = _fetch_all_rows("strings", {"filter": filter} if filter else None)
    if err:
        return [err] if not rows else [f"{len(rows)} rows collected before: {err}"]
    return _report(rows, save, max_inline, "strings.txt")


@mcp.tool()
@_safe_tool
def search_functions_all(query: str, save: str | None = None,
                         max_inline: int = DEFAULT_MAX_INLINE) -> list:
    """
    Every function whose name contains query, as "name @ address", collected across
    pages in one call.
    """
    if not query:
        return ["Error: query string is required"]
    rows, err = _fetch_all_rows("searchFunctions", {"query": query})
    if err:
        return [err] if not rows else [f"{len(rows)} rows collected before: {err}"]
    return _report(rows, save, max_inline, f"search_{_safe_file_name(query)}.txt")


@mcp.tool()
@_safe_tool
def xrefs_all(target: str, direction: str = "to", save: str | None = None,
              max_inline: int = DEFAULT_MAX_INLINE) -> list:
    """
    Collect every cross-reference for one target in a single call.

    Args:
        target: 0x-prefixed address for direction "to"/"from", or a function name for "function"
        direction: "to" (default), "from", or "function"
    """
    if direction == "function":
        endpoint, params = "function_xrefs", {"name": target}
    elif direction == "from":
        endpoint, params = "xrefs_from", {"address": target}
    else:
        endpoint, params = "xrefs_to", {"address": target}
    rows, err = _fetch_all_rows(endpoint, params)
    if err:
        return [err] if not rows else [f"{len(rows)} rows collected before: {err}"]
    return _report(rows, save, max_inline, f"xrefs_{direction}_{_safe_file_name(target)}.txt")


@mcp.tool()
@_safe_tool
def decompile_batch(functions: list[str], save_dir: str | None = "decompiled",
                    max_inline: int = 0) -> list:
    """
    Decompile many functions in one call. Each entry is a function name or an address
    (bare hex or 0x-prefixed). With save_dir set the pseudocode is written to
    <dump dir>/<save_dir>/<name>.c and only a per-function summary comes back; with
    save_dir=None the bodies come back inline, each capped at max_inline lines (0 = all).
    """
    target_dir = (save_dir or "").strip("/")
    report: list[str] = []
    succeeded = 0
    for token in functions:
        if ADDRESS_PATTERN.match(token):
            name = _function_name_at(token)
            ok, body = _get_text("decompile_function", {"address": token})
        else:
            name = token
            ok, body = _post_body("decompile", token)
        if not ok or _decompile_failed(body):
            report.append(f"FAIL {token} -> {body}")
            continue
        succeeded += 1
        lines = body.splitlines()
        if not target_dir:
            report.append(f"=== {name or token} ===")
            if max_inline and len(lines) > max_inline:
                report.extend(lines[:max_inline])
                report.append(f"... {len(lines) - max_inline} more lines")
            else:
                report.extend(lines)
            continue
        try:
            path = _dump_path(f"{target_dir}/{_safe_file_name(name or token)}.c")
            path.write_text(body + "\n", encoding="utf-8")
        except (ValueError, OSError) as e:
            report.append(f"FAIL {token} -> {str(e)}")
            continue
        report.append(f"OK {name or token} ({len(lines)} lines) -> {path}")
    report.append(f"decompiled {succeeded} of {len(functions)} functions")
    return report


@mcp.tool()
@_safe_tool
def dump_inventory(kinds: list[str] | None = None, save_dir: str | None = None) -> list:
    """
    Write whole inventories to files under the dump directory in one call and return the
    counts. kinds defaults to functions, strings, methods, classes, namespaces, imports,
    exports, segments; "data" is available but is six figures of rows in this program.
    Grep those files locally instead of paging the list_* tools one page at a time.
    """
    selected = kinds or ["functions", "strings", "methods", "classes", "namespaces",
                         "imports", "exports", "segments"]
    folder = (save_dir or "").strip("/")
    report: list[str] = []
    for kind in selected:
        endpoint = INVENTORY_KINDS.get(kind)
        if not endpoint:
            report.append(f"SKIP unknown kind '{kind}' - choose from {sorted(INVENTORY_KINDS)}")
            continue
        rows, err = _fetch_all_rows(endpoint)
        if err:
            report.append(f"FAIL {kind} -> {err}")
            continue
        name = f"{folder}/{kind}.txt" if folder else f"{kind}.txt"
        try:
            path = _dump_path(name)
            path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        except (ValueError, OSError) as e:
            report.append(f"FAIL {kind} -> {str(e)}")
            continue
        report.append(f"{kind}: {len(rows)} rows -> {path}")
    return report


@mcp.tool()
@_safe_tool
def update_batch(operations: list[dict], dry_run: bool = False) -> list:
    """
    Apply many updates in one call instead of one round trip each. Each item is
    {"tool": "<update endpoint>", "args": {...}} using the exact argument names from
    bridge_info(). The plugin answers 200 even when an edit fails, so every line reports
    the plugin's own verdict; dry_run validates without touching the program.
    """
    return _apply_operations(operations, dry_run)


@mcp.tool()
@_safe_tool
def rename_functions_batch(functions: list[dict], dry_run: bool = False) -> list:
    """
    Rename many functions in one call. Each item is either
      {"address": "004989c0", "new_name": "TAssetMgr_GetBitmap"}
    or
      {"old_name": "FUN_004989c0", "new_name": "TAssetMgr_GetBitmap"}.
    Addresses are bare hex. dry_run reports what would change without touching the program.
    """
    operations: list = []
    for item in functions:
        new_name = item.get("new_name") or item.get("newName") or ""
        address = item.get("address") or item.get("function_address") or ""
        old_name = item.get("old_name") or item.get("oldName") or ""
        if address:
            operations.append({"tool": "rename_function_by_address",
                               "args": {"function_address": address, "new_name": new_name}})
        elif old_name:
            operations.append({"tool": "renameFunction",
                               "args": {"oldName": old_name, "newName": new_name}})
        else:
            operations.append("SKIP give either address or old_name")
    return _apply_operations(operations, dry_run)

@mcp.tool()
@_safe_tool
def set_comments_batch(comments: list[dict], dry_run: bool = False) -> list:
    """
    Annotate many addresses in one call. Each item is
      {"address": "004989c0", "comment": "...", "kind": "decompiler" | "disassembly"}
    kind defaults to "decompiler". Comments should carry the Ghidra address they came from.
    """
    operations: list = []
    for item in comments:
        endpoint = ("set_disassembly_comment"
                    if str(item.get("kind", "decompiler")).lower().startswith("dis")
                    else "set_decompiler_comment")
        operations.append({"tool": endpoint,
                           "args": {"address": item.get("address", ""),
                                    "comment": item.get("comment", "")}})
    return _apply_operations(operations, dry_run)

@mcp.tool()
@_safe_tool
def set_prototypes_batch(prototypes: list[dict], dry_run: bool = False) -> list:
    """
    Set many function prototypes in one call. Each item is
      {"function_address": "00413330", "prototype": "undefined * FUN_00413330(void * this, byte param_1)"}
    Send a bare return type, optionally with "*", then the row's own name, then the parameter
    list. The plugin folds everything before the name into the return type, so a calling-convention
    token in that prefix answers "Can't resolve return type: <that prefix>", and a name that is
    not the row's own renames the row. dry_run validates the argument names without applying.
    """
    operations = [{"tool": "set_function_prototype",
                   "args": {"function_address": item.get("function_address", ""),
                            "prototype": item.get("prototype", "")}} for item in prototypes]
    return _apply_operations(operations, dry_run)


def main():
    parser = argparse.ArgumentParser(description="MCP server for Ghidra")
    parser.add_argument("--ghidra-server", type=str, default=None,
                        help=f"Ghidra server URL, default: {DEFAULT_GHIDRA_SERVER} or $"
                             f"{{{ENV_SERVER}}}")
    parser.add_argument("--mcp-host", type=str, default="127.0.0.1",
                        help="Host to run MCP server on, default: 127.0.0.1")
    parser.add_argument("--mcp-port", type=int,
                        help=f"Port to run MCP server on, default: {DEFAULT_MCP_PORT}")
    parser.add_argument("--transport", type=str, default="streamable-http", choices=["streamable-http"],
                        help="Transport protocol for MCP, default: streamable-http")
    parser.add_argument("--timeout", type=int, default=None,
                        help=f"Upstream request timeout in seconds, default: {DEFAULT_TIMEOUT} or "
                             f"{{{ENV_TIMEOUT}}}")
    parser.add_argument("--workspace", type=str, default=None,
                        help="Directory that bulk dumps must stay inside, default: the current "
                             f"directory or {{{ENV_WORKSPACE}}}")
    parser.add_argument("--dump-dir", type=str, default=None,
                        help=f"Subdirectory of --workspace that bulk results land in, default: "
                             f"{DEFAULT_DUMP_DIR} or {{{ENV_DUMP_DIR}}}")
    parser.add_argument("--page-size", type=int, default=None,
                        help=f"Rows per upstream request when paging, default: {DEFAULT_PAGE_SIZE} "
                             f"or {{{ENV_PAGE_SIZE}}}")
    parser.add_argument("--max-rows", type=int, default=None,
                        help=f"Rows a single bulk call may collect, default: {DEFAULT_MAX_ROWS} "
                             f"or {{{ENV_MAX_ROWS}}}")
    parser.add_argument("--log-level", type=str, default=None, choices=list(LOG_LEVELS),
                        help=f"Log verbosity, default: INFO or {{{ENV_LOG_LEVEL}}}")
    parser.add_argument("--stateful", action="store_true",
                        help="Keep MCP session state between requests. The default is stateless: "
                             "each request is served on a fresh transport, so no call depends on "
                             f"an earlier one. {{{ENV_STATELESS}}}=0 does the same.")
    args = parser.parse_args()

    # Seed the environment once; every later read happens in _config(), so the process
    # keeps no mutable configuration of its own. An unset flag leaves the inherited
    # environment alone, so precedence is flag, then environment, then built-in default.
    for name, value in ((ENV_SERVER, args.ghidra_server),
                        (ENV_TIMEOUT, args.timeout),
                        (ENV_PAGE_SIZE, args.page_size),
                        (ENV_MAX_ROWS, args.max_rows),
                        (ENV_WORKSPACE, args.workspace),
                        (ENV_DUMP_DIR, args.dump_dir),
                        (ENV_LOG_LEVEL, args.log_level),
                        (ENV_STATELESS, "0" if args.stateful else None)):
        if value is not None:
            os.environ[name] = str(value)
    cfg = _config()
    _configure_logging(cfg.log_level)

    # The installed game and its data are read-only; dumps may only ever land inside the
    # workspace, so refuse a dump directory that points out of it.
    if not (cfg.workspace / cfg.dump_dir).resolve().is_relative_to(cfg.workspace):
        parser.error(f"--dump-dir {cfg.dump_dir} escapes the workspace {cfg.workspace}")

    ok, rows = _get_rows("segments")
    if ok:
        logger.info("Ghidra server reachable at %s (%d segments)", cfg.server, len(rows))
    else:
        logger.warning("Ghidra server not reachable at %s: %s", cfg.server, rows[0])

    transport = args.transport or "streamable-http"
    host = args.mcp_host or "127.0.0.1"
    port = args.mcp_port or DEFAULT_MCP_PORT

    try:
        logger.info("Connecting to Ghidra server at %s", cfg.server)
        logger.info("Starting MCP server on http://%s:%d/mcp (%s)", host, port,
                    "stateless" if cfg.stateless else "session")
        logger.info("Bulk dumps go to %s", (cfg.workspace / cfg.dump_dir).resolve())

        mcp.run(transport=transport, host=host, port=port, stateless_http=cfg.stateless)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")

if __name__ == "__main__":
    main()

