"""Engine-load routing for the four strata engines in testconfig.yaml / config.yaml.

Each deployment is one local engine that runs a single request at a time - hence
`max_parallel_requests: 1` - and it reports its own queue over HTTP. Measured against
the four live endpoints with this repo's `.venv` interpreter (2026-11-13):

    GET <api_base without /v1>/metrics    401 without the deployment key,
                                          200 application/json with it
    live.state                            "idle" | "reading" | "generating"
    live.queued                           requests waiting behind the running one
    conversation_cache                    reused_tokens / prompt_tokens / enabled
    engine.context                        262144 on all four, so no discriminator

Queue depths reach 5 there because several proxies feed the same engines and each
engine holds a 200k-token prefill for minutes. Only the engine knows that, so
`least-busy` and `latency-based-routing` cannot see it: they count only what their own
proxy put on an endpoint, and will happily stack a queue on one engine while three sit
idle.

Routing policy, in this order:

1. A fresh engine beats a queued one. If any candidate has nothing in front of it
   (engine idle, engine queue empty, no local pick already landing there), route
   there even if the session's usual engine is warm and this one is cold. A cold
   prefill costs seconds-to-minutes; a queue behind deep prefills costs minutes.
2. If every candidate already has work in front of it, queue on the best node:
   shortest queue first, then the node this session is pinned to (its conversation
   is parked there), then the highest measured conversation-cache hit rate.
3. A session follows its engine only while that engine is free. When it has to move,
   the LiteLLM session pin is rewritten to the new engine (LiteLLM's own claim
   helper never replaces an existing pin) so the session settles instead of
   re-deciding every turn.

Point 3 needs the request to carry a session id where LiteLLM looks for one. The
affinity check reads request metadata only, and apply_missing_session_id_policy()
in litellm_pre_call_utils.py:1117 returns as soon as a callback has set the top-level
litellm_session_id, without stamping metadata - measured on the live proxy that left
every session unpinned, so every turn picked an engine from scratch. codex_session_hook
stamps the caller thread id into the metadata so the pin exists to be honoured or moved.

Registration is lazy on purpose: `litellm_settings.callbacks` imports this module
before the proxy has built its Router, so the first request registers the strategy.
"""

import asyncio
import os
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

import httpx
from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.router import CustomRoutingStrategyBase

# Tunables. The values suit a LAN engine whose /metrics answers in 20-250 ms
# (measured) and whose requests run for tens of seconds to minutes (measured).
METRICS_PATH = os.environ.get("STRATA_METRICS_PATH", "/metrics")
REFRESH_TTL_SECONDS = float(os.environ.get("STRATA_METRICS_TTL_SECONDS", "2.0"))
SCRAPE_TIMEOUT_SECONDS = float(os.environ.get("STRATA_METRICS_TIMEOUT_SECONDS", "1.5"))
# A reading older than this says nothing about the engine now, so it stops ranking.
MAX_AGE_SECONDS = float(os.environ.get("STRATA_METRICS_MAX_AGE_SECONDS", "15.0"))
# A pick made here is invisible to the engine until the request reaches it, and is
# invisible to a cached reading that was taken before it. Only for that long does the
# pick have to be counted by hand; after it, the engine report covers it. The window
# is REFRESH_TTL_SECONDS, which bounds how stale the newest cached reading can be,
# plus SCRAPE_TIMEOUT_SECONDS, which covers the measured 20-250 ms round trip.
# Measured on the live proxy (2026-11-13): holding a pick for a flat 6 s instead made
# an engine that had already served a turn and gone idle look busy again, and moved a
# session off the one engine that already had its prompt parked.
PICK_VISIBLE_SECONDS = float(
    os.environ.get(
        "STRATA_PICK_VISIBLE_SECONDS", str(REFRESH_TTL_SECONDS + SCRAPE_TIMEOUT_SECONDS)
    )
)
IDLE_STATE = "idle"


class EngineView(NamedTuple):
    """One engine report. `known` is False when the scrape failed or the JSON lacked
    the fields this strategy needs, so a failed scrape is never read as an idle engine.

    A NamedTuple, not a dataclass: the proxy execs callback modules without
    registering them in sys.modules, and on Python 3.10 @dataclass needs
    sys.modules[cls.__module__] and raises AttributeError there."""

    known: bool
    queued: int = 0
    busy: bool = False
    state: str = ""
    reuse: float = 0.0
    scraped_at: float = 0.0
    error: str = ""

    @property
    def ahead(self) -> float:
        """Requests the engine itself reports as being in front of a new arrival:
        its running request plus its queue. -1 when the report is unusable."""
        if not self.known:
            return -1.0
        return float(self.queued + (1 if self.busy else 0))


def _metrics_url(api_base: str, path: str = METRICS_PATH) -> str:
    """`http://abigail:8086/v1` -> `http://abigail:8086/metrics`, the path that
    answered 200 during the probe."""
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base + path


def _number(value: Any) -> Optional[float]:
    """JSON numbers only; `bool` is an `int` subclass and is not a count."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


class StrataMetricsRouter(CustomLogger, CustomRoutingStrategyBase):
    """A `CustomLogger` so the proxy config can import it, and the Router routing
    plugin that picks an engine by reported load."""

    def __init__(self) -> None:
        self._router: Any = None
        self._sync_selection: Any = None
        self._registered = False
        self._client: Optional[httpx.AsyncClient] = None
        self._views: Dict[str, EngineView] = {}
        self._scrapes: Dict[str, "asyncio.Task[EngineView]"] = {}
        # Picks this proxy has made that the engines cannot report yet:
        # (picked-at, deployment id, LiteLLM call id).
        self._picks: List[Tuple[float, str, Optional[str]]] = []
        # Call ids whose request has come back.
        self._finished: Set[str] = set()

    # ------------------------------------------------------------------ lifecycle

    def register(self, router: Any) -> None:
        """Install this strategy on a Router and keep the selection it replaces, so
        the synchronous call path keeps working unchanged."""
        core = getattr(router, "_core", router)
        self._sync_selection = getattr(core, "get_available_deployment", None)
        register_plugin = getattr(router, "register_plugin", None)
        if register_plugin is None:  # LiteLLM naming before the registry split
            register_plugin = router.set_custom_routing_strategy
        register_plugin(self)
        self._router = router
        self._registered = True
        verbose_router_logger.info(
            "[strata-metrics] engine-load routing registered for deployments %s",
            [deployment.get("model_info", {}).get("id") for deployment in router.get_model_list()],
        )

    def _register_if_needed(self) -> None:
        if self._registered:
            return
        try:
            from litellm.proxy.proxy_server import llm_router
        except Exception:  # proxy not up yet - a later request retries
            return
        if llm_router is None:
            return
        try:
            self.register(llm_router)
        except Exception as exc:
            verbose_router_logger.warning("[strata-metrics] registration failed: %s", exc)

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        """First request in: the Router exists by then, so this is where the strategy
        gets installed."""
        self._register_if_needed()
        return data

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Let go of the pick this request was made on. While the answer is out, the
        engine really is holding that slot; once it is back, holding the pick any
        longer is what wrongly calls a free engine busy."""
        self._release_of(kwargs)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """A failed turn frees the slot as well; its retry re-picks and re-books."""
        self._release_of(kwargs)

    async def async_log_stream_event(self, kwargs, response_obj, start_time, end_time):
        """A streaming turn reports through this hook when the last chunk arrives."""
        self._release_of(kwargs)

    @staticmethod
    def _call_id_of(request_kwargs: Optional[Dict]) -> Optional[str]:
        """LiteLLM's id for the request being routed, which its logging hooks carry
        back. A pick without one can only be held for PICK_VISIBLE_SECONDS."""
        call_id = (request_kwargs or {}).get("litellm_call_id")
        return str(call_id) if call_id else None

    def _release_of(self, kwargs: Any) -> None:
        """Release by call id from a logging hook payload. Unknown ids are ignored:
        the pick they belong to has already been swept away."""
        if isinstance(kwargs, dict):
            call_id = kwargs.get("litellm_call_id")
            if call_id:
                self._release(str(call_id))

    # -------------------------------------------------------------- engine reports

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=SCRAPE_TIMEOUT_SECONDS)
        return self._client

    async def _scrape(self, deployment: Dict[str, Any]) -> EngineView:
        """Ask one engine how loaded it is. Never raises: an unreachable engine is a
        dropped reading, not a routing failure."""
        params = deployment.get("litellm_params") or {}
        info = deployment.get("model_info") or {}
        deployment_id = str(info.get("id"))
        api_base = params.get("api_base")
        if not api_base:
            return EngineView(known=False, error="no api_base")
        url = _metrics_url(api_base, str(info.get("strata_metrics_path") or METRICS_PATH))
        api_key = params.get("api_key")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            response = await (await self._http()).get(url, headers=headers, timeout=SCRAPE_TIMEOUT_SECONDS)
            response.raise_for_status()
            body = response.json()
            live = body.get("live") or {}
            cache_state = body.get("conversation_cache") or {}
            queued = _number(live.get("queued"))
            state = str(live.get("state") or "")
            if queued is None and not state:
                raise ValueError(f"no live block in {url}")
            reused = _number(cache_state.get("reused_tokens")) or 0.0
            prompts = _number(cache_state.get("prompt_tokens")) or 0.0
            # Token-level conversation-cache hit rate over the traffic each engine has
            # already served, measured on the live endpoints: 0.92 (abby), 0.71
            # (abigail), 0.26 (ava), 0.25 (alice, whose cache reports enabled=false).
            # Only breaks ties once everything is already queueing.
            reuse = reused / prompts if prompts > 0 else 0.0
            view = EngineView(
                known=True,
                queued=int(queued or 0),
                # Measured vocabulary is idle / reading / generating; anything that is
                # not idle means the single slot is taken.
                busy=bool(state) and state != IDLE_STATE,
                state=state,
                reuse=reuse,
                scraped_at=asyncio.get_running_loop().time(),
            )
        except Exception as exc:
            verbose_router_logger.warning("[strata-metrics] %s: %s", url, exc)
            view = EngineView(known=False, error=str(exc))
        self._views[deployment_id] = view
        return view

    async def _view(self, deployment: Dict[str, Any]) -> EngineView:
        """Cached reading, refreshed at most once per TTL per engine (concurrent picks
        share one in-flight scrape)."""
        deployment_id = self._id_of(deployment)
        now = asyncio.get_running_loop().time()
        cached = self._views.get(deployment_id)
        if cached is not None and now - cached.scraped_at < REFRESH_TTL_SECONDS:
            return cached
        task = self._scrapes.get(deployment_id)
        if task is None:
            task = asyncio.ensure_future(self._scrape(deployment))
            self._scrapes[deployment_id] = task
            task.add_done_callback(lambda _t, key=deployment_id: self._scrapes.pop(key, None))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if cached is not None:
                return cached
            raise

    # ------------------------------------------------------- sessions and pins

    @staticmethod
    def _id_of(deployment: Dict[str, Any]) -> str:
        return str((deployment.get("model_info") or {}).get("id"))

    @staticmethod
    def _session_of(request_kwargs: Optional[Dict]) -> str:
        """The session this pick belongs to. The proxy puts it in metadata; direct
        Router calls carry it as litellm_session_id."""
        kwargs = request_kwargs or {}
        session = kwargs.get("litellm_session_id")
        if session:
            return str(session)
        for key in ("litellm_metadata", "metadata"):
            metadata = kwargs.get(key)
            if isinstance(metadata, dict) and metadata.get("session_id"):
                return str(metadata["session_id"])
        return "-"

    def _affinity_check(self) -> Any:
        """The LiteLLM session-affinity pre-call check, which owns the session to
        deployment pins. None when the config does not enable it."""
        if self._router is None:
            return None
        for callback in getattr(self._router, "optional_callbacks", None) or []:
            if type(callback).__name__ == "DeploymentAffinityCheck":
                return callback
        return None

    def _pin_key(
        self,
        affinity: Any,
        request_kwargs: Optional[Dict],
        model: str,
        candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """The cache key LiteLLM uses for this session's pin.

        The group key is the stable model-map key the affinity check derives from the
        deployment set it was filtering (deployment_affinity_check.py:424 and :447), so
        it is derived here from the same candidate list rather than from the whole
        model list - a different set can hash to a different key and the pin would be
        read, and written, under a key LiteLLM never looks at."""
        get_key = getattr(affinity, "get_session_affinity_cache_key", None)
        get_user_key = getattr(affinity, "get_user_key_from_request_kwargs", None)
        get_session = getattr(affinity, "_get_session_id_from_request_kwargs", None)
        if get_key is None or get_session is None:
            return None
        session_id = get_session(request_kwargs or {})
        if not session_id:
            return None
        group = model
        stable = getattr(affinity, "_get_stable_model_map_key_from_deployments", None)
        if stable is not None:
            group = stable(candidates or self._router.get_model_list()) or model
        user_key = get_user_key(request_kwargs or {}) if get_user_key else None
        return get_key(model_group=group, session_id=str(session_id), user_key=user_key)

    async def _pinned_id(
        self,
        request_kwargs: Optional[Dict],
        model: str,
        candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """Which engine LiteLLM has pinned this session to, if any."""
        affinity = self._affinity_check()
        if affinity is None or not getattr(affinity, "enable_session_id_affinity", False):
            return None
        cache_key = self._pin_key(affinity, request_kwargs, model, candidates)
        if cache_key is None:
            return None
        try:
            stored = await affinity.cache.async_get_cache(key=cache_key)
        except Exception as exc:
            verbose_router_logger.debug("[strata-metrics] pin read failed: %s", exc)
            return None
        if isinstance(stored, dict):
            model_id = stored.get("model_id")
            return str(model_id) if model_id else None
        return str(stored) if isinstance(stored, str) and stored else None

    async def _move_pin(
        self,
        request_kwargs: Optional[Dict],
        model: str,
        new_id: str,
        candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Follow the session to the engine it was just routed to.

        LiteLLM's claim helper keeps an existing pin, so without this the moved
        session would be pinned to the queued engine and re-decide every turn.
        """
        affinity = self._affinity_check()
        if affinity is None:
            return False
        cache_key = self._pin_key(affinity, request_kwargs, model, candidates)
        set_pin = getattr(affinity, "_set_local_pin", None)
        if cache_key is None or set_pin is None:
            verbose_router_logger.warning(
                "[strata-metrics] session pin not moved to %s: no affinity hook available", new_id
            )
            return False
        try:
            set_pin(cache_key, {"model_id": new_id}, getattr(affinity, "ttl_seconds", 3600))
        except Exception as exc:
            verbose_router_logger.warning("[strata-metrics] session pin move to %s failed: %s", new_id, exc)
            return False
        return True

    async def _healthy_without_session(
        self,
        model: str,
        request_kwargs: Optional[Dict],
        messages: Optional[List[Dict[str, str]]],
        input: Optional[Any],
        specific_deployment: Optional[bool],
    ) -> List[Dict[str, Any]]:
        """The healthy list again with the session id hidden, so the affinity check
        cannot narrow it straight back to the queued engine being left. The caller's
        own kwargs are not modified."""
        kwargs = dict(request_kwargs or {})
        for key in ("metadata", "litellm_metadata"):
            metadata = kwargs.get(key)
            if isinstance(metadata, dict) and "session_id" in metadata:
                widened = dict(metadata)
                widened.pop("session_id", None)
                kwargs[key] = widened
        found = await self._router.async_get_healthy_deployments(
            model=model,
            request_kwargs=kwargs,
            messages=messages,
            input=input,
            specific_deployment=specific_deployment,
        )
        return [deployment for deployment in found] if isinstance(found, list) else [found]

    # ----------------------------------------------------------------- selection

    @staticmethod
    def _failed_ids(request_kwargs: Optional[Dict]) -> Set[str]:
        """Deployments this request has already failed on. The Router carries them on
        the retry (`_excluded_deployment_ids`, `_retry_skipped_deployment_ids`) and the
        affinity check refuses to pin back onto one (routing_common_utils.
        deployment_ids_failed_in_request); re-picking one would burn a retry on an
        engine that has already refused this request."""
        failed: Set[str] = set()
        for key in ("_excluded_deployment_ids", "_retry_skipped_deployment_ids"):
            value = (request_kwargs or {}).get(key)
            if isinstance(value, (list, tuple, set, frozenset)):
                failed |= {item for item in map(str, value) if item}
        return failed

    @classmethod
    def _drop_failed(
        cls,
        candidates: List[Dict[str, Any]],
        failed: Set[str],
    ) -> List[Dict[str, Any]]:
        """Drop the engines this request has already failed on. If every candidate
        has failed, keep them: serving from a refused engine still beats failing."""
        if not failed:
            return candidates
        untried = [deployment for deployment in candidates if cls._id_of(deployment) not in failed]
        return untried or candidates

    def _reserve(self, deployment_id: str, call_id: Optional[str]) -> None:
        self._picks.append((asyncio.get_running_loop().time(), deployment_id, call_id))

    def _release(self, call_id: str) -> None:
        self._finished.add(call_id)
        self._sweep()

    def _sweep(self) -> List[Tuple[float, str, Optional[str]]]:
        """Keep only the picks that are still load: the answer has not come back, and
        the reading in hand could not yet have reported the request to the engine."""
        now = asyncio.get_running_loop().time()
        live = [
            pick
            for pick in self._picks
            if now - pick[0] < PICK_VISIBLE_SECONDS and pick[2] not in self._finished
        ]
        if len(live) != len(self._picks):
            self._picks = live
            self._finished = {pick[2] for pick in live if pick[2]}
        return live

    def _reserved(self, deployment_id: str) -> int:
        return sum(1 for pick in self._sweep() if pick[1] == deployment_id)

    def _ahead_of_arrival(self, deployment: Dict[str, Any]) -> float:
        """Requests a new arrival would wait behind, engine report plus this proxy's
        own un-visible picks. -1 marks an engine this strategy cannot read."""
        deployment_id = self._id_of(deployment)
        view = self._views.get(deployment_id)
        if view is None or not view.known or asyncio.get_running_loop().time() - view.scraped_at > MAX_AGE_SECONDS:
            return -1.0
        return view.ahead + self._reserved(deployment_id)

    def _decide(
        self, candidates: List[Dict[str, Any]], pinned_id: Optional[str]
    ) -> Tuple[Dict[str, Any], List[Tuple[str, float, float]]]:
        """Pick an engine and report what each one cost, for the decision log.

        Anything with nothing in front of it (ahead == 0) wins over anything with a
        queue, so a cold idle engine beats a warm queueing one. Once every candidate
        is queueing, rank by shortest queue, then the session's own pinned engine
        (its conversation is parked there), then the measured cache hit rate, then
        configured order, then id so a tie is reproducible. Unknown readings sort
        last and never win while a readable engine exists."""
        ahead = {self._id_of(deployment): self._ahead_of_arrival(deployment) for deployment in candidates}
        fresh = [deployment for deployment in candidates if ahead[self._id_of(deployment)] == 0.0]
        pool = fresh or candidates

        def rank(deployment: Dict[str, Any]) -> Tuple[float, float, float, float, str]:
            deployment_id = self._id_of(deployment)
            view = self._views.get(deployment_id) or EngineView(known=False)
            queue = ahead[deployment_id]
            order = _number((deployment.get("litellm_params") or {}).get("order"))
            if fresh:
                return (0.0, 0.0 if deployment_id == pinned_id else 1.0, -view.reuse, order if order is not None else 0.0, deployment_id)
            return (
                queue if queue >= 0 else float("inf"),
                0.0 if deployment_id == pinned_id else 1.0,
                -view.reuse,
                order if order is not None else 0.0,
                deployment_id,
            )

        ranked = sorted(pool, key=rank)
        report = [(self._id_of(deployment), ahead[self._id_of(deployment)], (self._views.get(self._id_of(deployment)) or EngineView(known=False)).reuse) for deployment in candidates]
        return ranked[0], report

    async def async_get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Any] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        candidates = await self._router.async_get_healthy_deployments(
            model=model,
            request_kwargs=dict(request_kwargs or {}),
            messages=messages,
            input=input,
            specific_deployment=specific_deployment,
        )
        if isinstance(candidates, dict):  # a specific deployment was requested
            return candidates
        if not candidates:
            return None

        failed = self._failed_ids(request_kwargs)
        candidates = self._drop_failed(candidates, failed)

        pinned_id = await self._pinned_id(request_kwargs, model, candidates)
        await asyncio.gather(*(self._view(deployment) for deployment in candidates))

        if len(candidates) == 1 and pinned_id is not None:
            # The affinity check narrowed the pool to this session's engine. Keep that
            # only while the engine has nothing in front of a new arrival.
            if self._ahead_of_arrival(candidates[0]) > 0.0:
                widened = self._drop_failed(
                    await self._healthy_without_session(model, request_kwargs, messages, input, specific_deployment),
                    failed,
                )
                if len(widened) > 1:
                    await asyncio.gather(*(self._view(deployment) for deployment in widened))
                    candidates = widened

        chosen, report = self._decide(candidates, pinned_id)
        chosen_id = self._id_of(chosen)
        self._reserve(chosen_id, self._call_id_of(request_kwargs))
        if pinned_id is not None and chosen_id != pinned_id:
            moved = await self._move_pin(request_kwargs, model, chosen_id, candidates)
            verbose_router_logger.warning(
                "[strata-metrics] model=%s session=%s moved %s -> %s (queue ahead of the pinned engine); pin %s; %s",
                model,
                self._session_of(request_kwargs),
                pinned_id,
                chosen_id,
                "moved" if moved else "NOT moved",
                report,
            )
        else:
            verbose_router_logger.info(
                "[strata-metrics] model=%s session=%s pick=%s pinned=%s ahead=%s",
                model,
                self._session_of(request_kwargs),
                chosen_id,
                pinned_id,
                report,
            )
        return chosen

    def get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Any] = None,
        specific_deployment: Optional[bool] = None,
        request_kwargs: Optional[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """Synchronous call path: no event loop to scrape with, so hand the call to the
        selection this strategy replaced. The proxy only uses the async path."""
        if self._sync_selection is None:
            verbose_router_logger.warning("[strata-metrics] no synchronous selection; model=%s", model)
            return None
        return self._sync_selection(
            model=model,
            messages=messages,
            input=input,
            specific_deployment=specific_deployment,
            request_kwargs=request_kwargs,
        )


# The handler the proxy config imports by dotted path; registration happens on the
# first request, once the proxy has a Router.
proxy_handler_instance = StrataMetricsRouter()
proxy_handler_instance._register_if_needed()
