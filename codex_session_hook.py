import json
from litellm.integrations.custom_logger import CustomLogger

class CodexSessionHook(CustomLogger):
    from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
    from litellm.types.router import PreRoutingHookResponse
    from litellm.types.utils import (
        AdapterCompletionStreamWrapper,
        CallTypes,
        CallTypesLiteral,
        LLMResponseTypes,
        ModelResponse,
        ModelResponseStream,
        StandardAuditLogPayload,
        StandardBatchLineItemsPayload,
        StandardCallbackDynamicParams,
        StandardLoggingPayload,
    )

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: "DualCache",
        data: dict,
        call_type: CallTypesLiteral,
    ) -> (
        Exception | str | dict | None
    ):  # raise exception if invalid, return a str for the user to receive - if rejected, or return a modified dictionary for passing into litellm
        request = data.get("proxy_server_request") or {}
        raw_headers = request.get("headers") or {}
        headers = {str(k).lower(): v for k, v in raw_headers.items()}

        # Respect explicit LiteLLM headers if already provided
        if headers.get("x-litellm-session-id") or headers.get("x-litellm-trace-id"):
            return data

        # Prefer thread_id (or thread-id), falling back to session_id
        session_id = None
        turn_meta = headers.get("x-codex-turn-metadata")
        if turn_meta:
            try:
                parsed = json.loads(turn_meta)
                session_id = (
                    parsed.get("thread_id")
                    or parsed.get("thread-id")
                    or parsed.get("session_id")
                )
            except (TypeError, ValueError):
                pass
        if not session_id:
            session_id = headers.get("thread-id") or headers.get("session-id")
        if session_id:
            data["litellm_session_id"] = session_id
            # LiteLLM reads the session id for deployment affinity only from request
            # metadata (DeploymentAffinityCheck._get_session_id_from_metadata_dict), and
            # apply_missing_session_id_policy() in litellm_pre_call_utils.py:1117 returns as
            # soon as only litellm_session_id is set, without stamping metadata. With
            # general_settings.missing_session_id: "generate" the metadata instead holds a
            # per-turn uuid marked litellm_session_id_generated, which affinity ignores.
            # Measured on the live proxy (2026-11-13): no session was ever pinned, so every
            # turn picked an engine from scratch and a session whose prompt was parked on
            # one engine paid a cold prefill on its next turn. Stamp the caller thread id
            # where affinity looks for it, and drop the generated marker that hides it.
            stamped = False
            for key in ("metadata", "litellm_metadata"):
                block = data.get(key)
                if isinstance(block, dict):
                    block["session_id"] = session_id
                    block.pop("litellm_session_id_generated", None)
                    stamped = True
            if not stamped:
                data["metadata"] = {"session_id": session_id}
        return data

    async def async_pre_call_deployment_hook(
        self, kwargs: dict[str, object], call_type: CallTypes | None
    ) -> dict | None:
        """
        Allow modifying the request just before it's sent to the deployment.

        Use this instead of 'async_pre_call_hook' when you need to modify the request AFTER a deployment is selected, but BEFORE the request is sent.

        Used in managed_files.py
        """
        print(f'Routing {kwargs["litellm_session_id"]} to {kwargs["api_base"]}.')
        return None

proxy_handler_instance = CodexSessionHook()