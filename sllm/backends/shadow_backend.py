# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
import asyncio
import gc
import os
from dataclasses import fields
from typing import Any, Dict, List, Optional

from vllm.shadow.runtime.arg_utils import ShadowEngineArgs
from vllm.shadow.runtime.engine import AsyncShadow

from sllm.backends.backend_utils import BackendStatus, SllmBackend
from sllm.logger import init_logger

logger = init_logger(__name__)


class ShadowBackend(SllmBackend):
    """Ray actor backend for the vLLM shadow CPU engine (``AsyncShadow``).

    Used on a cold instance during scale-up: receives KVHTS from the hot GPU,
    runs CPU decode, and can send KVSTC to the cold GPU when ready.
    """

    def __init__(
        self, model: str, backend_config: Optional[Dict[str, Any]] = None
    ) -> None:
        if backend_config is None:
            raise ValueError("Backend config is missing")

        self.status: BackendStatus = BackendStatus.UNINITIALIZED
        self.status_lock = asyncio.Lock()
        self.backend_config = backend_config
        self.model_name = model

        shadow_engine_fields = {f.name for f in fields(ShadowEngineArgs)}
        filtered_engine_config = {
            k: v for k, v in backend_config.items() if k in shadow_engine_fields
        }

        load_format = backend_config.get("load_format")
        torch_dtype = backend_config.get("torch_dtype")
        if torch_dtype is not None:
            filtered_engine_config["dtype"] = torch_dtype

        if load_format is not None:
            filtered_engine_config["load_format"] = load_format
            filtered_engine_config["model"] = backend_config.get(
                "pretrained_model_name_or_path", model
            )
        else:
            storage_path = os.getenv(
                "STORAGE_PATH", os.path.expanduser("~/models")
            )
            filtered_engine_config["model"] = os.path.join(
                storage_path, "vllm", model
            )
            filtered_engine_config["load_format"] = "serverless_llm"

        logger.info(
            "Creating shadow engine with config: %s", filtered_engine_config
        )

        self.engine_args = ShadowEngineArgs(**filtered_engine_config)

        self.engine: AsyncShadow | None = None

    async def init_backend(self) -> None:
        async with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                return
            self.engine = AsyncShadow.from_engine_args(self.engine_args)
            self.status = BackendStatus.RUNNING

    async def generate(self, request_data: Dict[str, Any]):
        return {
            "error": (
                "ShadowBackend does not serve chat completions; "
                "decode runs inside the shadow engine after KVHTS recv."
            )
        }

    async def shutdown(self):
        async with self.status_lock:
            if self.status == BackendStatus.DELETING:
                return
            self.status = BackendStatus.DELETING

        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None
        gc.collect()

    async def stop(self) -> None:
        async with self.status_lock:
            if self.status.value >= BackendStatus.STOPPING.value:
                return
            self.status = BackendStatus.STOPPING
        await self.shutdown()

    async def get_current_tokens(self) -> List[List[int]]:
        return []

    async def resume_kv_cache(self, request_datas: List[List[int]]) -> None:
        logger.warning(
            "ShadowBackend.resume_kv_cache is a no-op; use KVHTS migration instead"
        )

    async def encode(self, request_data: Dict[str, Any]):
        return {"error": "ShadowBackend does not support embeddings"}

    async def shadow_migration_recv(
        self, migration_id: int, kvhts_ipc_path: str
    ) -> None:
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                raise RuntimeError("Shadow engine is not running")
        assert self.engine is not None
        await self.engine.shadow_migration_recv(migration_id, kvhts_ipc_path)
        logger.info(
            "KVHTS listener ready migration_id=%s path=%s",
            migration_id,
            kvhts_ipc_path,
        )

    async def shadow_migration_migrate(
        self, migration_id: int, kvstc_ipc_path: str
    ) -> list[str]:
        """Send CPU KV to cold GPU via KVSTC (Phase 4)."""
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                raise RuntimeError("Shadow engine is not running")
        assert self.engine is not None
        request_ids = await self.engine.shadow_migration_migrate(
            migration_id, kvstc_ipc_path
        )
        logger.info(
            "KVSTC migrate done migration_id=%s requests=%s path=%s",
            migration_id,
            request_ids,
            kvstc_ipc_path,
        )
        return request_ids

    async def shadow_migration_completed(self) -> dict[str, list[int]]:
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"completed_kvhts_sessions": []}
        assert self.engine is not None
        return await self.engine.shadow_migration_completed()
