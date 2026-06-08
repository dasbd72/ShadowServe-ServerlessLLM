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
import copy
import json
import time
from typing import Any, Dict, List, Optional

import ray

from sllm.inference_instance import start_instance
from sllm.logger import init_logger
from sllm.routers import RoundRobinRouter
from sllm.routers.roundrobin_router import auto_scaler

from ..utils import InstanceHandle

logger = init_logger(__name__)


class ShadowRouter(RoundRobinRouter):
    def __init__(
        self,
        model_name: str,
        resource_requirements: Dict[str, int],
        backend: str,
        backend_config: Dict,
        router_config: Dict,
        enable_lora: bool = False,
        lora_adapters: Optional[Dict[str, str]] = None,
    ) -> None:
        self.waiting_count = 0
        self.waiting_count_lock = asyncio.Lock()

        self.shadow_num_cpus = router_config.get(
            "shadow_num_cpus",
            resource_requirements.get("num_cpus", 1),
        )
        self.shadow_migration_target = router_config.get("target")
        self.shadow_scale_up_active_sources: set[str] = set()
        self.migration_id_counter = 0
        self.migration_id_lock = asyncio.Lock()
        self.migrated_request_targets: Dict[
            str, tuple[InstanceHandle, asyncio.Event]
        ] = {}
        self.migrated_request_lock = asyncio.Lock()

        super().__init__(
            model_name,
            resource_requirements,
            backend,
            backend_config,
            router_config,
            enable_lora,
            lora_adapters,
        )

        assert (
            self.backend == "vllm"
        ), f"Backend {self.backend} is not supported for shadow router"
        assert (
            not self.enable_lora
        ), "Enable LORA is not supported for shadow router"

    async def inference(self, request_data: dict, action: str):
        async with self.running_lock:
            if not self.running:
                return {"error": "Instance stopped"}

        async with self.request_count_lock:
            self.request_count += 1

        async with self.idle_time_lock:
            self.idle_time = 0

        instance_allocation = self.loop.create_future()
        await self.request_queue.put(instance_allocation)
        async with self.waiting_count_lock:
            self.waiting_count += 1
        logger.info(f"Enqueued {action} request for model {self.model_name}")

        instance_id = await instance_allocation
        async with self.waiting_count_lock:
            self.waiting_count -= 1
        logger.info(f"{request_data}, type: {type(request_data)}")
        async with self.instance_management_lock:
            if instance_id not in self.ready_inference_instances:
                logger.error(f"Instance {instance_id} not found")
                return {"error": "Instance not found"}
            instance = self.ready_inference_instances[instance_id]

        # sanity check
        if self.enable_lora and "lora_adapter_name" in request_data:
            lora_adapter_name = request_data["lora_adapter_name"]
            if lora_adapter_name not in self.loaded_lora_adapters:
                logger.error(f"Lora adapter {lora_adapter_name} not found")
                return {"error": f"Lora adapter {lora_adapter_name} not found"}
            await instance.backend_instance.load_lora_adapter.remote(
                lora_name=lora_adapter_name,
                lora_path=self.loaded_lora_adapters[lora_adapter_name],
            )
        # NOTE: `.remote(request_data)` does not work, don't know why.
        # Looks like a known issue:
        # https://github.com/ray-project/ray/issues/26283#issuecomment-1780691475
        if action == "generate":
            result = await instance.backend_instance.generate.remote(
                request_data=request_data
            )
            if isinstance(result, dict) and "_sllm_migration" in result:
                result = await self._resume_migrated_generate(
                    result, request_data
                )
                # Set to none to avoid adding -1 to the instance
                instance = None
        elif action == "encode":
            result = await instance.backend_instance.encode.remote(
                request_data=request_data
            )
        else:
            result = {"error": "Invalid action"}
        logger.info(f"Finished processing request")
        if instance is not None:
            await instance.add_requests(-1)
        async with self.request_count_lock:
            self.request_count -= 1
        return result

    # === Start of migrating request helper functions ===

    async def _resume_migrated_generate(
        self, hot_result: dict, original_request_data: dict
    ) -> dict:
        migration_meta = hot_result.get("_sllm_migration") or {}
        request_id: Optional[str] = str(
            migration_meta.get("external_request_id")
            or migration_meta.get("request_id")
            or hot_result.get("id")
        )
        if not request_id:
            return self._strip_internal_response_fields(hot_result)
        input_tokens: Optional[List[int]] = migration_meta.get("input_tokens")
        if not input_tokens:
            return self._strip_internal_response_fields(hot_result)
        completion_tokens: Optional[int] = migration_meta.get(
            "completion_tokens"
        )
        if not completion_tokens:
            return self._strip_internal_response_fields(hot_result)

        async with self.migrated_request_lock:
            target = self.migrated_request_targets.get(request_id)
        if target is None:
            logger.error(
                "No cold target found for migrated request %s", request_id
            )
            return self._strip_internal_response_fields(hot_result)

        target_instance, kvstc_done = target
        try:
            async with asyncio.timeout(300):
                await kvstc_done.wait()
        except TimeoutError:
            logger.error(
                "Timed out waiting for KVSTC for request %s", request_id
            )
            return self._strip_internal_response_fields(hot_result)

        if target_instance.backend_instance is None:
            logger.error(
                "Cold target has no backend for migrated request %s", request_id
            )
            return self._strip_internal_response_fields(hot_result)

        cold_request_data = copy.deepcopy(original_request_data)
        cold_request_data["request_id"] = request_id
        cold_request_data["input_tokens"] = list(input_tokens)
        self._adjust_remaining_tokens(cold_request_data, completion_tokens)

        try:
            cold_result = (
                await target_instance.backend_instance.generate.remote(
                    request_data=cold_request_data
                )
            )
        finally:
            await target_instance.add_requests(-1)
            async with self.migrated_request_lock:
                self.migrated_request_targets.pop(request_id, None)

        return self._merge_migrated_generate_results(hot_result, cold_result)

    def _strip_internal_response_fields(self, result: dict) -> dict:
        result = copy.deepcopy(result)
        result.pop("_sllm_migration", None)
        return result

    def _adjust_remaining_tokens(
        self, request_data: dict, completed: int
    ) -> None:
        for key in ("max_tokens", "max_completion_tokens"):
            if key not in request_data or request_data[key] is None:
                continue
            request_data[key] = max(1, int(request_data[key]) - completed)
            break

    def _merge_migrated_generate_results(
        self, hot_result: dict, cold_result: Any
    ) -> dict:
        hot = self._strip_internal_response_fields(hot_result)
        if not isinstance(cold_result, dict) or "error" in cold_result:
            return cold_result if isinstance(cold_result, dict) else hot
        cold = self._strip_internal_response_fields(cold_result)
        hot_choices = hot.get("choices") or []
        cold_choices = cold.get("choices") or []
        for idx, hot_choice in enumerate(hot_choices):
            if idx >= len(cold_choices):
                continue
            cold_choice = cold_choices[idx]
            hot_msg = hot_choice.get("message", {})
            cold_msg = cold_choice.get("message", {})
            hot_msg["content"] = (hot_msg.get("content") or "") + (
                cold_msg.get("content") or ""
            )
            hot_choice["message"] = hot_msg
            hot_choice["finish_reason"] = cold_choice.get("finish_reason")
            hot_choice["logprobs"] = cold_choice.get("logprobs")
        hot_usage = hot.get("usage", {})
        cold_usage = cold.get("usage", {})
        if hot_usage or cold_usage:
            assert hot_usage and cold_usage
            prompt_tokens = int(hot_usage.get("prompt_tokens") or 0)
            hot_completion = int(hot_usage.get("completion_tokens") or 0)
            cold_completion = int(cold_usage.get("completion_tokens") or 0)
            hot["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": hot_completion + cold_completion,
                "total_tokens": prompt_tokens
                + hot_completion
                + cold_completion,
            }
        hot_metrics = hot.get("_sllm_metrics")
        cold_metrics = cold.get("_sllm_metrics")
        if hot_metrics or cold_metrics:
            assert hot_metrics and cold_metrics
            assert hot_usage and cold_usage
            hot["_sllm_metrics"] = self._merge_sllm_metrics(
                hot_metrics,
                cold_metrics,
                prompt_tokens,
                hot_completion + cold_completion,
            )
        return hot

    def _merge_sllm_metrics(
        self,
        hot_metrics: Optional[Dict],
        cold_metrics: Optional[Dict],
        prompt_tokens: int,
        completion_tokens: int,
    ) -> dict:
        hot_metrics = hot_metrics or {}
        cold_metrics = cold_metrics or {}
        arrival_ts = hot_metrics.get("_arrival_ts") or cold_metrics.get(
            "_arrival_ts"
        )
        end_ts = cold_metrics.get("_end_ts") or hot_metrics.get("_end_ts", 0.0)
        hot_first = hot_metrics.get("_first_token_ts") or 0.0
        cold_first = cold_metrics.get("_first_token_ts") or 0.0
        first_token_ts = None
        for candidate in (hot_first, cold_first):
            if candidate > 0.0 and (
                first_token_ts is None or candidate < first_token_ts
            ):
                first_token_ts = candidate
        if arrival_ts is None:
            return {
                "ttft_s": 0.0,
                "tpot_s": 0.0,
                "e2e_s": 0.0,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        ttft_s = (first_token_ts - arrival_ts) if first_token_ts else 0.0
        e2e_s = end_ts - arrival_ts if end_ts else 0.0
        if completion_tokens > 1 and first_token_ts is not None:
            tpot_s = (end_ts - first_token_ts) / (completion_tokens - 1)
        else:
            tpot_s = 0.0
        return {
            "ttft_s": round(ttft_s, 6),
            "tpot_s": round(tpot_s, 6),
            "e2e_s": round(e2e_s, 6),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    # === End of migrating request helper functions ===

    async def fine_tuning(self, request_data: dict):
        raise NotImplementedError(
            "Fine-tuning is not supported for shadow router"
        )

    async def _create_ft_instance(self):
        raise NotImplementedError(
            "Creating FT instances is not supported for shadow router"
        )

    async def _start_instance(self, instance_id):
        timing_results = {}
        t_total = time.perf_counter()
        async with self.instance_management_lock:
            if instance_id not in self.starting_inference_instances:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = self.starting_inference_instances[instance_id]
        # Now ask model loading scheduler to load the model
        logger.info(
            f"Allocating resources for model {self.model_name} on instance {instance_id}"
        )
        t_allocate_resource = time.perf_counter()
        startup_node = (
            await self.model_loading_scheduler.allocate_resource.remote(
                self.model_name, instance_id, self.resource_requirements
            )
        )
        timing_results["allocate_resource"] = (
            time.perf_counter() - t_allocate_resource
        )

        shadow_scale_up_source = None
        reserved_shadow_source = False

        async with self.instance_management_lock:
            available_sources = [
                h
                for h in self.ready_inference_instances.values()
                if h.instance_id not in self.shadow_scale_up_active_sources
            ]
            if available_sources:
                shadow_scale_up_source = max(
                    available_sources, key=lambda h: h.concurrency
                )
                self.shadow_scale_up_active_sources.add(
                    shadow_scale_up_source.instance_id
                )
                reserved_shadow_source = True

        try:
            if shadow_scale_up_source:
                results = await self._on_shadow_scale_up(
                    startup_node, shadow_scale_up_source, instance
                )
                timing_results.update(results)
            else:
                results = await self._start_both_backend_instances(
                    startup_node, instance
                )
                timing_results.update(results)
        finally:
            if reserved_shadow_source and shadow_scale_up_source:
                async with self.instance_management_lock:
                    self.shadow_scale_up_active_sources.discard(
                        shadow_scale_up_source.instance_id
                    )

        async with self.instance_management_lock:
            self.ready_inference_instances[instance_id] = instance
            self.starting_inference_instances.pop(instance_id)

        timing_results["total"] = time.perf_counter() - t_total

        # logging timing results
        logger.info(
            f"Started instance {instance_id} for model {self.model_name} "
            f"{json.dumps(timing_results)}"
        )
        return instance_id

    async def _start_both_backend_instances(
        self, startup_node: str, instance: InstanceHandle
    ):
        t_total = time.perf_counter()
        backend_results, shadow_results = await asyncio.gather(
            self._start_backend_instance(startup_node, instance),
            self._start_shadow_instance(startup_node, instance),
        )
        elapsed_total = time.perf_counter() - t_total
        logger.info(
            f"Initialized all backend for instance {instance.instance_id} for "
            f"model {self.model_name} took {elapsed_total} seconds"
        )
        return backend_results | shadow_results

    async def _start_backend_instance(
        self, startup_node: str, instance: InstanceHandle
    ):
        t_start_backend_instance = time.perf_counter()
        instance_id = instance.instance_id
        startup_config = {
            "num_cpus": self.resource_requirements["num_cpus"],
            "num_gpus": self.resource_requirements["num_gpus"],
            "resources": {
                "worker_node": 0.1,
                f"worker_id_{startup_node}": 0.1,
            },
        }
        logger.info(f"Startup config: {startup_config}, {self.backend_config}")

        await start_instance.options(
            resources={
                "worker_node": 0.1,
                f"worker_id_{startup_node}": 0.1,
            }
        ).remote(
            instance_id,
            self.backend,
            self.model_name,
            self.backend_config,
            startup_config,
        )
        elapsed_start_backend_instance = (
            time.perf_counter() - t_start_backend_instance
        )
        t_init_backend = time.perf_counter()
        instance.backend_instance = ray.get_actor(instance_id)
        async with instance.lock:
            instance.ready = True
            instance.node_id = startup_node
        await instance.backend_instance.init_backend.remote()
        elapsed_init_backend = time.perf_counter() - t_init_backend
        logger.info(
            f"Initialized backend for instance {instance_id} for "
            f"model {self.model_name} took {elapsed_init_backend} seconds"
        )
        return {
            "start_backend_instance": elapsed_start_backend_instance,
            "init_backend": elapsed_init_backend,
        }

    async def _start_shadow_instance(
        self, startup_node: str, instance: InstanceHandle
    ):
        t_start_shadow_instance = time.perf_counter()
        instance_id = f"shadow-{instance.instance_id}"
        shadow_startup = {
            "num_cpus": self.shadow_num_cpus,
            "num_gpus": 0,
            "resources": {
                "worker_node": 0.1,
                f"worker_id_{startup_node}": 0.1,
            },
        }
        logger.info(
            f"Shadow startup config: {shadow_startup}, {self.backend_config}"
        )
        await start_instance.options(
            resources={
                "worker_node": 0.1,
                f"worker_id_{startup_node}": 0.1,
            },
        ).remote(
            instance_id,
            "shadow",
            self.model_name,
            self.backend_config,
            shadow_startup,
        )
        elapsed_start_shadow_instance = (
            time.perf_counter() - t_start_shadow_instance
        )
        t_init_shadow = time.perf_counter()
        instance.shadow_backend_instance = ray.get_actor(instance_id)
        await instance.shadow_backend_instance.init_backend.remote()
        elapsed_init_shadow = time.perf_counter() - t_init_shadow
        return {
            "start_shadow_instance": elapsed_start_shadow_instance,
            "init_shadow_backend": elapsed_init_shadow,
        }

    async def _next_migration_id(self) -> int:
        async with self.migration_id_lock:
            self.migration_id_counter += 1
            return self.migration_id_counter

    def _kvhts_ipc_path_for_migration(self, migration_id: int) -> str:
        prefix = self.router_config.get(
            "kvhts_ipc_prefix", "/tmp/vllm-shadow-kvhts"
        ).rstrip("/")
        return f"{prefix}-{migration_id}.sock"

    def _kvstc_ipc_path_for_migration(self, migration_id: int) -> str:
        prefix = self.router_config.get(
            "kvstc_ipc_prefix", "/tmp/vllm-shadow-kvstc"
        ).rstrip("/")
        return f"{prefix}-{migration_id}.sock"

    def _tksth_ipc_path_for_migration(self, migration_id: int) -> str:
        prefix = self.router_config.get(
            "tksth_ipc_prefix", "/tmp/vllm-shadow-tksth"
        ).rstrip("/")
        return f"{prefix}-{migration_id}.sock"

    async def _on_shadow_scale_up(
        self,
        startup_node: str,
        source: InstanceHandle,
        instance: InstanceHandle,
    ) -> None:
        migration_id = await self._next_migration_id()
        logger.info(
            "Shadow scale-up migration_id=%s source=%s instance=%s",
            migration_id,
            source.instance_id,
            instance.instance_id,
        )

        start_backend_instance_task = asyncio.create_task(
            self._start_both_backend_instances(startup_node, instance)
        )

        # start shadow migration
        # kvhts migration
        kvstc_done = asyncio.Event()
        kvhts_path = self._kvhts_ipc_path_for_migration(migration_id)
        tksth_path = self._tksth_ipc_path_for_migration(migration_id)
        additional_blocks_per_request = self.router_config.get(
            "additional_blocks_per_request", 0
        )
        # kvhts shadow side
        t_kvhts_shadow_side = time.perf_counter()
        await source.shadow_backend_instance.shadow_migration_recv.remote(
            migration_id, kvhts_path
        )
        elapsed_kvhts_shadow_side = time.perf_counter() - t_kvhts_shadow_side
        # kvhts hot side
        async with self.waiting_count_lock:
            num_waiting = self.waiting_count
        if num_waiting <= 0:
            return await start_backend_instance_task
        max_requests = num_waiting
        if self.shadow_migration_target is not None:
            max_requests = min(num_waiting, self.shadow_migration_target)
        if max_requests <= 0:
            return await start_backend_instance_task
        t_kvhts_hot_side = time.perf_counter()
        migrated_requests: List[
            Dict[str, int | str]
        ] = await source.backend_instance.shadow_migration_migrate.remote(
            migration_id,
            kvhts_path,
            tksth_path,
            max_requests,
            additional_blocks_per_request,
        )
        elapsed_kvhts_hot_side = time.perf_counter() - t_kvhts_hot_side
        if not migrated_requests:
            return await start_backend_instance_task

        async with self.migrated_request_lock:
            for request in migrated_requests:
                external_request_id = str(request["external_request_id"])
                self.migrated_request_targets[external_request_id] = (
                    instance,
                    kvstc_done,
                )

        logger.info(
            "Shadow scale-up KVHTS migrate started migration_id=%s requests=%s",
            migration_id,
            migrated_requests,
        )

        # wait for kvhts migration to be completed
        async with asyncio.timeout(300):
            while True:
                completed = await source.shadow_backend_instance.shadow_migration_completed.remote()
                if migration_id in completed.get(
                    "completed_kvhts_sessions", []
                ):
                    break
                await asyncio.sleep(0.5)
        elapsed_kvhts_migration = time.perf_counter() - t_kvhts_shadow_side
        logger.info(
            "Shadow scale-up KVHTS migrate completed migration_id=%s",
            migration_id,
        )
        await source.add_requests(-len(migrated_requests))

        start_backend_instance_results = await start_backend_instance_task

        # kvstc migration
        await instance.add_requests(len(migrated_requests))
        kvstc_path = self._kvstc_ipc_path_for_migration(migration_id)

        # kvstc cold side
        t_kvstc_cold_side = time.perf_counter()
        await instance.backend_instance.shadow_migration_recv.remote(
            migration_id, kvstc_path
        )
        elapsed_kvstc_cold_side = time.perf_counter() - t_kvstc_cold_side
        # kvstc shadow side
        t_kvstc_shadow_side = time.perf_counter()
        migrated_requests = await source.shadow_backend_instance.shadow_migration_migrate.remote(
            migration_id, kvstc_path
        )
        elapsed_kvstc_shadow_side = time.perf_counter() - t_kvstc_shadow_side
        if not migrated_requests:
            return start_backend_instance_results
        logger.info(
            "Shadow scale-up KVSTC migrate started migration_id=%s requests=%s",
            migration_id,
            migrated_requests,
        )

        async with asyncio.timeout(300):
            while True:
                completed = await instance.backend_instance.shadow_migration_completed.remote()
                if migration_id in completed.get(
                    "completed_kvstc_sessions", []
                ):
                    break
                await asyncio.sleep(0.5)
        kvstc_done.set()
        elapsed_kvstc_migration = time.perf_counter() - t_kvstc_cold_side
        logger.info(
            "Shadow scale-up KVSTC migrate completed migration_id=%s",
            migration_id,
        )

        return {
            "kvhts_shadow_side": elapsed_kvhts_shadow_side,
            "kvhts_hot_side": elapsed_kvhts_hot_side,
            "kvhts_migration": elapsed_kvhts_migration,
            "kvstc_cold_side": elapsed_kvstc_cold_side,
            "kvstc_shadow_side": elapsed_kvstc_shadow_side,
            "kvstc_migration": elapsed_kvstc_migration,
        } | start_backend_instance_results

    async def _start_ft_instance(self, instance_id: str):
        raise NotImplementedError(
            "Creating FT instances is not supported for shadow router"
        )

    async def _finish_instance(self, instance_id: str):
        async with self.instance_management_lock:
            if instance_id not in self.deleting_inference_instances:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = self.deleting_inference_instances.pop(instance_id)
        async with instance.lock:
            instance.status = False
        if instance.shadow_backend_instance:
            await instance.shadow_backend_instance.stop.remote()
            ray.kill(instance.shadow_backend_instance)
        await instance.backend_instance.stop.remote()
        ray.kill(instance.backend_instance)
        await self.model_loading_scheduler.deallocate_resource.remote(
            self.model_name, instance_id, self.resource_requirements
        )

    async def _shutdown_instance(self, instance_id: str, is_ft: bool = False):
        logger.info(
            f"Force deleting an instance (even if it is busy) for model {self.model_name}"
        )
        async with self.instance_management_lock:
            if is_ft:
                pool = self.ready_ft_instances
            else:
                pool = self.ready_inference_instances
            if instance_id not in pool:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = pool.pop(instance_id)
            async with instance.lock:
                instance.status = False
        if instance.shadow_backend_instance:
            await instance.shadow_backend_instance.shutdown.remote()
            ray.kill(instance.shadow_backend_instance)
        await instance.backend_instance.shutdown.remote()
        ray.kill(instance.backend_instance)
        await self.model_loading_scheduler.deallocate_resource.remote(
            self.model_name, instance_id, self.resource_requirements
        )
        return
