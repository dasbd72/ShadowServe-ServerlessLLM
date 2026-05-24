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
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import ray

from sllm.fine_tuning_instance import start_ft_instance
from sllm.inference_instance import start_instance
from sllm.logger import init_logger

from ..utils import InstanceHandle
from .router_utils import SllmRouter

logger = init_logger(__name__)


async def auto_scaler(
    auto_scaling_metrics: Dict[str, int], auto_scaling_config: Dict[str, int]
) -> int:
    """
    Returns desired number of instances for a model based on the auto-scaling policy
    """

    request_count = auto_scaling_metrics.get("request_count", 0)

    min_instances = auto_scaling_config.get("min_instances", 0)
    max_instances = auto_scaling_config.get("max_instances", 10)
    target_ongoing_requests = auto_scaling_config.get("target", 2)

    desired_instances = (
        request_count + target_ongoing_requests - 1
    ) // target_ongoing_requests
    desired_instances = min(
        max_instances, max(min_instances, desired_instances)
    )

    return desired_instances


class RoundRobinRouter(SllmRouter):
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
        self.model_name = model_name
        self.resource_requirements = resource_requirements
        self.backend = backend
        self.backend_config = backend_config
        self.router_config = router_config

        self.loop_interval = 1
        self.loop = asyncio.get_running_loop()
        self.request_queue = asyncio.Queue()  # type:ignore
        # Inference instance pools
        self.starting_inference_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.deleting_inference_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.ready_inference_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.shadow_backend_instance: Optional[ray.actor.ActorHandle] = None  # type:ignore
        # Fine-tuning instance pools
        self.starting_ft_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.deleting_ft_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.ready_ft_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.instance_management_lock = asyncio.Lock()

        self.auto_scaling_config = {}
        self.auto_scaling_lock = asyncio.Lock()

        self.request_count = 0
        self.request_count_lock = asyncio.Lock()

        self.waiting_count = 0
        self.waiting_count_lock = asyncio.Lock()

        self.fine_tuning_count = 0
        self.fine_tuning_count_lock = asyncio.Lock()

        self.running = False
        self.running_lock = asyncio.Lock()

        self.idle_time = 0
        self.idle_time_lock = asyncio.Lock()

        self.enable_lora = enable_lora
        self.loaded_lora_adapters = lora_adapters
        self.lora_lock = asyncio.Lock()

        self.auto_scaler = None
        self.shadow_scale_up = router_config.get("shadow_scale_up", False)
        self.shadow_num_cpus = router_config.get(
            "shadow_num_cpus",
            resource_requirements.get("num_cpus", 1),
        )
        self.migration_id_counter = 0
        self.migration_id_lock = asyncio.Lock()
        self.migrated_request_targets: Dict[
            str, tuple[InstanceHandle, asyncio.Event]
        ] = {}
        self.migrated_request_lock = asyncio.Lock()
        logger.info(f"Created new handler for model {self.model_name}")

    async def start(
        self, auto_scaling_config: Dict[str, int], mode: str = "inference"
    ):
        self.model_loading_scheduler = ray.get_actor("model_loading_scheduler")
        if mode == "inference":
            async with self.auto_scaling_lock:
                self.auto_scaling_config = auto_scaling_config
            self.auto_scaler = asyncio.create_task(self._auto_scaler_loop())
            self.load_balancer = asyncio.create_task(self._load_balancer_loop())
            if self.shadow_scale_up and self.backend == "vllm":
                self.create_shadow_task = asyncio.create_task(
                    self._create_shadow_instance()
                )
        async with self.running_lock:
            self.running = True
        logger.info(f"Started handler for model {self.model_name}")

    async def update(
        self,
        auto_scaling_config: Optional[Dict[str, int]] = None,
        lora_adapters: Optional[Dict[str, str]] = None,
    ):
        if auto_scaling_config is not None:
            async with self.auto_scaling_lock:
                self.auto_scaling_config = auto_scaling_config

        if lora_adapters is not None:
            async with self.lora_lock:
                self.loaded_lora_adapters = lora_adapters

        logger.info(
            f"Model {self.model_name}'s auto scaling config updated to {auto_scaling_config}"
        )

    def _new_instance_id(self):
        pattern = "{model_name}_{id}"
        return pattern.format(model_name=self.model_name, id=uuid.uuid4())

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

    async def _resume_migrated_generate(
        self, hot_result: dict, original_request_data: dict
    ) -> dict:
        migration_meta = hot_result.get("_sllm_migration") or {}
        request_id: str | None = str(
            migration_meta.get("external_request_id")
            or migration_meta.get("request_id")
            or hot_result.get("id")
        )
        if not request_id:
            return self._strip_internal_response_fields(hot_result)
        input_tokens: list[int] | None = migration_meta.get("input_tokens")
        if not input_tokens:
            return self._strip_internal_response_fields(hot_result)
        completion_tokens: int | None = migration_meta.get("completion_tokens")
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
        hot_metrics: dict | None,
        cold_metrics: dict | None,
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

    async def fine_tuning(self, request_data: dict):
        logger.info(f"Starting fine-tuning for model {self.model_name}")
        async with self.running_lock:
            if not self.running:
                return {"error": "Instance stopped"}

        async with self.fine_tuning_count_lock:
            self.fine_tuning_count += 1

        try:
            instance_id = await self._create_ft_instance()
        except Exception as e:
            logger.error(f"Failed to create fine-tuning instance: {str(e)}")
            async with self.fine_tuning_count_lock:
                self.fine_tuning_count -= 1
            return {"error": f"Failed to create fine-tuning instance: {str(e)}"}

        max_wait_time = 300
        wait_time = 0
        while wait_time < max_wait_time:
            async with self.instance_management_lock:
                if instance_id in self.ready_ft_instances:
                    instance = self.ready_ft_instances[instance_id]
                    break
                elif instance_id not in self.starting_ft_instances:
                    logger.error(
                        f"Fine tuning instance {instance_id} not found in starting or ready instances"
                    )
                    async with self.fine_tuning_count_lock:
                        self.fine_tuning_count -= 1
                    return {"error": "Fine tuning instance not found"}
            await asyncio.sleep(0.1)
            wait_time += 0.1
        else:
            logger.error(
                f"Timeout waiting for fine tuning instance {instance_id} to be ready"
            )
            async with self.fine_tuning_count_lock:
                self.fine_tuning_count -= 1
            return {
                "error": "Timeout waiting for fine tuning instance to be ready"
            }

        try:
            logger.info(f"Calling fine_tuning method on instance {instance_id}")
            result = await instance.backend_instance.fine_tuning.remote(
                request_data=request_data
            )

            logger.info(f"Finished processing fine-tuning {self.model_name}")
            await instance.add_requests(-1)

            await self._shutdown_instance(instance_id, is_ft=True)

            async with self.fine_tuning_count_lock:
                self.fine_tuning_count -= 1

            return result
        except Exception as e:
            logger.error(f"Fine-tuning failed: {str(e)}")
            await instance.add_requests(-1)
            await self._shutdown_instance(instance_id, is_ft=True)
            async with self.fine_tuning_count_lock:
                self.fine_tuning_count -= 1
            logger.info(
                f"Fine-tuning failed and cleaned up for model {self.model_name}"
            )
            return {"error": f"Fine-tuning failed: {str(e)}"}

    async def delete_adapters(self, lora_adapters: List[str]):
        async with self.lora_lock:
            for adapter_name in lora_adapters:
                if adapter_name in self.loaded_lora_adapters:
                    del self.loaded_lora_adapters[adapter_name]
        logger.info(
            f"Deleted LoRA adapters {lora_adapters} on model {self.model_name}"
        )

    async def shutdown(self):
        async with self.running_lock:
            self.running = False
        # stop all inference instances
        # return all unfinished requests
        while not self.request_queue.empty():
            instance_allocation = await self.request_queue.get()
            instance_allocation.set_result({"error": "Instance cancelled"})
            async with self.waiting_count_lock:
                self.waiting_count -= 1

        async with self.instance_management_lock:
            deleted_instance_id = list(self.ready_inference_instances.keys())
        delete_tasks = [
            self._shutdown_instance(instance_id)
            for instance_id in deleted_instance_id
        ]
        await asyncio.gather(*delete_tasks)

        return deleted_instance_id

    async def _load_balancer_loop(self):
        # this is a simple round-robin load balancer
        round_robin_index = 0
        while True:
            instance_allocation = await self.request_queue.get()
            allocated = False
            logger.info(f"A request is waiting for model {self.model_name}")
            while not allocated:
                # 1. get ready instances
                instance_options = None
                while not instance_options:
                    await asyncio.sleep(1)
                    async with self.instance_management_lock:
                        instance_options = list(
                            self.ready_inference_instances.keys()
                        )
                    logger.info(f"{instance_options}")
                logger.info(f"Got ready instances {instance_options}")
                instance_id = instance_options[
                    round_robin_index % len(instance_options)
                ]
                round_robin_index += 1
                async with self.instance_management_lock:
                    if instance_id not in self.ready_inference_instances:
                        continue
                    instance = self.ready_inference_instances[instance_id]
                    # check if the request queue reaches max length
                    if await instance.check_request_queue():
                        allocated = await instance.add_requests(1)
                        if allocated:
                            instance_allocation.set_result(instance_id)
                            async with self.waiting_count_lock:
                                self.waiting_count -= 1
                    else:
                        logger.info(
                            f"Instance {instance_id} cannot add another request"
                        )
                if not allocated:
                    await asyncio.sleep(self.loop_interval)

    async def _auto_scaler_loop(self):
        while True:
            # logger.info(f"Auto-scaling for model {self.model_name}")
            async with self.auto_scaling_lock:
                auto_scaling_config = self.auto_scaling_config.copy()
            async with self.request_count_lock:
                request_count = self.request_count
            auto_scaling_metrics = {"request_count": request_count}
            desired_instances = await auto_scaler(
                auto_scaling_metrics, auto_scaling_config
            )
            async with self.instance_management_lock:
                num_starting_instances = len(self.starting_inference_instances)
                num_running_instances = len(self.ready_inference_instances)
            logger.info(
                f"{self.model_name}: {num_running_instances} running instances, "
                f"{num_starting_instances} starting instances, "
                f"{desired_instances} instances needed",
            )
            if (
                desired_instances
                > num_running_instances + num_starting_instances
            ):
                logger.info("Creating new instance")
                await self._create_instance()
            elif desired_instances < num_running_instances:
                keep_alive = auto_scaling_config.get("keep_alive", 0)
                if self.idle_time >= keep_alive:
                    logger.info(
                        f"Stopping instance, idle_time: {self.idle_time}, keep_alive: {keep_alive}"
                    )
                    await self._stop_instance()
                    async with self.idle_time_lock:
                        self.idle_time = 0
                else:
                    logger.info(
                        f"idle_time: {self.idle_time}, keep_alive: {keep_alive}"
                    )
                    async with self.idle_time_lock:
                        self.idle_time += self.loop_interval
            else:
                # logger.info("No scaling needed")
                pass
            await asyncio.sleep(self.loop_interval)

    async def _create_instance(self):
        instance_id = self._new_instance_id()
        logger.info(
            f"Creating new instance {instance_id} for model {self.model_name}"
        )
        # get max_queue_length from auto_scaling_config
        if self.auto_scaling_config.get("metric", "") == "concurrency":
            max_request_length = self.auto_scaling_config.get("target", 1)
        else:
            max_request_length = 1
        logger.info(
            f"Creating new instance {instance_id} for model {self.model_name}, max queue length is {max_request_length}"
        )
        instance = InstanceHandle(
            instance_id=instance_id,
            max_queue_length=max_request_length,
            num_gpu=self.resource_requirements["num_gpus"],
        )
        async with self.instance_management_lock:
            self.starting_inference_instances[instance_id] = instance
        self.loop.create_task(self._start_instance(instance_id))

        return instance_id

    async def _create_ft_instance(self):
        instance_id = self._new_instance_id()
        logger.info(
            f"Creating new FT instance {instance_id} for model {self.model_name}"
        )

        instance = InstanceHandle(
            instance_id=instance_id,
            max_queue_length=1,
            num_gpu=self.resource_requirements["num_gpus"],
        )
        async with self.instance_management_lock:
            self.starting_ft_instances[instance_id] = instance
        self.loop.create_task(self._start_ft_instance(instance_id))
        logger.info(f"Created task for starting FT instance {instance_id}")
        return instance_id

    async def _create_shadow_instance(self):
        instance_id = self._new_instance_id()
        logger.info(
            f"Creating new shadow instance {instance_id} for model {self.model_name}"
        )
        t_allocate_resource = time.perf_counter()
        startup_node = (
            await self.model_loading_scheduler.allocate_resource.remote(
                self.model_name,
                instance_id,
                {
                    "num_cpus": self.shadow_num_cpus,
                    "num_gpus": 0,
                },
            )
        )
        elapsed_allocate_resource = time.perf_counter() - t_allocate_resource

        t_start_shadow_instance = time.perf_counter()
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
        logger.info(
            f"Started shadow instance {instance_id} for model {self.model_name}"
        )
        elapsed_start_shadow_instance = (
            time.perf_counter() - t_start_shadow_instance
        )

        t_init_shadow_backend = time.perf_counter()
        self.shadow_backend_instance = ray.get_actor(instance_id)
        await self.shadow_backend_instance.init_backend.remote()
        elapsed_init_shadow_backend = (
            time.perf_counter() - t_init_shadow_backend
        )
        timing_results = {
            "allocate_resource": elapsed_allocate_resource,
            "start_shadow_instance": elapsed_start_shadow_instance,
            "init_shadow_backend": elapsed_init_shadow_backend,
        }
        logger.info(
            f"Created shadow instance {instance_id} for model {self.model_name} "
            f"{json.dumps(timing_results)}"
        )

    async def _start_instance(self, instance_id):
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
        timing_results = {}
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
        if self.shadow_scale_up and self.backend == "vllm":
            async with self.instance_management_lock:
                sources = list(self.ready_inference_instances.values())
            if sources:
                shadow_scale_up_source = max(
                    sources, key=lambda h: h.concurrency
                )

        if shadow_scale_up_source:
            results = await self._on_shadow_scale_up(
                startup_node, shadow_scale_up_source, instance
            )
            timing_results.update(results)
        else:
            results = await self._start_backend_instance(startup_node, instance)
            timing_results.update(results)

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
        logger.info(
            f"Started instance {instance_id} for model {self.model_name} took "
            f"{elapsed_start_backend_instance} seconds"
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
            self._start_backend_instance(startup_node, instance)
        )

        # start shadow migration
        # kvhts migration
        kvstc_done = asyncio.Event()
        kvhts_path = self._kvhts_ipc_path_for_migration(migration_id)
        # kvhts shadow side
        t_kvhts_shadow_side = time.perf_counter()
        await self.shadow_backend_instance.shadow_migration_recv.remote(
            migration_id, kvhts_path
        )
        elapsed_kvhts_shadow_side = time.perf_counter() - t_kvhts_shadow_side
        # kvhts hot side
        async with self.waiting_count_lock:
            num_waiting = self.waiting_count
        if num_waiting <= 0:
            return await start_backend_instance_task
        t_kvhts_hot_side = time.perf_counter()
        migrated_requests: list[
            dict[str, int | str]
        ] = await source.backend_instance.shadow_migration_migrate.remote(
            migration_id, kvhts_path, num_waiting
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
                completed = await self.shadow_backend_instance.shadow_migration_completed.remote()
                if migration_id in completed.get(
                    "completed_kvhts_sessions", []
                ):
                    break
                await asyncio.sleep(0.5)
        elapsed_kvhts_migration = time.perf_counter() - t_kvhts_shadow_side
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
        await self.shadow_backend_instance.shadow_migration_migrate.remote(
            migration_id, kvstc_path
        )
        elapsed_kvstc_shadow_side = time.perf_counter() - t_kvstc_shadow_side
        logger.info(
            "Shadow scale-up KVSTC migrate started migration_id=%s",
            migration_id,
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

        return {
            "kvhts_shadow_side": elapsed_kvhts_shadow_side,
            "kvhts_hot_side": elapsed_kvhts_hot_side,
            "kvhts_migration": elapsed_kvhts_migration,
            "kvstc_cold_side": elapsed_kvstc_cold_side,
            "kvstc_shadow_side": elapsed_kvstc_shadow_side,
            "kvstc_migration": elapsed_kvstc_migration,
        } | start_backend_instance_results

    async def _start_ft_instance(self, instance_id: str):
        async with self.instance_management_lock:
            if instance_id not in self.starting_ft_instances:
                logger.error(f"FT Instance {instance_id} not found")
                return
            instance = self.starting_ft_instances[instance_id]

        logger.info(
            f"Allocating FT resources for model {self.model_name} on {instance_id}"
        )
        try:
            startup_node = (
                await self.model_loading_scheduler.allocate_resource.remote(
                    self.model_name, instance_id, self.resource_requirements
                )
            )
            logger.debug(
                f"Allocated resources on node {startup_node} for FT instance {instance_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to allocate resources for FT instance {instance_id}: {str(e)}"
            )
            raise

        startup_config = {
            "num_cpus": self.resource_requirements["num_cpus"],
            "num_gpus": self.resource_requirements["num_gpus"],
            "resources": {
                "worker_node": 0.1,
                f"worker_id_{startup_node}": 0.1,
            },
        }

        try:
            instance.backend_instance = await start_ft_instance.options(
                resources=startup_config["resources"]
            ).remote(
                instance_id,
                self.backend,
                self.model_name,
                self.backend_config,
                startup_config,
            )
        except Exception as e:
            logger.error(
                f"Failed to create Ray actor for fine-tuning instance {instance_id}: {str(e)}"
            )
            raise
        async with instance.lock:
            instance.ready = True
            instance.node_id = startup_node
        try:
            await instance.backend_instance.init_backend.remote()
        except Exception as e:
            logger.error(
                f"Failed to initialize backend for fine-tuning instance {instance_id}: {str(e)}"
            )
            raise

        async with self.instance_management_lock:
            self.ready_ft_instances[instance_id] = instance
            self.starting_ft_instances.pop(instance_id)
        logger.info(f"Fine-tuning instance {instance_id} is now ready")
        return instance_id

    async def _stop_instance(self, instance_id: Optional[str] = None):
        while len(self.ready_inference_instances) <= 0:
            await asyncio.sleep(1)

        async with self.instance_management_lock:
            if instance_id is None:
                # Stop the lowest concurrency instance if instance_id is not provided
                instance_id = min(
                    self.ready_inference_instances.keys(),
                    key=lambda x: self.ready_inference_instances[x].concurrency,
                )
            if instance_id in self.ready_inference_instances:
                instance = self.ready_inference_instances.pop(instance_id)
            else:
                logger.error(f"Instance {instance_id} not found")
                return
            self.deleting_inference_instances[instance_id] = instance
        logger.info(
            f"Stopping instance {instance_id} for model {self.model_name}"
        )
        self.loop.create_task(self._finish_instance(instance_id))

    async def _finish_instance(self, instance_id: str):
        async with self.instance_management_lock:
            if instance_id not in self.deleting_inference_instances:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = self.deleting_inference_instances.pop(instance_id)
        async with instance.lock:
            instance.status = False
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
        await instance.backend_instance.shutdown.remote()
        ray.kill(instance.backend_instance)
        await self.model_loading_scheduler.deallocate_resource.remote(
            self.model_name, instance_id, self.resource_requirements
        )
        return
