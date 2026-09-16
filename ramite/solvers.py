# -*- coding: utf-8 -*-
import time, sys
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, List
import logging
from ortools.linear_solver import pywraplp

from ramite.resources import ResourceWorkflow, Task


class BaseSolver(ABC):
    def __init__(
        self,
        max_hops: int = 2,
        precision: float = 1,
        obj_lambda: float = 0,
        max_switches: int = 2,
        global_bandwidth: float = 1e9,
        use_cache: bool = False,
    ):
        """
        A abstract base solver for using a mix integer solver to place a set of tasks on a network of resources.
        :param max_hops: The maximum number of hops between the source request and a potential offload resource
        :param precision: Precision (number of decimals) to preserve when converting floating point values to integers
        :param obj_lambda: Weighting to tune the importance of compute and latency
        :param max_switches: The maximum number of allowable changes when finding solution in intertask solver/adaptation
        :param use_cache: Cache the workflows used to prevent recomputing path lengths on subsequent calls to solve
        """
        self.scenario = None
        self.solver = None
        self.precision = precision
        self.use_cache = use_cache
        self.cache_valid = False
        self.cache = defaultdict(list)
        self.max_hops = max_hops
        self.obj_lambda = obj_lambda
        self.max_switches = max_switches
        self.global_bandwidth = global_bandwidth

        # Bandwidths to consider allocating. Should probabaly move to resource?
        self.bandwidths = frozenset([1e6, 1e7, 1e8, 1e9])
        self.global_resource_name = "GLOBAL_RESOURCES"

        # members used to track information for solver
        self.variables = {}
        self.node_constraints = defaultdict(list)
        self.node_constraints_cpu = defaultdict(list)
        self.node_constraints_memory = defaultdict(list)
        self.task_constraints = defaultdict(list)
        self.edge_constraints_bandwidth = defaultdict(list)
        self.global_edge_constraint = []
        self.objective_terms = []

    @property
    def max_hops(self) -> int:
        """
        :return: The maximum number of hops considered
        """
        return self._max_hops

    @max_hops.setter
    def max_hops(self, new_max_hops: int) -> None:
        """
        :param new_max_hops: A new max_hops limit
        :return: None
        """
        self._max_hops = new_max_hops
        self.network_change()

    def network_change(self) -> None:
        """
        Invalidate the cache when the network changes
        :return: None
        """
        self.cache.clear()
        self.cache_valid = False

    def get_workflows(
        self, task: Task, filter_fn: Callable[..., bool] = lambda **_: True
    ) -> List[ResourceWorkflow]:
        """
        An iterator that provides all candidate workflows that could accept a task from the source node
        :param source_name: The orignination point of a task
        :param filter_fn: A callable that can be used to determine if the workflow is a candidate for selection
        :return: A ResourceWorkflow object
        """
        # Go through each resource
        # 1. If we are useing a cache and it is valid and the task source is in the cache then walk over each
        #    destination within max_hops and yield each workflow
        # 2. If we are not using a cache, or the cache is invalid, or the destination is not in the cache then
        #    Go through each node in the network, check if it is within max_hops and the filter callback returns true,
        #    if yield each Workflow.
        #    Add each nodes workflows to the cache if we are using it.
        source_name = task.source
        if self.use_cache and self.cache_valid and source_name in self.cache:
            for resource_workflow in self.cache.get(source_name, []):
                assert filter_fn(resource_workflow)
                yield resource_workflow
        else:
            for dest_name in self.scenario.resources():
                paths = self.scenario.get_hops_all_paths(source_name, dest_name)
                for path in paths:
                    path_length = len(path)
                    if path_length > 0 and path_length - 1 <= self.max_hops:
                        dest_resource = self.scenario.get_attributes(dest_name)[
                            "resource"
                        ]
                        workflows = dest_resource.workflows
                        for workflow_num, workflow in enumerate(workflows):
                            if workflow.num_hops == path_length - 1:
                                inference_time = workflow.inference_time(
                                    task.num_images
                                )
                                execution_time = workflow.total_time(task.num_images)
                                bandwidth_allocation = workflow.bandwidth
                                resource_workflow = ResourceWorkflow(
                                    dest_name,
                                    workflow_num,
                                    workflow,
                                    bandwidth_allocation,
                                    path,
                                    inference_time,
                                    execution_time,
                                )
                                if filter_fn(task, resource_workflow):
                                    if self.use_cache:
                                        self.cache[source_name].append(
                                            resource_workflow
                                        )
                                    yield resource_workflow
            self.cache_valid = True

    def _prepare_solver(self, scenario):
        if scenario != self.scenario:
            self.network_change()
            self.scenario = scenario

        # Bandwidths to consider allocating. Should probabaly move to resource?
        #self.bandwidths = [1e5, 1e6, 1e7, 1e8, 1e9]

        # Solver
        # Create the mip solver with the SCIP backend.
        self.solver = pywraplp.Solver.CreateSolver("SCIP")

        # Variables
        # x[i, j, k] is an array of 0-1 variables, which will be 1
        # if task j from class i is chosen for server k.
        self.variables = {}
        self.node_constraints = defaultdict(list)
        self.node_constraints_cpu = defaultdict(list)
        self.node_constraints_memory = defaultdict(list)
        self.task_constraints = defaultdict(list)
        self.edge_constraints_bandwidth = defaultdict(list)
        self.task_switch_constraints = []
        self.global_edge_constraint = []
        self.objective_terms = []

    def add_variable(self, task, possible_workflow, candidates = None, max_candidates = None):
        path = possible_workflow.path
        workflow = possible_workflow.workflow
        bandwidth_allocation = possible_workflow.bandwidth_allocation
        core_usage = workflow.core_usage
        inference_time = possible_workflow.inference_time
        img_comp = workflow.image_quality
        vid = (
            possible_workflow.dest_name,
            task.task_num,
            possible_workflow.workflow_num,
            str(possible_workflow.path),
            bandwidth_allocation,
        )
        assert vid not in self.variables, "Duplicate variable in solver"
        variable = self.solver.IntVar(
            0,
            1,
            f"x_{task.task_num}_{task.source}_{possible_workflow.dest_name}_{workflow.name}_{workflow.model_compression}_{bandwidth_allocation}_{path}_{core_usage}_{img_comp}",
        )
        self.variables[vid] = variable
        self.task_constraints[task.task_num].append(variable)
        if (
            task.assigned
            and task.destination != possible_workflow.dest_name
            and task.reassign == False
        ):
            self.task_switch_constraints.append(variable * self.precision)
        self.node_constraints_cpu[possible_workflow.dest_name].append(
            (workflow.core_usage * self.precision) * variable
        )
        self.node_constraints_memory[possible_workflow.dest_name].append(
            (workflow.mem_util_workers * self.precision) * variable
        )
        for src, dst in zip(path, path[1:]):
            self.edge_constraints_bandwidth[src, dst].append(
                (bandwidth_allocation * self.precision) * variable
            )
            self.global_edge_constraint.append(
                (bandwidth_allocation * self.precision) * variable
            )
        #print(f"obj_term: {(1 - self.obj_lambda) * workflow.acc - self.obj_lambda * inference_time}->{((1 - self.obj_lambda) * workflow.acc - self.obj_lambda * inference_time)}")
        self.objective_terms.append(
            (
                (
                    (1 - self.obj_lambda) * workflow.acc - self.obj_lambda * inference_time
                )
                * self.precision
                * task.priority
                * 1# if not candidates else (max_candidates/candidates)
            )
            * variable
        )

    def _add_single_workflow_constraint(self):
        # Constraint: Choose only 1 Workflow per task
        for tc in self.task_constraints.values():
            self.solver.Add(self.solver.Sum(tc) == 1)

    def _add_at_most_single_workflow_constraint(self):
        # Constraint: Choose only 1 Workflow per task
        for tc in self.task_constraints.values():
            self.solver.Add(self.solver.Sum(tc) <= 1)

    def _add_computational_capacity_constraint(self):
        # Constraint: Do not exceed computational capacity
        for dest_name, constraints in self.node_constraints.items():
            capacity = self.scenario.get_attributes(dest_name)["resource"].task_capacity
            self.solver.Add(sum(constraints) <= capacity * self.precision)

    def _add_memory_capacity_constraint(self):
        # Constraint: Do not exceed memory capacity
        for dest_name, constraints in self.node_constraints_memory.items():
            capacity = self.scenario.get_attributes(dest_name)["resource"].memory
            self.solver.Add(sum(constraints) <= capacity * self.precision)

    def _add_memory_capacity_constraint_switch(self):
        # Constraint: Do not exceed memory capacity
        for dest_name, constraints in self.node_constraints_memory.items():
            current_reservation = sum(
                task.mem_util_workers
                for task in self.scenario.running_tasks
                if task.destination == dest_name and task.reassign == False
            )
            #print(f"memory_current_reservation ({dest_name}): {current_reservation}")
            capacity = (
                self.scenario.get_attributes(dest_name)["resource"].memory
                - current_reservation
            )
            self.solver.Add(sum(constraints) <= capacity * self.precision)

    def _add_cpu_capacity_constraint(self):
        # Constraint: Do not exceed CPU core capacity
        for dest_name, constraints in self.node_constraints_cpu.items():
            capacity = self.scenario.get_attributes(dest_name)["resource"].CPU_cores
            self.solver.Add(sum(constraints) <= capacity * self.precision)

    def _add_cpu_capacity_constraint_switch(self):
        # Constraint: Do not exceed CPU core capacity
        for dest_name, constraints in self.node_constraints_cpu.items():
            current_reservation = sum(
                task.core_usage
                for task in self.scenario.running_tasks
                if task.destination == dest_name and task.reassign == False
            )
            capacity = (
                self.scenario.get_attributes(dest_name)["resource"].CPU_cores
                - current_reservation
            )
            print(f"cpu_avail ({dest_name}): capacity {int(capacity)}")
            if capacity<0:
                print(f"\tcurrent_reservation {int(current_reservation)}")
            self.solver.Add(sum(constraints) <= capacity * self.precision)

    def _add_edge_bandwidth_constraint(self):
        # Do not exceed edge capacity
        for edge, constraints in self.edge_constraints_bandwidth.items():
            capacity = self.scenario.network.get_edge_data(edge[0], edge[1])[
                "link_speed"
            ]
            self.solver.Add(sum(constraints) <= capacity * self.precision)

    def _add_edge_bandwidth_constraint_switch(self):
        # Do not exceed edge capacity
        current_reservation = defaultdict(int)
        for task in self.scenario.running_tasks:
            for src, dst in zip(task.path, task.path[1:]):
                current_reservation[src, dst] += task.bandwidth_allocation

        #print(f"edge_bandwith current_reservation: {current_reservation}")

        for edge, constraints in self.edge_constraints_bandwidth.items():
            capacity = (
                self.scenario.network.get_edge_data(edge[0], edge[1])["link_speed"]
                - current_reservation[edge[0], edge[1]]
            )
            print(f"edge_bandwith avail: [{edge[0], edge[1]}]: {capacity/1e6}")
            self.solver.Add(sum(constraints) <= capacity * self.precision)

    def _add_global_bandwidth_constraint(self):
        # Do not exceed global bandwidth
        capacity = self.global_bandwidth
        self.solver.Add(
            self.solver.Sum(self.global_edge_constraint) <= capacity * self.precision
        )

    def _add_global_bandwidth_constraint_switch(self):
        # Do not exceed global bandwidth
        current_reservation = sum(
            task.bandwidth_allocation * (len(task.path) - 1)
            for task in self.scenario.running_tasks
            if task.reassign == False
        )
        #print(f"global_bandwith: {current_reservation}")
        capacity = self.global_bandwidth + current_reservation
        self.solver.Add(
            self.solver.Sum(self.global_edge_constraint) <= capacity * self.precision
        )

    def _add_switching_constraint(self):
        # Do not exceed max number of switches
        capacity = self.max_switches
        self.solver.Add(
            self.solver.Sum(self.task_switch_constraints) <= capacity * self.precision
        )

    def _solve(self, tasks):
        self.solver.Maximize(self.solver.Sum(self.objective_terms))

        # Solve
        start_time = time.process_time()
        status = pywraplp.Solver.INFEASIBLE
        acc = 0
        total_execution_time = 0
        mean_execution_time = 0
        execution_time_per_frame = 0
        execution_time_per_frame_w_overhead = 0
        mean_execution_time_per_frame = 0
        mean_execution_time_per_frame_w_overhead = 0

        if (
            len(self.variables) > 0
        ):  # If we have no variables then there is no point running the solver
            status = self.solver.Solve()
            solve_time = time.process_time() - start_time
            if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
                placement = [
                    str(v) for v in self.solver.variables() if v.solution_value() > 0.5
                ]
                placement = sorted(placement, key=lambda k: int(k.split("_")[1]))
                for task in placement:
                    (
                        _,
                        task_idx,
                        source,
                        dest,
                        workflow_name1,
                        workflow_name2,
                        bandwidth_allocation,
                        path,
                        core_usage,
                        img_comp,
                    ) = task.split("_")
                    workflow_name = workflow_name1 + "_" + workflow_name2
                    bandwidth_allocation = float(bandwidth_allocation)
                    task = tasks[int(task_idx)]
                    assert task.source == source
                    task.dest = dest
                    task.workflow_name = workflow_name
                    task.bandwidth_allocation = bandwidth_allocation
                    task.path = path
                    num_hops = path.count(",")
                    task.core_usage = float(core_usage)
                    task.image_compression = img_comp

                    chosen_workflow = next(
                        (
                            x
                            for x in self.scenario.get_attributes(dest)[
                                "resource"
                            ].workflows
                            if x.name == workflow_name1
                            and x.core_usage == float(core_usage)
                            and x.image_quality == float(img_comp)
                            and x.num_hops == num_hops
                            and x.bandwidth == bandwidth_allocation
                            and x.model_compression == float(workflow_name2)
                        ),
                        None,
                    )
                    task.accuracy = chosen_workflow.acc
                    task.image_compression = chosen_workflow.image_quality
                    task.model_compression = chosen_workflow.model_compression
                    task.device_type = chosen_workflow.device_type
                    task.on_gpu = chosen_workflow.on_gpu
                    task.mem_util_workers = chosen_workflow.mem_util_workers
                    task.case = chosen_workflow.case
                    task.sensor_core_usage = chosen_workflow.sensor_core_usage
                    task.threads = chosen_workflow.inference_threads
                    task.model = chosen_workflow.model
                    task.model_runtime = chosen_workflow.model_runtime
                    acc += chosen_workflow.acc

                    path = path.strip("]['").split("', '")
                    execution_time = chosen_workflow.total_time(task.num_images)
                    #calculates the tasks expected end time (for later)
                    task.end_time = time.time()+execution_time
                    total_execution_time += execution_time
                    execution_time_per_frame += chosen_workflow.exec_time
                    execution_time_per_frame_w_overhead += chosen_workflow.inference_time(
                        task.num_images
                    )

                    task.path = path
                    task.execution_time = execution_time
                    task.inference_time = chosen_workflow.inference_time(
                        task.num_images
                    )
                    task.model_load_time = chosen_workflow.model_load_time
                    task.bandwidth = chosen_workflow.bandwidth
                    task.get_image_duration = chosen_workflow.get_image_duration
                    task.get_inference_duration = chosen_workflow.inference_duration
                    task.throughput = chosen_workflow.throughput
                    task.fps_o = 1 / chosen_workflow.exec_time
                if not placement:#add a blank placement for output formatting
                    placement = [0]
                acc /= len(placement)
                mean_execution_time = total_execution_time / len(placement)
                mean_execution_time_per_frame = execution_time_per_frame / len(placement)
                mean_execution_time_per_frame_w_overhead = execution_time_per_frame_w_overhead / len(placement)
                #Advanced usage-performance:
                if True:
                    logging.info('\nAdvanced usage:')
                    logging.info('Problem solved in %f milliseconds' % self.solver.wall_time())
                    logging.info('Problem solved in %d iterations' % self.solver.iterations())
                    logging.info
                    ('Problem solved in %d branch-and-bound nodes' % self.solver.nodes())

        return status, solve_time, tasks, acc, mean_execution_time, mean_execution_time_per_frame, mean_execution_time_per_frame_w_overhead

    @abstractmethod
    def __call__(self, scenario, tasks):
        pass


class OptimalSolverLatencyCpuRam(BaseSolver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def workflow_filter(self, task, candidate):
        return (
            candidate.inference_time <= task.timeliness
            and candidate.workflow.acc >= task.req_acc
        )

    def __call__(self, scenario, tasks):
        self._prepare_solver(scenario)

        for task_num, task in enumerate(tasks):
            task.task_num = task_num
            for candidate_workflow in self.get_workflows(
                task, filter_fn=self.workflow_filter
            ):
                self.add_variable(task, candidate_workflow)

        self._add_single_workflow_constraint()
        self._add_memory_capacity_constraint()
        self._add_cpu_capacity_constraint()
        self._add_edge_bandwidth_constraint()
        self._add_global_bandwidth_constraint()

        return self._solve(tasks)


class OptimalSolverWithPrePruner(BaseSolver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def workflow_filter(self, task, candidate):
        return (
            candidate.inference_time <= task.timeliness
            and candidate.workflow.acc >= task.req_acc
        )

    def __call__(self, scenario, tasks):
        self._prepare_solver(scenario)

        for task_num, task in enumerate(tasks):
            best_obj_val = {}
            chosen_workflows = {}
            task.task_num = task_num
            for candidate_workflow in self.get_workflows(
                task, filter_fn=self.workflow_filter
            ):
                obj_func_val = (
                    (1 - self.obj_lambda) * candidate_workflow.workflow.acc
                    - self.obj_lambda * candidate_workflow.inference_time
                )

                key = (
                    candidate_workflow.bandwidth_allocation,
                    candidate_workflow.dest_name,
                    tuple(candidate_workflow.path),
                    candidate_workflow.workflow.device_type,
                    candidate_workflow.workflow.core_usage,
                )
                if key not in best_obj_val or obj_func_val > best_obj_val[key]:
                    chosen_workflows[key] = candidate_workflow
                    best_obj_val[key] = obj_func_val

            for workflow in chosen_workflows.values():
                self.add_variable(task, workflow)

        self._add_single_workflow_constraint()
        self._add_memory_capacity_constraint()
        self._add_cpu_capacity_constraint()
        self._add_edge_bandwidth_constraint()
        self._add_global_bandwidth_constraint()

        return self._solve(tasks)


class OptimalSolverDropTasks(BaseSolver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def workflow_filter(self, task, candidate):
        return (
            candidate.inference_time <= task.timeliness
            and candidate.workflow.acc >= task.req_acc
        )

    def __call__(self, scenario, tasks):
        self._prepare_solver(scenario)

        for task_num, task in enumerate(tasks):
            task.task_num = task_num
            for candidate_workflow in self.get_workflows(
                task, filter_fn=self.workflow_filter
            ):
                self.add_variable(task, candidate_workflow)

        self._add_at_most_single_workflow_constraint()
        self._add_memory_capacity_constraint()
        self._add_cpu_capacity_constraint()
        self._add_edge_bandwidth_constraint()
        self._add_global_bandwidth_constraint()

        return self._solve(tasks)


class HybridSolverLatencyCpuRam(BaseSolver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.used_resources = None

    def _place_local_first(self, task):
        local_workflows = self.scenario.get_attributes(task.source)[
            "resource"
        ].workflows
        local_resource = self.scenario.get_attributes(task.source)["resource"]
        obj_func_max = float("-inf")
        chosen_workflow = None
        chosen_workflow_num = None
        task_placed = False
        for workflow_num, workflow in enumerate(local_workflows):
            if (
                workflow.acc >= task.req_acc
                and workflow.exec_time <= task.timeliness
                and workflow.core_usage + self.used_resources[task.source]["core_usage"]
                <= local_resource.CPU_cores
                and workflow.mem_util_workers + self.used_resources[task.source]["mem_util"]
                <= local_resource.memory
            ):
                obj_func_temp = (
                    1 - self.obj_lambda
                ) * workflow.acc - self.obj_lambda * workflow.exec_time
                if obj_func_temp > obj_func_max:
                    obj_func_max = obj_func_temp
                    chosen_workflow = workflow
                    chosen_workflow_num = workflow_num
        if chosen_workflow is not None:
            candidate_workflow = ResourceWorkflow(
                task.source,
                chosen_workflow_num,
                chosen_workflow,
                0.0,
                [task.source],
                chosen_workflow.exec_time,
            )
            self.add_variable(task, candidate_workflow)
            self.used_resources[task.source]["core_usage"] += chosen_workflow.core_usage
            self.used_resources[task.source]["mem_util"] += chosen_workflow.mem_util_workers
            task_placed = True
        return task_placed

    def workflow_filter(self, task, candidate):
        return (
            candidate.inference_time <= task.timeliness
            and candidate.workflow.acc >= task.req_acc
        )

    def __call__(self, scenario, tasks):
        self._prepare_solver(scenario)
        self.used_resources = defaultdict(lambda: {"core_usage": 0.0, "mem_util": 0.0})
        for task_num, task in enumerate(
            sorted(tasks, key=lambda x: x.priority, reverse=True)
        ):
            task.task_num = task_num
            task_placed = self._place_local_first(task)
            if not task_placed:
                for candidate_workflow in self.get_workflows(
                    task, filter_fn=self.workflow_filter
                ):
                    self.add_variable(task, candidate_workflow)

        self._add_single_workflow_constraint()
        self._add_memory_capacity_constraint()
        self._add_cpu_capacity_constraint()
        self._add_edge_bandwidth_constraint()
        self._add_global_bandwidth_constraint()

        return self._solve(tasks)


class GreedySolver(BaseSolver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.used_resources = None
        self.used_capacity = None

    def filter(self, task, resource_workflow):
        network = self.scenario.network
        dest_name = resource_workflow.dest_name
        dest_resource = network.nodes[dest_name]["resource"]
        workflow = resource_workflow.workflow
        path = resource_workflow.path
        bandwidth_allocation = resource_workflow.bandwidth_allocation

        # make sure the resource can accomodate the workflow
        is_candidate = (
            workflow.core_usage
            <= dest_resource.CPU_cores - self.used_resources[dest_name]["core_usage"]
            and workflow.mem_util_workers
            <= dest_resource.memory - self.used_resources[dest_name]["mem_util"]
            and bandwidth_allocation
            <= self.global_bandwidth
            - self.used_resources[self.global_resource_name]["global_bandwidth_util"]
            and workflow.acc >= task.req_acc
        )
        if not is_candidate:
            return False

        # make sure that the route can accommodate the available capacity and maximum latency
        if resource_workflow.inference_time >= task.timeliness:
            return False

        # make sure link bandwidth isn't exceeded
        for src, dst in zip(path, path[1:]):
            if (
                bandwidth_allocation
                > network.get_edge_data(src, dst)["link_speed"]
                - self.used_resources[src, dst]["link_bandwidth_util"]
            ):
                return False
        return True

    def _prepare_solver(self, scenario):
        self.used_resources = defaultdict(
            lambda: {
                "core_usage": 0.0,
                "mem_util": 0.0,
                "global_bandwidth_util": 0.0,
                "link_bandwidth_util": 0.0,
            }
        )
        self.used_capacity = defaultdict(float)
        if scenario != self.scenario:
            self.network_change()
            self.scenario = scenario

        # Bandwidths to consider allocating. Should probabaly move to resource?
        self.bandwidths = [1e5, 1e6, 1e7, 1e8, 1e9]

        # Solver
        # Create the mip solver with the SCIP backend
        # This is not used in the greed solver but is here too keep a standard interface
        self.solver = pywraplp.Solver.CreateSolver("SCIP")

    def __call__(self, scenario, tasks):
        self._prepare_solver(scenario)

        # Solve
        start_time = time.process_time()

        acc = 0
        total_execution_time = 0
        num_placed_tasks = 0
        status = pywraplp.Solver.INFEASIBLE
        for task_num, task in enumerate(
            sorted(tasks, key=lambda x: x.priority, reverse=True)
        ):
            task.task_num = task_num
            obj_func_max = float("-inf")
            chosen_workflow = None
            chosen_inference_time = None
            chosen_execution_time = None
            for candidate_workflow in self.get_workflows(task, filter_fn=self.filter):
                inference_time = candidate_workflow.inference_time
                execution_time = candidate_workflow.execution_time
                obj_func_temp = (
                    (1 - self.obj_lambda) * candidate_workflow.workflow.acc
                    - self.obj_lambda * inference_time
                )
                if obj_func_temp > obj_func_max:
                    obj_func_max = obj_func_temp
                    chosen_workflow = candidate_workflow
                    chosen_inference_time = inference_time
                    chosen_execution_time = execution_time

            # Mark which resources are consumed
            if chosen_workflow is not None:
                self.used_resources[chosen_workflow.dest_name][
                    "core_usage"
                ] += chosen_workflow.workflow.core_usage
                self.used_resources[chosen_workflow.dest_name][
                    "mem_util"
                ] += chosen_workflow.workflow.mem_util_workers
                self.used_resources[self.global_resource_name][
                    "global_bandwidth_util"
                ] += chosen_workflow.bandwidth_allocation
                for src, dst in zip(chosen_workflow.path, chosen_workflow.path[1:]):
                    self.used_resources[src, dst][
                        "link_bandwidth_util"
                    ] += chosen_workflow.bandwidth_allocation

                # Report the decision
                task.dest = chosen_workflow.dest_name
                task.workflow_name = chosen_workflow.workflow.name
                task.accuracy = chosen_workflow.workflow.acc
                task.bandwidth_allocation = chosen_workflow.bandwidth_allocation
                task.path = chosen_workflow.path
                task.inference_time = chosen_inference_time
                task.execution_time = chosen_execution_time
                task.image_compression = chosen_workflow.workflow.image_quality
                task.model_compression = chosen_workflow.workflow.model_compression
                task.device_type = chosen_workflow.workflow.device_type
                task.core_usage = chosen_workflow.workflow.core_usage
                task.mem_util_workers = chosen_workflow.workflow.mem_util_workers
                task.on_gpu = chosen_workflow.workflow.on_gpu

                # Add to global performance
                acc += chosen_workflow.workflow.acc
                total_execution_time += chosen_execution_time
                num_placed_tasks += 1

        if num_placed_tasks == len(tasks):
            status = pywraplp.Solver.FEASIBLE

        solve_time = time.process_time() - start_time
        acc = acc / max(1, num_placed_tasks)
        mean_execution_time = total_execution_time / max(1, num_placed_tasks)
        return status, solve_time, tasks, acc, mean_execution_time


class MiniSolver(BaseSolver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def workflow_filter(self, adaptation, candidate, max_bandwidth):
        return (
            candidate.bandwidth <= max_bandwidth
            and candidate.task_type == adaptation.type  # make sure its appropriate model to task_type
            and candidate.num_hops == len(adaptation.path) - 1
            and candidate.acc >= adaptation.req_acc
            and candidate.exec_time <= adaptation.timeliness
            and candidate.on_gpu == adaptation.on_gpu
        )

    def __call__(self, scenario, adaptations):
        self._prepare_solver(scenario)

        print("Adaptation requested!")

        adaptations_out = []
        adaptations_failed = []

        for adaptation in adaptations:
            dest_name = adaptation.dest
            workflows = self.scenario.get_attributes(dest_name)["resource"].workflows
            cores_available = (
                0  # self.scenario.get_attributes(dest_name)["resource"].CPU_cores
            )
            current_workflow = next(
                (
                    x
                    for x in workflows
                    if x.model == adaptation.model
                    and x.image_quality == adaptation.image_quality
                    and x.model_compression == adaptation.model_compression
                    and x.bandwidth == adaptation.bandwidth_allocation
                    and x.num_hops == len(adaptation.path) - 1
                    and x.on_gpu == adaptation.on_gpu
                    and x.core_usage == adaptation.core_usage
                    and x.inference_threads == adaptation.threads
                ),
                None,
            )
            if (
                adaptation.recent_inference_duration
                >= 1.2 * current_workflow.inference_duration
            ):
                used_cores = adaptation.current_cores_used  # <- add current core usage
            else:
                used_cores = current_workflow.core_usage

            if (
                adaptation.recent_get_image_duration
                >= 1.2 * current_workflow.get_image_duration
            ):
                throughput_est = adaptation.recent_throughput
            else:
                throughput_est = current_workflow.bandwidth

            print(f"throughput_est: {throughput_est}")
            print(f"Current core usage: {used_cores}")

            count = 0
            obj_max = float("-inf")
            new_workflow = None
            for workflow in workflows:
                if self.workflow_filter(adaptation, workflow, throughput_est):
                    obj_val = (
                        1 - self.obj_lambda
                    ) * workflow.acc - self.obj_lambda * workflow.exec_time
                    can_fit = workflow.core_usage <= cores_available + used_cores
                    # print(f"{obj_val}\n{can_fit}\n{workflow}\n{obj_val > obj_max}\n")
                    if obj_val > obj_max and can_fit:
                        obj_max = obj_val
                        new_workflow = workflow
                    if can_fit:
                        count += 1
            # adaptation.image_compression = max(adaptation.image_compression - 10, 10)
            print(f"There are {count} adaptations to choose from")
            if new_workflow is not None:
                diff_model = (
                    adaptation.workflow_name
                    != f"{new_workflow.model}_{new_workflow.model_compression}"
                )
                if diff_model or (
                    not diff_model
                    and (adaptation.image_quality != new_workflow.image_quality)
                ):
                    print(
                        f"Model Compression changed from {adaptation.model_compression} to {new_workflow.model_compression}"
                    )
                    print(
                        f"Image Quality changed from {adaptation.image_compression} to {new_workflow.image_quality}"
                    )
                    print(
                        f"Accuracy changed from {adaptation.accuracy} to {new_workflow.acc}"
                    )
                    print(
                        f"Frame Duration changed from {adaptation.inference_time} to {new_workflow.exec_time}"
                    )
                    adaptation.image_compression = new_workflow.image_quality
                    adaptation.model_compression = new_workflow.model_compression
                    adaptation.bandwidth_allocation = new_workflow.bandwidth
                    adaptation.core_usage = new_workflow.core_usage
                    adaptation.accuracy = new_workflow.acc
                    adaptation.mem_util_workers = new_workflow.mem_util_workers
                    # adaptation.execution_time = new_workflow.total_time(adaptation.num_images)   <- update with num of images left
                    # adaptation.inference_time = new_workflow.inference_time(adaptation.num_images) <- update with num of images left

                    adaptations_out.append(adaptation)
                else:
                    print("No changes to image_quality or model_compression required")
            else:
                print("No suitable adaptations found")
                adaptations_failed.append(adaptation)

        return adaptations_out, adaptations_failed


class IntertaskSolver(BaseSolver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def workflow_filter(self, task, candidate):
        keep_same_dest = True
        if task.reassign and task.destination == candidate.dest_name:
            keep_same_dest = False
        return (
            candidate.inference_time <= task.timeliness
            and candidate.workflow.acc >= task.req_acc
            and candidate.workflow.task_type == task.type  # image or audio task_request?
            and keep_same_dest
        )

    def __call__(self, scenario, task_requests):
        self._prepare_solver(scenario)
        running_tasks = {t.id: t for t in self.scenario.running_tasks}
        tasks = []
        for task in task_requests:
            if task.id in running_tasks:
                running_tasks[task.id].reassign = True
            else:
                tasks.append(task)
        tasks.extend(list(running_tasks.values()))
        for task_num, task in enumerate(tasks):
            task.task_num = task_num
            for candidate_workflow in self.get_workflows(
                task, filter_fn=self.workflow_filter
            ):
                self.add_variable(task, candidate_workflow)

        self._add_single_workflow_constraint()
        #self._add_at_most_single_workflow_constraint()
        #self._add_memory_capacity_constraint_switch()
        self._add_cpu_capacity_constraint_switch()
        self._add_edge_bandwidth_constraint_switch()
        self._add_global_bandwidth_constraint_switch()
        self._add_switching_constraint()

        return self._solve(tasks)
class IntertaskDropSolver(BaseSolver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def workflow_filter(self, task, candidate):
        keep_same_dest = True
        if task.reassign and task.destination == candidate.dest_name:
            keep_same_dest = False
        return (
            candidate.inference_time <= task.timeliness
            and candidate.workflow.acc >= task.req_acc
            and candidate.workflow.task_type == task.type  # filter for image or audio task_request?
            and keep_same_dest
        )

    def __call__(self, scenario, task_requests):
        self._prepare_solver(scenario)
        running_tasks = {t.id: t for t in self.scenario.running_tasks}
        tasks = []
        for task in task_requests:
            if task.id in running_tasks:
                running_tasks[task.id].reassign = True
            else:
                tasks.append(task)
        tasks.extend(list(running_tasks.values()))
        #max_candidates = 0
        for task_num, task in enumerate(tasks):
            task.task_num = task_num
            candidate_workflows = self.get_workflows(task, filter_fn=self.workflow_filter)
            #candidate_workflows = list(candidate_workflows)
            #candidates = sum(1 for _ in candidate_workflows)
            #if candidates > max_candidates:
            #    max_candidates = candidates
            #print(f"task: {task.source} -> candidates: {len(candidate_workflows)}")
            for candidate in candidate_workflows:
                self.add_variable(task, candidate)

        self._add_at_most_single_workflow_constraint()
        #self._add_memory_capacity_constraint_switch()
        #self._add_single_workflow_constraint()
        self._add_cpu_capacity_constraint_switch()
        self._add_edge_bandwidth_constraint_switch()
        #self._add_global_bandwidth_constraint_switch()
        self._add_switching_constraint()

        return self._solve(tasks)

class newSolver(BaseSolver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Here you can filter out certain workflows, e.g., only consider 1 ML model
    def workflow_filter(self, task, candidate):
        return (
            candidate.inference_time <= task.timeliness
            and candidate.workflow.acc >= task.req_acc
            and candidate.workflow.model_compression == 0
            and candidate.workflow.image_quality == 100
        )

    def __call__(self, scenario, task_requests):
        if scenario != self.scenario:
            self.network_change()
            self.scenario = scenario
               
        for task_num, task in enumerate(task_requests):
            task.task_num = task_num
            # Generate the workflows (enumerate options like path, models, bw slices, etc.)
            for candidate_workflow in self.get_workflows(
                task, filter_fn=self.workflow_filter # filter out the ones we don't want or don't meet requirements
            ):
                new_variable = [] # Do something with candidate_workflow (create and add to list of variables)
        
        # After collecting the variables, create optimization problem, and solve
        
        solution = [] # placeholder
        '''
        Expected outputs in solution variable
        status: 0 for optimal, 1 for feasible, 2 for infeasible (from OR-Tools)
        solve_time: How long it takes to solve
        tasks: Task object (see ramite/resources.py for description)
        acc: average accuracy across tasks
        mean_execution_time: mean total expected time to complete (in our case process all images)
        mean_execution_time_per_frame: mean_execution_time per frame (image)
        mean_execution_time_per_frame_w_overhead: above + time to start-up services
        '''
        return solution 

SOLVER_OPTIONS = {
    "OptimalSolver_Latency_CPU_RAM": OptimalSolverLatencyCpuRam,
    "GreedySolver": GreedySolver,
    "HybridSolver_Latency_CPU_RAM": HybridSolverLatencyCpuRam,
    "greedy": GreedySolver,
    "hybrid": HybridSolverLatencyCpuRam,
    "optimal": OptimalSolverLatencyCpuRam,
    "optimal2": OptimalSolverDropTasks,
    "pre-pruner": OptimalSolverWithPrePruner,
    "intertask": IntertaskSolver,
    "intertaskdrop": IntertaskDropSolver,
    "newSolver": newSolver
}

MINISOLVER_OPTIONS = {
    "minisolver": MiniSolver,
}