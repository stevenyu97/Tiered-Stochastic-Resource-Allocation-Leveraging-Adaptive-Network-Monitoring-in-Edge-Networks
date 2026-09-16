# -*- coding: utf-8 -*-
import ast
from functools import lru_cache
from typing import Dict, List, Tuple, Type

import attrs
import matplotlib.pyplot as plt
import networkx as nx
import shortuuid
from attrs import define, field
from pandas import DataFrame as df


def is_atleast(atype: Type, min_value: float):
    return attrs.validators.and_(
        attrs.validators.instance_of(atype), attrs.validators.ge(min_value)
    )


def is_oneof(atype: Type, options: List):
    return attrs.validators.and_(
        attrs.validators.instance_of(atype), attrs.validators.in_(options)
    )


@define(frozen=True)
class Workflow:
    """Class for keeping track of capabilities and requirements of an ISR workflow."""

    name = field(validator=attrs.validators.instance_of(str))
    acc = field(converter=float, validator=is_atleast(float, 0.0))
    exec_time = field(converter=float, validator=is_atleast(float, 0.0))
    get_image_duration = field(converter=float, validator=is_atleast(float, 0.0))
    inference_duration = field(converter=float, validator=is_atleast(float, 0.0))
    core_usage = field(converter=float, validator=is_atleast(float, 0.0))
    model_compression = field(converter=int, validator=is_atleast(int, 0))
    image_quality = field(converter=int, validator=is_atleast(int, 0))
    device_type = field(validator=is_oneof(str, ["UGS", "PLT", "APC", "CHQ"]))
    bandwidth = field(converter=float, validator=is_atleast(float, 0.0))
    throughput = field(converter=float, validator=is_atleast(float, 0.0))
    num_hops = field(converter=int, validator=is_atleast(int, 0))
    on_gpu = field(converter=bool, validator=attrs.validators.instance_of(bool))
    model_load_time = field(converter=float, validator=is_atleast(float, 0.0))
    #added fields for new benchmarks
    mem_util_sensor = field(converter=float, validator=is_atleast(float, 0.0))
    mem_util_workers = field(converter=float, validator=is_atleast(float, 0.0))
    num_workers = field(converter=int, validator=is_atleast(int, 0))
    sensor_core_usage = field(converter=float, validator=is_atleast(float, 0.0))
    case = field(validator=is_oneof(str, ["ondevice", "offload", "mdi", "ddi"]))
    inference_threads = field(converter=int, validator=is_atleast(int, 0))
    model_runtime = field(validator=is_oneof(str, ["pytorch"]))
    model = field(validator=is_oneof(str,['mobilenetv2','resnet50','acdnet']))  
    task_type = field(validator=is_oneof(str, ["audio", "image"])) 

    def total_time(self, num_images: int) -> float:
        if num_images < 0:
            raise ValueError("num_images must be greater than zero")
        return self.model_load_time + self.exec_time * num_images

    def inference_time(self, num_images: int) -> float:
        if num_images < 0:
            raise ValueError("num_images must be greater than zero")
        return self.model_load_time / num_images + self.exec_time


@define
class Resource:
    """Class representing an the properties of an assignable resource"""

    name: str
    workflows: List[Workflow]
    task_capacity: field(validator=is_atleast(int, 0))
    CPU_cores: field(validator=is_atleast(int, 0))
    memory: field(validator=is_atleast(int, 0))
    has_gpu: field(converter=bool, default=False)


@define(slots=False)
class Task:
    """A task request and if selected assignment."""

    source: str
    type: field(validator=is_oneof(str, ["image", "audio"]))#what kind of task (to filter workflows)
    req_acc: field(validator=is_atleast(int, 0))
    priority: field(validator=is_atleast(int, 0))
    num_images: field(converter=int, validator=is_atleast(int, 1))
    timeliness: field(default=1000, validator=is_atleast(int, 0))
    id: str = field(default="")
    task_num: int = field(default=None)
    workflow_name: str = field(default=None)
    bandwidth_allocation: float = field(default=None)
    path: List[str] = field(default=None)
    core_usage: float = field(default=None)
    image_compression: float = field(default=None)
    accuracy: float = field(default=None)
    model_compression: float = field(default=None)
    device_type: str = field(default=None)
    on_gpu: bool = field(default=None)
    mem_util_workers: float = field(default=None)
    inference_time: float = field(default=None)
    execution_time: float = field(default=None)
    destination: str = field(default=None)
    reassign: bool = field(default=False)
    end_time: float = field(default=None)
    frames_left: int = field(default=None)
    cpu_deg: float = field(default=None)  # rate of cpu degredation
    bwp_deg: float = field(default=None)  # rate of bwp degredation
    #set_band: float = field(default=None)
    #set_cpu: float = field(default=None)
    #set_ugs_cpu: float = field(default=None)
    #num_tasks: float = field(default=None)
    #num_hops: float = field(default=None)
    threads: int = field(default=None)
    case: str = field(default=None)
    model: str = field(default=None)
    model_runtime: str = field(default=None)
    bw_usage: float = field(default=None)#Mbps
    dfps: int = field(default=None)#server data send rate

    def __attrs_post_init__(self):
        if self.id == "":
            self.id = shortuuid.uuid().lower()
        self._assigned = self.destination is not None
        if isinstance(self.path, str):
            self.path = ast.literal_eval(self.path)

    @property
    def assigned(self):
        return self._assigned

    @property
    def dest(self):
        if self._assigned:
            return self.destination
        return None

    @dest.setter
    def dest(self, destination):
        self._assigned = True
        self.destination = destination


@define
class ResourceWorkflow:
    "A class to hold information about selected workflow on a specific resource"
    dest_name: str
    workflow_num: field(validator=is_atleast(int, 0))
    workflow: Workflow
    bandwidth_allocation: field(validator=is_atleast(float, 0.0))
    path: List[Tuple[str, str]]
    inference_time: field(validator=is_atleast(float, 0.0))
    execution_time: field(validator=is_atleast(float, 0.0))


class Scenario:
    def __init__(self, network=None, solver=None, minisolver=None):
        self.solver = solver
        self.minisolver = minisolver
        if network is None:
            self.network = nx.Graph()
        else:
            self.network = network
        self.running_tasks = []
        self._solver = solver
        self._minisolver = minisolver

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, new_network):
        self._network = new_network
        self._network_change()

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, new_solver):
        self._solver = new_solver

    @property
    def minisolver(self):
        return self._minisolver

    @minisolver.setter
    def minisolver(self, new_minisolver):
        self._minisolver = new_minisolver

    def resources(self):
        return self.network.nodes()

    def get_attributes(self, resource_name):
        return self.network.nodes[resource_name]

    def update_resource(self, resource_name, new_resource):
        current_attributes = self.get_attributes(resource_name)["resources"]
        current_attributes.update(new_resource)

    def add_node(self, node_id, **kwargs):
        self.network.add_node(node_id, **kwargs)
        self._network_change()

    def remove_nodes(self, node_ids):
        if not isinstance(node_ids, list):
            node_ids = [node_ids]
        self.network.remove_nodes_from(node_ids)
        self._network_change()

    def add_edge(self, src, dst, **kwargs):
        self.network.add_edge(src, dst, **kwargs)
        self._network_change()

    def add_edges_from(self, all_edges):
        self.network.add_edges_from(all_edges)
        self._network_change()

    def remove_edge(self, src, dst):
        try:
            self.network.remove_edge(src, dst)
            self._network_change()
        except nx.NetworkXError:
            pass

    def update_edge_info(self, src, dest, **kwargs):
        edge_info = self.network.get_edge_data(src, dest)
        edge_info.update(**kwargs)

    @lru_cache()
    def get_hops(self, resource1, resource2):
        if resource1 == resource2:
            return 0
        try:
            num_hops = nx.shortest_path_length(self._network, resource1, resource2)
        except nx.NetworkXNoPath:
            num_hops = float("inf")
        return num_hops

    def get_hops_path(self, resource1, resource2):
        try:
            path = nx.shortest_path(self._network, resource1, resource2)
        except nx.NetworkXNoPath:
            path = []
        return path

    def get_hops_all_paths(self, resource1, resource2, greedy_flag=False):
        weight = "link_speed" if greedy_flag is True else None
        try:
            return list(
                nx.all_shortest_paths(
                    self._network, resource1, resource2, weight=weight
                )
            )
        except nx.NetworkXNoPath:
            return []

    def place_tasks(self, tasks):
        self.last_solution = self._solver(self, tasks)
        return self.last_solution

    def adapt_tasks(self, tasks):
        self.last_adaptation = self._minisolver(self, tasks)
        return self.last_adaptation

    def render(self):
        node_color = [self.get_attributes(r)["color"] for r in self.resources()]
        nx.draw_kamada_kawai(self._network, node_color=node_color, with_labels=True)
        plt.show()

    def _network_change(self):
        self.get_hops.cache_clear()
        if self.solver is not None and hasattr(self.solver, "network_change"):
            self.solver.network_change()


@define
class TaskAssignment:
    """
    A class that holds task assignments
    """

    tasks: List[Task]
    stats: Dict[str, object]

    def placement_dataframe(self):
        results = []
        for task in self.tasks:
            task_info = {"source": task.source}
            if task.assigned:
                task_info["id"] = task.id
                task_info["destination"] = task.dest
                task_info["path"] = task.path
                task_info["workflow_name"] = task.workflow_name
                task_info["cores_requested"] = task.core_usage
                task_info["BW_alloc_bps"] = task.bandwidth_allocation
                task_info["assigned"] = task.assigned
                task_info["accuracy"] = task.accuracy
                task_info["priority"] = task.priority
                task_info["img_qual"] = task.image_compression
                task_info["model_comp"] = task.model_compression
                task_info["dev_type"] = task.device_type
                task_info["gpu"] = task.on_gpu
                task_info["cpu_deg"] = task.cpu_deg
                task_info["bwp_deg"] = task.bwp_deg
                task_info["fps_o"] = task.fps_o
                task_info["req_acc"] = task.req_acc
                task_info["timeliness"] = task.timeliness
                task_info["sensor_core_usage"] = task.sensor_core_usage
                task_info["case"] = task.case
                task_info["threads"] = task.threads
                task_info["model"] = task.model
                task_info["model_runtime"] = task.model_runtime
                if hasattr(task, "fps_p"):
                    task_info["fps_p"] = task.fps_p
                if hasattr(task, "execution_time"):
                    task_info["task_execution_time"] = task.execution_time
                if hasattr(task, "execution_time_actual"):
                    task_info["task_execution_actual"] = task.execution_time_actual
                if hasattr(task, "inference_time"):
                    task_info["task_inference_time"] = task.inference_time
                if hasattr(task, "cores_available"):
                    task_info['cores_available'] = task.set_cpu
                if hasattr(task, "req_lat"):
                    task_info['req_lat'] = task.req_lat
                if hasattr(task, "num_images"):
                    task_info['num_images'] = task.num_images

            results.append(task_info)
        return df(results, index=range(len(self.tasks)))

    def stats_dataframe(self):
        return df(self.stats, index=[0])
