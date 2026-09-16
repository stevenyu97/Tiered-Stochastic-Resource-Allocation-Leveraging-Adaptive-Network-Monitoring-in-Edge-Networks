# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:04:33 2023

@author: jake.b.perazzone

This file creates resource from full RAMITE
"""

from enum import Enum
from functools import lru_cache
from math import ceil
from typing import Dict, List, Optional, Tuple, Type

import attrs
import matplotlib.pyplot as plt
import networkx as nx
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
    precision = field(converter=float, validator=is_atleast(float, 0.0))
    acc = field(converter=float, validator=is_atleast(float, 0.0))
    exec_time = field(converter=float, validator=is_atleast(float, 0.0))
    weight = field(converter=float, validator=is_atleast(float, 0.0))
    payload = field(converter=float, validator=is_atleast(float, 1.0))
    core_usage = field(converter=float, validator=is_atleast(float, 0.0))
    mem_util = field(converter=float, validator=is_atleast(float, 0.0))
    model_compression = field(converter=int, validator=is_atleast(int, 0))
    image_quality = field(converter=int, validator=is_atleast(int, 0))
    device_type = field(validator=is_oneof(str, ["UGS", "PLT", "APC", "CHQ"]))
    bandwidth = field(converter=float, validator=is_atleast(float, 0.0))
    total_time = field(converter=float, validator=is_atleast(float, 0.0))
    num_hops = field(converter=int, validator=is_atleast(int, 0))
    on_gpu = field(converter=bool, validator=attrs.validators.instance_of(bool))

    def commstime(self, linkspeed: float) -> float:
        return self.payload / linkspeed


@define
class Resource:
    """Class representing an the properties of an assignable resource"""

    name: str
    cpu: field(validator=is_atleast(float, 0.0)) 
    gpu: field(converter=bool, validator=attrs.validators.instance_of(bool))
    mem: field(validator=is_atleast(float, 0.0)) 


@define(slots=False)
class Task:
    """A task request and if selected assignment."""

    source: str
    bits: field(default=10_000, validator=is_atleast(int, 0))
    cycles_per_bit: field(default=1, validator=is_atleast(int, 0))
    path: List[str] = field(default=None)
    _assigned: bool = False
    _dest: str = None

    @property
    def assigned(self):
        return self._assigned

    @property
    def dest(self):
        if self._assigned:
            return self._dest
        return None
    
    # @property
    # def path(self):
    #     if self._assigned:
    #         return self._path
    #     return None

    @dest.setter
    def dest(self, destination):
        self._assigned = True
        self._dest = destination
        
    # @path.setter
    # def path(self, path):
    #     self._assigned = True
    #     self._path = path


@define
class ResourceWorkflow:
    "A class to hold information about selected workflow on a specific resource"
    dest_name: str
    path: List[Tuple[str, str]]
    # inference_time: field(validator=is_atleast(float, 0.0))
    # comms_time: field(validator=is_atleast(float, 0.0))


class Scenario:
    def __init__(self, network=None, solver=None):
        self.solver = solver
        if network is None:
            self.network = nx.Graph()
        else:
            self.network = network
        self._solver = solver

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
                task_info["destination"] = task.dest
                task_info["path"] = task.path
                task_info["workflow_name"] = task.workflow_name
                task_info["cores_req"] = ceil(task.core_usage / 100)
                task_info["BW_alloc_bps"] = task.bandwidth_allocation
                task_info["assigned"] = task.assigned
                task_info["accuracy"] = task.accuracy
                task_info["priority"] = task.priority
                task_info["img_comp"] = task.image_compression
                task_info["model_comp"] = task.model_compression
                task_info["dev_type"] = task.device_type
                task_info["gpu"] = task.on_gpu
                if hasattr(task, "latency"):
                    task_info["task_execution_time"] = task.latency

            results.append(task_info)
        return df(results, index=range(len(self.tasks)))

    def stats_dataframe(self):
        return df(self.stats, index=[0])
