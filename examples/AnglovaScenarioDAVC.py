# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import queue
import time, math
import random
import shortuuid
import pandas as pd
import datetime as dt
from copy import deepcopy
from threading import Timer
from config import *

import networkx as nx
from tabulate import tabulate

from ramite.davc import (
    filter_running_tasks,
    get_loe1_influxdata,
    get_loe4_data,
    get_workflows_data)
from ramite.resources import (
    Resource,
    Scenario,
    Task,
    TaskAssignment,
)
from ramite.solvers import SOLVER_OPTIONS, MINISOLVER_OPTIONS

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def create_workflows(hops = None):
    inf_workflows = get_workflows_data(hops)
    return inf_workflows


LINK_CAPACITY = 10
LINK_THROUGHPUT = 70e6
GLOBAL_BANDWIDTH = 1e9
UGS_BANDWIDTH = 70e6
PLT_BANDWIDTH = 70e6
APC_BANDWIDTH = 70e6

NODE_COLORS = {
    "UGS": "brown",
    "PLATOON": "cornflowerblue",
    "APC": "darkblue",
    "CHQ": "blue",
}

NODE_TYPE_MAPPING = {
    "UGS": "UGS",
    "PL0": "PLATOON",
    "PLT": "PLATOON",
    "APC": "APC",
    "CHQ": "CHQ",
}


def make_df(self, assignments, stats, mean_execution_time_per_frame,mean_execution_time_per_frame_w_overhead):
    os.makedirs("results", exist_ok=True)
    results_file_name = "task_placement.csv"
    #unset header for existing csv output
    header = True
    if os.path.isfile(f"results/{results_file_name}"):
        header = False

    output = assignments.placement_dataframe()
    assignment_columns = pd.DataFrame(columns=['source', 'id', 'destination', 'path', 'workflow_name',
   'cores_requested', 'BW_alloc_bps', 'assigned', 'accuracy', 'priority',
   'img_qual', 'model_comp', 'dev_type', 'gpu', 'cpu_deg', 'bwp_deg',
   'fps_o', 'req_acc', 'timeliness', 'num_tasks', 'task_execution_time',
   'task_inference_time', 'num_images', 'result'])
    output = pd.concat([assignment_columns,output], ignore_index=True, sort=False)
    output['seed'] = self.seed
    output['lambda'] = self.scenario.solver.obj_lambda
    output['max_hops'] = self.scenario.solver.max_hops
    output['required_accuracy'] = self.req_acc
    output['cores_available'] = self.set_cpu
    output['req_lat'] = self.req_lat
    output['num_images'] = self.num_images
    output['set_band'] = self.set_band
    output['num_tasks'] = self.num_tasks
    output['num_hops'] = self.num_hops
    output['result'] = assignments.stats["result"]
    output.to_csv(f"results/{results_file_name}", header=header,  mode="a")

    output = pd.DataFrame.from_dict(stats,orient='index').T
    output['seed'] = self.seed
    output['lambda'] = self.scenario.solver.obj_lambda
    output['max_hops'] = self.scenario.solver.max_hops
    output['mean_execution_time_per_frame'] = mean_execution_time_per_frame
    output['mean_execution_time_per_frame_w_overhead'] = mean_execution_time_per_frame_w_overhead
    output['required_accuracy'] = self.req_acc
    output['cores_available'] = self.set_cpu
    output['req_lat'] = self.req_lat
    output['num_images'] = self.num_images
    output['set_band'] = self.set_band
    output['num_tasks'] = self.num_tasks
    output['num_hops'] = self.num_hops
    output['result'] = assignments.stats["result"]
    output.to_csv(f"results/avg_{results_file_name}", header=header, mode="a")

def name_to_type(name, mapping):
    prefix = name.split("-")[0]
    return mapping.get(prefix, None)


def generate_tasks(payload):
    tasks = []
    task_list = json.loads(payload)
    num_tasks = len(task_list)
    for task in task_list:
        a_task = Task(
            task["dataLoc"],
            id=task.get("id", ""),
            num_images=task.get("num_images", 1),
            timeliness=float(task["timeliness"]),
            req_acc=float(task["req_acc"]),
            priority=1 / num_tasks,
            cpu_deg=task["cpu_deg"],
            bwp_deg=task["bwp_deg"],
            type = task["type"]
        )
        if task.get("num_images") is not None:
            a_task.num_images = task["num_images"]
        tasks.append(a_task)
    return tasks


class DAVCTaskAllocator():
    def __init__(
        self,
        update_interval=60,
        submit_tasks=False,
        adjust_workflows=False,
        solver_name="intertask",
        minisolver_name="minisolver",
        add_error='none',
        err_amnt='0',
        err_loc='none',
        err_time='none',
        set_band='none',
        set_cpu='none',
        set_ugs_cpu='none',
        req_acc='none',
        req_lat='none',
        num_images='none',
        num_tasks='none',
        seed=None,
        num_hops = None,
        obj_lambda = None,
    ):

        self.add_error = add_error
        self.err_amnt = err_amnt
        self.err_loc = err_loc
        self.err_time = err_time
        self.set_band = set_band
        self.update_interval = update_interval
        self.submit_tasks = submit_tasks
        self.adjust_workflows = adjust_workflows
        self.shutdown = False
        self.queue = queue.Queue()
        self.set_cpu = set_cpu
        self.set_ugs_cpu = set_ugs_cpu
        self.req_acc = req_acc
        self.req_lat = req_lat
        self.num_images = num_images
        self.num_tasks = num_tasks
        self.seed = seed
        self.num_hops = num_hops
        self.obj_lambda = obj_lambda

        self.parameters = {
            "allocator": {
                "update_interval": self.update_interval,
                "submit_tasks": self.submit_tasks,
            },
            "network": {
                "workflows": create_workflows(self.num_hops),
                "link_capacity": LINK_CAPACITY,
                "link_throughput": LINK_THROUGHPUT,
                "global_bandwidth": GLOBAL_BANDWIDTH,
                "node_colors": NODE_COLORS,
                "node_type_mapping": NODE_TYPE_MAPPING,
            },
            "solver": {
                "obj_lambda": self.obj_lambda,
                "max_switches": 6,
                "precision": 1,
                "max_hops": config_num_hops,
                "solver_name": solver_name,
                "minisolver_name": minisolver_name,
                "use_cache": False,
            },
        }

        self.create_scenario()

        self.update_timer = None
        if self.update_interval:
            self.update_timer = Timer(self.update_interval, self.schedule_update)
            self.update_timer.start()

    def schedule_update(self):
        logging.debug("Enqueuing Update")
        self.queue.put(("UpdateResources", None))


    def set_params(self, payload, properties=None):
        params = json.loads(payload)
        recreate_scenario = False
        for subkey, updates in params.items():
            if subkey == "solver":
                obj = self.scenario.solver
            elif subkey == "minisolver":
                obj = self.scenario.minisolver
            elif subkey == "network":
                obj = self
                recreate_scenario = True
            else:
                obj = self
            for param, new_value in updates.items():
                if subkey == "solver" and (
                    param == "solver_name" or param == "minisolver_name"
                ):
                    recreate_scenario = True
                elif hasattr(obj, param):
                    setattr(obj, param, new_value)
            self.parameters[subkey].update(updates)
        if recreate_scenario:
            self.create_scenario()
        return self.parameters

    def get_params(self, payload=None, properties=None):
        return self.parameters

    def create_network(self):
        logging.debug("received payload")
        loe1_data = get_loe1_influxdata(self.add_error, self.err_amnt, self.err_loc, self.err_time, self.set_band, self.set_cpu, self.set_ugs_cpu)
        logging.debug("LOE1 Data Received")

        memory = loe1_data["memory"]
        cpu_cores = loe1_data["cpu_available"]
        has_gpu = loe1_data["has_gpu"]
        transfer_throughput = loe1_data["transfer_throughput"]
        throughput = transfer_throughput
        compute_capacity = {k: 200 if k.startswith("CHQ") else 100 for k in cpu_cores}

        # Get a copy of the network parameters
        network_parameters = deepcopy(self.parameters["network"])
        workflows = network_parameters["workflows"]
        link_throughput = network_parameters["link_throughput"]
        global_bandwidth = network_parameters["global_bandwidth"]
        link_capacity = network_parameters["link_capacity"]
        node_colors = network_parameters["node_colors"]
        node_type_mapping = network_parameters["node_type_mapping"]

        # check to make sure all of our dictionarys have the exact same set of keys
        cpuset = set(cpu_cores.keys())
        memoryset = set(memory.keys())
        computeset = set(compute_capacity.keys())
        edgeset = set(throughput.keys())
        if not cpuset == memoryset == computeset:
            raise ValueError(
                "All resources must have entries for memory, cpu, and compute_capacity"
            )

        network = nx.Graph()
        for node in cpuset:
            node_type = name_to_type(node, node_type_mapping)
            pd.options.display.max_columns = None
            if node_type is None:
                raise ValueError(f"Cannot map resource {node} to a known type")

            resource = Resource(
                node,
                [
                    w
                    for w in workflows[node_type]
                    if (w.on_gpu == False) or (w.on_gpu == has_gpu[node])
                ],
                compute_capacity[node],
                cpu_cores[node],
                memory[node],
                has_gpu[node],
            )
            network.add_node(
                node, resource=resource, color=node_colors.get(node_type, "red")
            )

        missing_link = False
        for src, dst in edgeset:
            if (src, dst) not in throughput and (dst, src) not in throughput:
                missing_link = True
                logging.debug(
                    f"Missing edge {(src, dst)} in throughtput information"
                    f"substituting {link_throughput} as default"
                )
                throughput[(src, dst)] = link_throughput
            elif (src, dst) not in throughput:
                # Ignore directionallity
                throughput[(src, dst)] = throughput[(dst, src)]
            if throughput[(src, dst)] == 0:
                logging.warning(
                    f"Edge {(src, dst)} has throughput of 0. Setting to 1 to avoid div by zero"
                )
                throughput[(src, dst)] = 1
            attributes = {
                "link_capacity": link_capacity,
                "link_speed": throughput[(src, dst)],
            }
            if src in cpuset and dst in cpuset:
                network.add_edge(src, dst, **attributes)
        if missing_link:
            logging.warning("Missing links in throughput information")
        return network

    def create_scenario(self, payload=None, properties=None):
        anglova_network = self.create_network()
        solver_params = self.parameters["solver"]
        network_params = self.parameters["network"]
        if solver_params["solver_name"] not in SOLVER_OPTIONS:
            raise ValueError(f"Unsupported solver: {solver_params['solver_name']}")
        elif solver_params["minisolver_name"] not in MINISOLVER_OPTIONS:
            raise ValueError(
                f"Unsupported minisolver: {solver_params['minisolver_name']}"
            )
        else:
            solver_cls = SOLVER_OPTIONS[solver_params["solver_name"]]
            anglova_solver = solver_cls(
                max_hops=solver_params["max_hops"],
                precision=solver_params["precision"],
                obj_lambda=solver_params["obj_lambda"],
                max_switches=solver_params["max_switches"],
                use_cache=solver_params["use_cache"],
                global_bandwidth=network_params["global_bandwidth"],
            )
            minisolver_cls = MINISOLVER_OPTIONS[solver_params["minisolver_name"]]
            anglova_minisolver = minisolver_cls(
                max_hops=solver_params["max_hops"],
                precision=solver_params["precision"],
                obj_lambda=solver_params["obj_lambda"],
                use_cache=solver_params["use_cache"],
                global_bandwidth=network_params["global_bandwidth"],
            )
            self.scenario = Scenario(
                anglova_network, anglova_solver, anglova_minisolver
            )

    def place_tasks(self, payload):
        if isinstance(payload, bytes):
            task_requests = generate_tasks(payload)
        elif len(payload) > 0 and isinstance(payload[0], Task):
            task_requests = payload
        else:
            task_requests = generate_tasks(payload)
        #    return "Invalid submission to place tasks"
        save_max_switches = self.scenario.solver.max_switches
        switch_options = [save_max_switches]

        #remove comleted tasks from running_tasks before placing new tasks
        self.scenario.running_tasks = filter_running_tasks(self.scenario.running_tasks)
        #logging.info(f"currently running tasks: {[t.id for t in self.scenario.running_tasks]}")
        if self.scenario.running_tasks:
            switch_options.insert(0, 0)
        for idx, switch in enumerate(switch_options):
            self.scenario.solver.max_switches = switch
            logging.info(f"Placing Tasks: solver_type={self.parameters['solver']['solver_name']};obj_lambda={self.parameters['solver']['obj_lambda']};switch:{switch}")
            start_time = time.process_time()
            (
                result,
                solver_time,
                placement,
                acc,
                mean_execution_time,
                mean_execution_time_per_frame,
                mean_execution_time_per_frame_w_overhead,
            ) = self.scenario.place_tasks(task_requests)
            if result != 2:  # if an optimal/feasable solution
                    break
            if result == 2 and len(switch_options) == idx+1 and self.parameters["solver"]["solver_name"]=="intertask":
                #attempt to place with max_switches and intertaskdrop solver bc intertask didin't work:
                self.set_params(payload=f"{{\"solver\":{{\"obj_lambda\": 0.999,\"solver_name\": \"intertask\"}}}}")
                logging.info(f"Placing Tasks: solver_type={self.parameters['solver']['solver_name']};obj_lambda={self.parameters['solver']['obj_lambda']};switch:{switch}")
                (
                    result,
                    solver_time,
                    placement,
                    acc,
                    mean_execution_time,
                    mean_execution_time_per_frame,
                    mean_execution_time_per_frame_w_overhead,
                ) = self.scenario.place_tasks(task_requests)
                self.set_params(payload=f"{{\"solver\":{{\"obj_lambda\": 0.9997,\"solver_name\": \"intertask\"}}}}")
                break
            if len(switch_options) > 1:
                logging.info("Solver considering re-allocation of running tasks")
        self.scenario.solver.max_switches = save_max_switches
        elapsed_time = time.process_time() - start_time
        stats = {
            "result": result,
            "task_num": 1,
            "total_tasks": len(placement),
            "mean_accuracy": acc,
            "mean_exectime_s": mean_execution_time,
            "solver_time_s": solver_time,
            "placement_time_s": elapsed_time,
        }
        assignments = TaskAssignment(placement, stats)
        logging.info("Optimization Complete")

        #this writes the palacement csv's
        make_df(self,assignments,stats,mean_execution_time_per_frame,mean_execution_time_per_frame_w_overhead)

        #deal with no solution to placement case
        if assignments.stats["result"] == 2:
            logging.info("Not enough resources to place tasks=No Solution")
            return None
        else:#solution case
            logging.info("Task assignments")
            logging.info("\n" + tabulate(
                assignments.placement_dataframe().sort_values(by="source"),
                headers="keys"))
            logging.info("Task placement stats")
            logging.info("\n" + tabulate(assignments.stats_dataframe(), headers="keys"))
            #add the new placed tasks to running tasks: this is to count them when placing new tasks in the future
            #account already running tasks that were switched
            running_task_dict = {t.id:t for t in self.scenario.running_tasks}
            logging.info(f"running tasks: {running_task_dict.keys()}")
            placed_task_dict = {t.id:t for t in placement if t.dest}#filter out un-placed tasks..nan
            switched_running_tasks=set(running_task_dict.keys()).intersection(set(placed_task_dict.keys()))
            for id in switched_running_tasks:
                if running_task_dict[id].destination is not placed_task_dict[id].destination:
                    print(f"switched task: {id}")
                del running_task_dict[id]
            placed = [t for t in placement if t.dest]
            self.scenario.running_tasks = placed+list(running_task_dict.values())
            logging.info(f"new running tasks: {[(t.id, t.dest) for t in self.scenario.running_tasks]}")
        return assignments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        type=int,
        help="Number of tasks to send (per UGS if -r is False)",
        default=1,
    )
    parser.add_argument(
        "--num_seq_sub",
        type=int,
        help="Number in sequence of submissions in time of a the total group tasks",
        default=1,
    )
    parser.add_argument("-f", type=str, help="JSON file to use", default="")
    parser.add_argument(
        "-seed", type=int, help="Seed for random task assignment", default=None
    )
    parser.add_argument(
        "--rand", action="store_true", help="Toggle random assignment", default=False
    )
    parser.add_argument(
        "--response",
        action="store_true",
        help="Request a response with the allocation",
        default=False,
    )
    parser.add_argument(
        "--acc", type=float, help="Set minimum accuracy requirement", default=60
    )
    parser.add_argument(
        "--lat", type=float, help="Set maximum latency requirement (seconds)", default=1
    )
    parser.add_argument(
        "--num-ugs",
        type=float,
        help="The total number of UGS in the scenario",
        default=30,
    )
    parser.add_argument(
        "--num-images",
        type=int,
        help="The number of images to process per task request",
        default=10000,
    )
    parser.add_argument(
        "-update-interval",
        help="period to update resource information (secs)",
        type=int,
        default=60,
    )
    parser.add_argument(
        "-adjust-workflows",
        help="adjust workflows to add artificial restrictions on nodes",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-solver", type=str, help="name of the solver to use", default="intertask"#"intertaskdrop"
    )
    parser.add_argument(
        "-minisolver",
        type=str,
        help="name of the minisolver to use",
        default="minisolver",
    )
    parser.add_argument(
        "-submit-tasks",
        help="send allocation result to task-manager to run tasks",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-logging",
        type=str,
        help="logging level",
        default="INFO",
        choices=["DEBUG", "ERROR", "INFO", "WARNING"],
    )
    parser.add_argument(
        "-error",
        type=str,
        help="add error to placement",
        default=None,
        choices=["none", "cpu", "lat", "band"],
    )
    parser.add_argument(
        "-task_cpu",
        type=float,
        help="add resource degredation to cpu assignment of tasks [0-1], where 1 is no degredation",
        default=1.0,
    )
    parser.add_argument(
        "-task_bwp",
        type=float,
        help="add resource degredation to bandwidth assignment of tasks[0-1], where 1 is no degredation",
        default=1.0,
    )
    parser.add_argument(
        "-erramnt",
        type=float,
        help="the percentage of error to add, where 1 is no degredation",
        default=1.0,
    )
    parser.add_argument(
        "-errloc",
        type=str,
        help="the location to add the error",
        default=None,
    )
    parser.add_argument(
        "-errtime",
        type=str,
        help="whether the error happens before or after placement",
        default=None,
    )
    parser.add_argument(
        "-set_band",
        type=float,
        help="sets a specific link rate for all edges",
        default=None
    )
    parser.add_argument(
        "-set_cpu",
        type=float,
        help="sets a specific core usage for none ugs nodes",
        default=None
    )
    parser.add_argument(
        "-set_ugs_cpu",
        type=float,
        help="sets a specific core usage for ugs nodes",
        default=None
    )
    parser.add_argument(
        "-set_num_hops",
        type=str,
        help="sets a fixed number of hops",
        default="None"
    )
    parser.add_argument(
        "-obj_lambda",
        type=float,
        help="sets current solver objective lambda",
        default=0.9997
    )
    parser.add_argument(
        "--first-ugs",
        type=int,
        help="The first UGS in the scenario",
        default=0,
    )
    parser.add_argument(
        "--last-ugs",
        type=int,
        help="The last UGS in the scenario",
        default=None,
    )


    args = parser.parse_args()
    num_tasks = args.n
    seq = args.num_seq_sub
    file_name = args.f
    seed = args.seed
    randFlag = args.rand
    req_acc = args.acc
    timeliness = args.lat
    num_images = args.num_images
    task_cpu_deg = args.task_cpu
    task_bwp_deg = args.task_bwp
    set_band = args.set_band
    set_cpu = args.set_cpu
    set_ugs_cpu = args.set_ugs_cpu


    random.seed(seed)


    formatter = "%(asctime)s - %(filename)s:%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=args.logging, format=formatter)

    logging.debug("Arguments Parsed")
    allocator = DAVCTaskAllocator(
        update_interval=args.update_interval,
        submit_tasks=args.submit_tasks,
        adjust_workflows=args.adjust_workflows,
        solver_name=args.solver,
        minisolver_name=args.minisolver,
        num_hops=None if args.set_num_hops=="None" else int(args.set_num_hops),
        obj_lambda=args.obj_lambda,
        add_error=args.error,
        err_amnt=args.erramnt,
        err_loc=args.errloc,
        err_time=args.errtime,
        set_band=args.set_band,
        set_cpu=args.set_cpu,
        set_ugs_cpu=args.set_ugs_cpu,
        req_acc=args.acc,
        req_lat=args.lat,
        num_images=args.num_images,
        num_tasks=args.n,
        seed=seed,
    )


    
    # generate tasks
    """START TASK ALLOCATOR"""
    if args.last_ugs == None:
        last_ugs = -1
        for node in allocator.scenario.network.nodes():
            if node.split("-")[0] == 'UGS':
                last_ugs += 1
    else:
        last_ugs = args.last_ugs
    
    ugs_list = [f"UGS-{i}" for i in range(args.first_ugs, last_ugs + 1)]
    taskObj = {}

    taskObj["req_acc"] = req_acc
    taskObj["timeliness"] = timeliness
    taskObj["num_images"] = num_images
    taskObj["cpu_deg"] = task_cpu_deg
    taskObj["bwp_deg"] = task_bwp_deg

    task_list_seq = [[] for i in range(seq)]
    task_seq = math.floor(num_tasks/seq)
    sub_counter=0
    task_list = []
    if randFlag:
        for idx, source in enumerate(random.sample(ugs_list, k=num_tasks)):
            task = taskObj.copy()
            task["dataLoc"] = source
            if idx>0 and idx%task_seq==0 and len(random.sample(ugs_list, k=num_tasks)) > idx+1:
                sub_counter += 1
            task["id"] = f"{str(sub_counter)}_{shortuuid.uuid().lower()}"
            task["type"] = "image"
            task_list_seq[sub_counter].append(task)

    """END TASK ALLOCATOR"""
    #This is where you submit tasks using the seq param
    #define how long and compute intensive they are
    #time.sleep just spreads the task submission
    
    for s in range(seq):
        print(f"Tasking Sent batch: {s}")
        allocator.place_tasks(json.dumps(task_list_seq[s]))
        logging.info("sleep between task requests----")
        time.sleep(25)


if __name__ == "__main__":
    main()
