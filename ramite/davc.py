# -*- coding: utf-8 -*-
import logging
import pandas as pd
from collections.abc import Iterable
from math import ceil
import time

import random
from ramite.resources import Workflow

import numpy as np


#filter out the expired tasks in the running_tasks list
#called before new tasks and adaptations or resource updates
def filter_running_tasks(running_tasks):
    current_tasks = []
    logging.info(f"running tasks-time to completion:")
    now = time.time()
    for task in running_tasks:
        task.reassign = False
        if task.end_time > now:
            print(f"{task.id}: {task.end_time-now}\t{task.dest}\t{task.core_usage}")
            current_tasks.append(task)
    return current_tasks


def get_loe4_data(conn, query):
    search = conn.cursor()
    search.execute(query)
    results = search.fetchall()
    return results


def get_workflows_data(hops=None, bench_name = "staging-bench"):
    #bench_name = "workflows"
    result = pd.read_csv(f"ramite/queries/{bench_name}.csv")
    if bench_name=="workflows":#old benchmarks
        result = result.drop(columns=["table", "result"])
        result["bandwidth"] = result["bandwidth"].astype(float) * 1e6
        result["throughput"] = result["throughput"].astype(float) * 1e6
        result["mem_util"] = result["mem_util"].astype(float) * 1e6
        result["num_hops"] = result["num_hops"].astype(int)
        if hops is not None:
            result = result[result["num_hops"] == hops]

        workflows = {"UGS": [], "PLATOON": [], "APC": [], "CHQ": []}
        UGS_workflows = result[result['device_type'] == "UGS"].to_dict('records')
        offload_workflows = result[result['device_type'] == "APC"].to_dict('records')

        for workflow in UGS_workflows:
            workflows["UGS"].append(Workflow(**workflow))

        for workflow in offload_workflows:
            workflows["APC"].append(Workflow(**workflow))
            workflow['device_type'] = "PLT"
            workflows["PLATOON"].append(Workflow(**workflow))
            workflow['device_type'] = "CHQ"
            workflows["CHQ"].append(Workflow(**workflow))
        return workflows
    elif bench_name == "staging-bench":
        result.rename(
        columns={
            "sensor_cpu_requested_millicores": "sensor_core_usage",
            "requested_bandwidth_mbps": "bandwidth",
            "frame_duration_s": "exec_time",
            "classifier_cpu_requested_millicores": "core_usage",
            "avg_througput_fps": "throughput",
            "data_quality": "image_quality",
            "frame_queue_duration_s": "inference_duration",
            "model_load_duration_s": "model_load_time",
            "gpu": "on_gpu",
            "tranfer_duration_s": "get_image_duration",
        }, inplace=True)
        result["bandwidth"] = result["bandwidth"].astype(float) * 1e6
        result["throughput"] = result["throughput"].astype(float) * 1e6
        result["num_hops"] = result["num_hops"].astype(int)
        result["num_workers"] = result["num_workers"].astype(int)
        ##convert all model_quality to compression
        result["model_compression"] = 100 - result["model_quality"]
        #####for now memory requested same in all benchmarks
        ##curr bench do not limit memory in pods!!
        result["mem_util_sensor"] = 1000.0 * 1e6
        result["mem_util_workers"] = 3500.0 * 1e6

        # merge in acc: pulling this from es
        acc = pd.read_csv(f"ramite/queries/model_accuracy.csv")
        acc_s = acc["model_name"].str.split("_", n=1, expand=True)
        acc["model_type"] = acc_s[0]
        pd.set_option("display.max_rows", None)
        acc["model_compression"] = acc_s[1].str[:-1]
        acc["model_compression"] = acc["model_compression"].astype(int)
        acc["image_quality"] = acc["image_quality"].astype(int)
        acc.rename(columns={"top1": "acc"}, inplace=True)
        acc["model"] = acc["model_name"].str.split("_").str[0].str.lower()
        acc.loc[acc["model"].str.contains("mbnv2"), "model"] = "mobilenetv2"

        acc=acc[["model_compression", "image_quality","acc", "model"]]
        #logging.info(acc[["model_compression","image_quality","acc", "model"]].drop_duplicates())
        #TO_DO: fix exclude missing acc workflows
        result = pd.merge(result, acc, on=["model_compression","image_quality", "model"], how='right')
        #set task_type=model (image/audio)
        result["task_type"]="image"
        result.loc[result["model"].str.contains("acdnet"), "task_type"] = "audio"
        #result = pd.merge(result, acc, on=["model_compression","image_quality", "model"], how='left')
        #TO_DO replace workflow name with model later...
        result["name"]=result["model"]
        del acc
        #logging.info(f'''have img qual: {result[["model_compression","image_quality","acc", "model", "task_type"]].drop_duplicates()}''')
        result.drop(columns=['avg_latency_spf', 'network_usage_megabits_ps','source_file','uploaded_at','classifier_cpu_mcores_ps',
                            'model_quality','run_time_s','get_image_duration_s','latency_spf','network_usage_mb',
                            'network_usage_megabits_ps','network_usage_megabits_ps_2','num_frames'], inplace=True)
        if hops:
            result = result[result["num_hops"] == hops]
        workflows = {"UGS": [], "PLATOON": [], "APC": [], "CHQ": []}
        for _, row in result.iterrows():
            if row["case"] == "ondevice":
                row["device_type"] = "UGS"
                # logging.info(f'''ONDEVICE: {Workflow(**row.to_dict())}''')
                ####for ondevice mem_util_sensor==mem_util_workers
                row["mem_util_workers"] = 1000.0 * 1e6
                workflows["UGS"].append(Workflow(**row.to_dict()))
            elif row["case"] == "offload":
                for device_type in ["PLT","APC"]:  # Each of these device types have the same hardware in DAVC cluster(took out CHQ bc not placed there)
                    row["device_type"] = device_type
                    # logging.info(f'''OFFLOAD: {row}\n{Workflow(**row)}''')
                    workflows["PLATOON" if device_type == "PLT" else device_type].append(
                        Workflow(**row)
                    )
        return workflows


def generate_resource_data(err, amnt, loc, time, set_cpu, set_ugs_cpu):
    resource_query = pd.read_csv(
        "ramite/queries/resource_query.csv"
    )
    import warnings
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    # Note: add CPU randomness back later. For now, control through .csv
    if set_cpu is not None:
        #resource_query.loc[resource_query["scenario_node"].str[:4] != "UGS", "cpu_available"] = set_cpu
        resource_query.loc[~resource_query["scenario_node"].str.contains("UGS"), "cpu_available"] = set_cpu

    if set_ugs_cpu is not None:
        resource_query.loc[resource_query["scenario_node"].str.contains("UGS"), "cpu_available"] = set_ugs_cpu

    if err == 'cpu' and time == 'before':
        if loc in ["UGS","PLT","APC"]:
            err_loc = resource_query.index[resource_query['scenario_node'].str.startswith(loc)]
        elif loc == "rand":
            err_loc = random.randrange(1, len(resource_query))
        elif loc == "all":
            err_loc = range(len(resource_query))
        elif loc not in resource_query['scenario_node'].values:
            logging.warning(f"Invalid device \"{loc}\" specified, proceeding without CPU error...")
            err_loc = []
        else:
            err_loc = resource_query.index[resource_query['scenario_node']==loc]

        resource_query.loc[err_loc,"cpu_available"] *= float(amnt)
    return resource_query

def generate_bandwidth_data(err, amnt, loc, time, set_band):
    resource_query = pd.read_csv(
        "ramite/queries/mean_band.csv"
    )

    # Note: add BW randomness back later. For now, control through .csv
    if set_band != None:
        resource_query["bandwidth"] = resource_query["bandwidth"].apply(lambda x: set_band)

    if err == 'band' and time == 'before':
        if loc in ["UGS","PLT","APC"]:
            err_loc = resource_query.index[resource_query['edge'].str.contains(loc)]
        elif loc == "rand":
            err_loc = random.randrange(1, len(resource_query))
        elif loc == "all":
            err_loc = range(len(resource_query))
        elif loc not in resource_query['edge'].values:
            if loc is not None:
                loc = ','.join(loc.split(',')[::-1])
            if loc not in resource_query['edge'].values:
                logging.warning(f"Invalid edge \"{loc}\" specified, proceeding without BW error...")
                err_loc = []
            else:
                err_loc = resource_query.index[resource_query['edge']==loc]
        else:
            err_loc = resource_query.index[resource_query['edge']==loc]

        if amnt == 1:
            logging.warning("Error option selected, but no amount entered. Proceeding without BW error...")

        resource_query.loc[err_loc,"bandwidth"] *= float(amnt)

    return resource_query


def get_loe1_influxdata(error, erramnt, errloc, errtime, set_band, set_cpu, set_ugs_cpu):
    resource_query = generate_resource_data(error, erramnt, errloc, errtime, set_cpu, set_ugs_cpu)
    result = resource_query
    cpu_idle = dict(zip(result.scenario_node, result.cpu_usage_idle / 100.0))
    cpu_count = dict(zip(result.scenario_node, result.cpu_available))
    cpu_available = dict(zip(result.scenario_node, result.cpu_available))
    memory = dict(zip(result.scenario_node, result.memory_available))
    has_gpu = dict(zip(result.scenario_node, result.has_gpu))

    result = generate_bandwidth_data(error, erramnt, errloc, errtime, set_band)
    if result.empty:
        transfer_bps = {}
    else:
        transfer_bps = {
            tuple(e.split(",")): v for e, v in zip(result.edge, result.bandwidth)
        }

    results = {
        "memory": memory,
        "cpus": cpu_count,
        "cpu_available": cpu_available,
        "transfer_throughput": transfer_bps,
        "has_gpu": has_gpu,
    }
    if isinstance(results["has_gpu"], str):
        print(f'gpu boolean: {results["has_gpu"]}')

    return results