**CPU queries processing for Solver input**
1. gets "cpu_usage_millicores" and memory_available per node
```
cpu_usage = get_telegraf_usage()
```

2. get kubernetes_node capacity cores: gets "allocatable_millicpu_cores","capacity_millicpu_cores" per node
```
cpu_alloc = get_telegraf_node()
```

3. calculate current cpu available using cpu_usage
```
cpu_usage_av=cpu_alloc["capacity_millicpu_cores"]-cpu_usage["cpu_usage_millicores"]
```

4. calculate available cpu from pod requests for each node:
First get and aggregare all the running pod requests for "resource_requests_millicpu_units" by node (account max reservations for running tasks and kubernetes overhead)
```
cpu_requested_av=cpu_alloc["allocatable_millicpu_cores"]-pod_reqs["resource_requests_millicpu_units"]
```

5. Finally get our Solver cpu available by getting the min of reservations or current usage:
```
cpu_available=min(cpu_request_av,cpu_usage_av)
```

##### ```cpu_available for each node``` -> input to the solver cpu constraint.
##### We query ```get_network_data()``` to get all edges bw -> the solver bw constraint.
************************************************************************************************************


**BELOW are the queries used to get RAMITE resources and RAMITE task info:**


##### Gets current running task data for each task (by task_id): basicly task's recent speed(frame_duration): frame_duration is a factor of "frame_getImageDuration" and "frame_inferenceDuration": We need "frame_getImageDuration" and "frame_inferenceDuration" to determine if a task is not meeting it's compute or bw benchmarks, so it can be adapted.
```
def get_task_data(self, task_id, log=True):
        query = {
                "bool": 
                {
                "must": 
                [
                    {
                    "match": {
                        "task_id.keyword": f"{task_id}"
                    }
                    },
                    {      
                    "range": 
                    {
                        "@timestamp": 
                        {
                        "gte": "now-30s",
                        "lt": "now"
                        }
                    }
                    }
                ]
                }
        }
        aggs = {
                "current_frame_number": {"max": {"field": "frame_number"}},
                "avg_frame_getImageDuration": {"avg": {"field": "frame_getImageDuration"}},
                "avg_frame_inferenceDuration": {"avg": {"field": "frame_inferenceDuration"}},
                "avg_frame_duration": {"avg": {"field": "frame_duration"}}
        }
        resp = self.client.search(index="task_data", size=1 , query=query, aggs=aggs)
        result = pd.json_normalize(resp["aggregations"], max_level=0)
        return result
``` 

#### Get current avg cpu_usage of the task pod on the node..this get requeried every time we get new task requests or adaptations. Sometimes other running tasks need to be reallocated due to a alow task. Setting range to less than 15 seconds results in no records sometimes..so we get the latest size records in the last minute instead
```
def get_task_cpu_data(self, task_id, log=True):
        query = {
            "bool": 
            {
            "must": 
            [
                {
                "range": {
                    "@timestamp": {
                    "gte": "now-1m",
                    "lt": "now"
                    }
                }
                },
                {
                "regexp": {
                    "tag.pod_name.keyword": f"classifier-{task_id}.*"
                }
                },
                {
                "exists": {"field": "kubernetes_pod_container.cpu_usage_nanocores"}}]}
        }
        sort= [{"@timestamp": {"order": "desc"}}]
        aggs = {"avg_cpu_usage": {"avg": {"field": "kubernetes_pod_container.cpu_usage_nanocores"}}}
        result = self.client.search(index="telegraf", size=6, query=query, sort=sort, aggs=aggs)
        #result = pd.json_normalize(result["aggregations"], max_level=0)
        return result
```

#### Get last 30s-1 minute avg bw_usage of the task edge (sensor to server)..this gets requeried every time we get new task requests or adaptations.    
```
def get_task_bw(self, task_id):
        query ={
                    "bool": {
                    "filter": {
                        "exists": {
                        "field": "kubernetes_pod_network.tx_bytes"
                        }
                    },
                    "must": [
                        {"range": {
                            "@timestamp": {
                            "gte": "now-30s",
                            "lt": "now"
                            }}},
                        {"regexp": {
                            "tag.pod_name.keyword": {
                            "value": f"sensor-{task_id}.*"}}}
                    ]}
                }
        aggs={
                "max_bw": {
                    "max": {
                        "field": "kubernetes_pod_network.tx_bytes"
                    }
                    },
                "min_bw": {
                    "min": {
                        "field": "kubernetes_pod_network.tx_bytes"
                    }
                    },
                "max_time": {
                    "max": {
                        "field": "@timestamp"
                    }
                    },
                "min_time": {
                    "min": {
                            "field": "@timestamp"
                    }
            }
        }
        resp = self.client.search(index="telegraf", size=3 , query=query, aggs=aggs)
        return resp
```

#### Get most recent latency in ElasticSearch per edge in our RAMITE network: done during update_resources ~30s or during new task requests/adaptation calls 
```
def get_latency_data(self, log=True):
        size = 100
        query={
                "bool": {
                "must": [
                    {
                    "range": {
                        "timestamp": {
                        "gte": "now-6m",
                        "lt": "now"
                        }
                    }
                    },
                    {
                    "exists": {
                        "field": "latency"
                    }
                    }
                ]
                }
            }
        collapse={
                "field": "edge.keyword",
                "inner_hits": {
                "name": "most_recent",
                "size": 1,
                "sort": [
                    {
                    "timestamp": "desc"
                    }
                ]
                }
            }
        resp = self.client.search(index="neighbors", size=size , query=query, collapse=collapse)
        return resp
```

#### Get most recent bw in ElasticSearch per edge in our RAMITE network: done during update_resources-every 30s or during new task requests/adaptation calls 
```
def get_network_data(self, log=True):
        size = 100
        query={
                "bool": {
                "must": [
                    {
                    "range": {
                        "timestamp": {
                        "gte": "now-6m",
                        "lt": "now"
                        }
                    }
                    },
                    {
                    "exists": {
                        "field": "bandwidth"
                    }
                    }
                ]
                }
            }
        collapse={
                "field": "edge.keyword",
                "inner_hits": {
                "name": "most_recent",
                "size": 1,
                "sort": [
                    {
                    "timestamp": "desc"
                    }
                ]
                }
            }
        resp = self.client.search(index="neighbors", size=size , query=query, collapse=collapse)
        return resp
```

#### Get a recent per node cpu_usage for RAMITE network: done during update_resources ~30s or during new task requests/adaptation calls 
```
def get_telegraf_usage(self, log=True):
        size = 100
        query = {
            "bool": {
            "must": [{
                "range": {
                    "@timestamp": {
                    "gte": "now-3m",
                    "lt": "now"
                    }}},{"exists": { "field": "kubernetes_node.cpu_usage_nanocores" }}]}}
        collapse={
            "field": "tag.scenario_node.keyword",
            "inner_hits": {
            "name": "most_recent",
            "size": 1,
            "sort": [{"@timestamp": "desc"}]}}
        resp = self.client.search(index="telegraf", size=size , query=query, collapse=collapse)
        return resp
```
    
#### Get per node running pod allocatable_cores for RAMITE network: done during update_resources ~30s or during new task requests/adaptation calls. "allocatable_millicpu_cores" comes from global telegraf
```
def get_telegraf_node(self, log=True):
        size = 100
        query = {
            "bool": {
            "must": [{
                "range": {
                    "@timestamp": {
                    "gte": "now-4m",
                    "lt": "now"
                    }}},{"exists": { "field": "kubernetes_node.allocatable_millicpu_cores" }}]}}
        collapse={
            "field": "tag.scenario_node.keyword",
            "inner_hits": {
            "name": "most_recent",
            "size": 1,
            "sort": [{"@timestamp": "desc"}]}}
        resp = self.client.search(index="telegraf", size=size , query=query, collapse=collapse)
        return resp
```

#### cpu_pod_requests query: kubernetes_pod_container.resource_requests_millicpu_units comes from global telegraf
```
def get_telegrapf_pod(self, log=True):
        size = 100
        query = {
            "bool": {
            "filter": {
                "term" : { "tag.state" : "running" }
            },
            "must": [{
                "range": {
                    "@timestamp": {
                    "gte": "now-2m",
                    "lt": "now"
                    }}},
                {"exists": {"field": "kubernetes_pod_container.resource_requests_millicpu_units"}}
        ]}}
        collapse = {
            "field": "tag.pod_name.keyword",
            "inner_hits": {
            "name": "most_recent",
            "size": 1,
            "sort": [{"@timestamp": "desc"}]
            }
        }
        resp = self.client.search(index="telegraf", size=size , query=query, collapse=collapse)
        return res
```