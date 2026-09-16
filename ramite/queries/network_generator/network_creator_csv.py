# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 15:48:16 2023

@author: jake.b.perazzone

This file generates (random) networks for RA problems

If random_network = False, then create_anglova_network() is called which
recreates our Anglova47 network with resources (in ALL CAPS) set below.

If random_network = True, then a random network topology generator is called.
Currently only geo_topology() exists. This creates a random network with some
properties set via the config.py file (like node density). It randomly places
n nodes of type (UGS,PLT,APC) in a NxN plane where location is uniformly chosen
and type is dictated via a draw from a categorical distribution with hard coded
parameters of (.7,.2,.1). Then connectivity is determine by distance where APCs
can connect to farther away nodes compared to UGS-UGS connections, for example.
The distance threshold that determines this is hard-coded as [.5,1,2.5], but
can be changed. Finally, resources are assigned to each node randomly within 
the ranges set in the config_csv.py file.

The final network is outputted to a .csv file for use in the standalone solver.
"""

import networkx as nx
from resources import Resource
from collections import defaultdict
import numpy as np
from config_csv import *
import csv


LINK_THROUGHPUT = 70e6
UGS_THROUGHPUT = 70e6
PLT_THROUGHPUT = 70e6
APC_THROUGHPUT = 70e6

UGS_CPU = 2000
PLT_CPU = 6000
APC_CPU = 6000

EDGES = [
    ("CHQ-1", "APC-0"),
    ("CHQ-1", "APC-1"),
    ("CHQ-1", "APC-2"),
    ("CHQ-1", "APC-3"),
    ("APC-0", "APC-1"),
    ("APC-0", "APC-2"),
    ("APC-0", "APC-3"),
    ("APC-0", "PLT-0"),
    ("APC-0", "PLT-1"),
    ("APC-0", "PLT-2"),
    ("APC-0", "PLT-3"),
    ("APC-0", "PLT-4"),
    ("APC-0", "UGS-0"),
    ("APC-0", "UGS-11"),
    ("APC-1", "APC-2"),
    ("APC-1", "APC-3"),
    ("APC-1", "PLT-5"),
    ("APC-1", "PLT-6"),
    ("APC-1", "PLT-7"),
    ("APC-1", "UGS-12"),
    ("APC-1", "UGS-18"),
    ("APC-2", "APC-3"),
    ("APC-2", "PLT-8"),
    ("APC-2", "PLT-9"),
    ("APC-2", "PLT-10"),
    ("APC-2", "UGS-19"),
    ("APC-2", "UGS-27"),
    ("APC-3", "PLT-11"),
    ("PLT-0", "PLT-1"),
    ("PLT-0", "PLT-2"),
    ("PLT-0", "PLT-3"),
    ("PLT-0", "PLT-4"),
    ("PLT-0", "UGS-0"),
    ("PLT-0", "UGS-1"),
    ("PLT-1", "PLT-2"),
    ("PLT-1", "PLT-3"),
    ("PLT-1", "PLT-4"),
    ("PLT-1", "UGS-2"),
    ("PLT-2", "PLT-3"),
    ("PLT-2", "PLT-4"),
    ("PLT-2", "UGS-3"),
    ("PLT-3", "PLT-4"),
    ("PLT-3", "UGS-4"),
    ("PLT-3", "UGS-5"),
    ("PLT-3", "UGS-6"),
    ("PLT-3", "UGS-7"),
    ("PLT-4", "UGS-7"),
    ("PLT-4", "UGS-8"),
    ("PLT-4", "UGS-9"),
    ("PLT-4", "UGS-10"),
    ("PLT-5", "PLT-6"),
    ("PLT-5", "PLT-7"),
    ("PLT-5", "UGS-12"),
    ("PLT-5", "UGS-14"),
    ("PLT-5", "UGS-15"),
    ("PLT-5", "UGS-16"),
    ("PLT-5", "UGS-17"),
    ("PLT-6", "PLT-7"),
    ("PLT-6", "UGS-16"),
    ("PLT-6", "UGS-17"),
    ("PLT-7", "UGS-13"),
    ("PLT-7", "UGS-18"),
    ("PLT-8", "PLT-9"),
    ("PLT-8", "PLT-10"),
    ("PLT-8", "UGS-19"),
    ("PLT-8", "UGS-20"),
    ("PLT-8", "UGS-27"),
    ("PLT-9", "PLT-10"),
    ("PLT-9", "UGS-21"),
    ("PLT-9", "UGS-22"),
    ("PLT-9", "UGS-27"),
    ("PLT-10", "UGS-22"),
    ("PLT-10", "UGS-23"),
    ("PLT-10", "UGS-24"),
    ("PLT-10", "UGS-25"),
    ("PLT-10", "UGS-26"),
    ("PLT-10", "UGS-27"),
    ("PLT-11", "UGS-28"),
    ("PLT-11", "UGS-29"),
]


NODE_COLORS = {
    "UGS": "red",
    "PLATOON": "cornflowerblue",
    "PLT": "green",
    "APC": "yellow",
    "CHQ": "blue",
}

NODE_SIZES = {
    "UGS": 100,
    "PLT": 200,
    "APC": 400,
    "CHQ": 400,
}

NODE_TYPE_MAPPING = {
    "UGS": "UGS",
    "PL0": "PLT",
    "PLT": "PLT",
    "APC": "APC",
    "CHQ": "CHQ",
}


NODES = set([item for sublist in EDGES for item in sublist])


def create_anglova_network():
    
    def name_to_type(name, mapping):
        prefix = name.split("-")[0]
        return mapping.get(prefix, None)
    
    def name_to_index(name):
         return list(network.nodes).index(name)

    nodes = NODES
    edges = EDGES

    network = nx.Graph()
    for node in nodes:
        node_type = name_to_type(node, NODE_TYPE_MAPPING)
        if node_type is None:
            raise ValueError(f"Cannot map resource {node} to a known type")
    
    
        if node.startswith("UGS"):
            cpu_freq = UGS_CPU
            # memory = 4e9
        elif node.startswith("PLT"):
            cpu_freq = PLT_CPU
            # memory = 8e9
        else: # node.startswith("APC"):
            cpu_freq = APC_CPU
            # memory = 16e9
    
        resource = Resource(node, cpu_freq)
        network.add_node(
            node,
            resource=resource,
            color=NODE_COLORS.get(node_type, "red"),
            size=NODE_SIZES.get(node_type, 100),
        )
    
    for src, dst in edges:
        
        if src.startswith("UGS") or dst.startswith("UGS"):
            link_throughput = UGS_THROUGHPUT
        elif src.startswith("PLT") or dst.startswith("PLT"):
            link_throughput = PLT_THROUGHPUT
        else:
            link_throughput = APC_THROUGHPUT
        
        
        attributes = {
            "link_speed": link_throughput,
        }
        network.add_edge(src, dst, **attributes)
    
    return network

def assign_resources(network):
    
    node_cpu_bounds = {'UGS': {'lower':UGS_CPU_LOWER, 'upper': UGS_CPU_UPPER},
                       'PLT': {'lower':PLT_CPU_LOWER, 'upper': PLT_CPU_UPPER},
                       'APC': {'lower':APC_CPU_LOWER, 'upper': APC_CPU_UPPER}}
    node_mem = {'UGS': UGS_MEM, 'PLT': PLT_MEM,'APC': APC_MEM}
    node_bw_bounds = {'UGS': {'lower':UGS_THROUGHPUT_LOWER, 'upper': UGS_THROUGHPUT_UPPER},
                       'PLT': {'lower':PLT_THROUGHPUT_LOWER, 'upper': PLT_THROUGHPUT_UPPER},
                       'APC': {'lower':APC_THROUGHPUT_LOWER, 'upper': APC_THROUGHPUT_UPPER}}
    node_attributes = {}
    node_colors = {}
    node_sizes = {}
    for node in network.nodes():
        node_type = node.split("-")[0]
        cpu = np.random.uniform(node_cpu_bounds[node_type]['lower'],node_cpu_bounds[node_type]['upper'])
        mem = node_mem[node_type]
        gpu = True if node_type == "APC" else False
        
        resource = Resource(node, cpu, gpu, mem)
        node_attributes[node] = resource
        node_colors[node] = NODE_COLORS[node_type]
        node_sizes[node] = NODE_SIZES[node_type]
        
    nx.set_node_attributes(network, node_attributes, name='resource')
    nx.set_node_attributes(network, node_colors, name='color')
    nx.set_node_attributes(network, node_sizes, name='size')
    
    edge_attributes = {}
    for src, dst in network.edges:
        if src.startswith("UGS") or dst.startswith("UGS"):
            link_throughput = np.random.uniform(node_bw_bounds["UGS"]['lower'],node_bw_bounds["UGS"]['upper'])
        elif src.startswith("PLT") or dst.startswith("PLT"):
            link_throughput = np.random.uniform(node_bw_bounds["PLT"]['lower'],node_bw_bounds["PLT"]['upper'])
        else:
            link_throughput = np.random.uniform(node_bw_bounds["APC"]['lower'],node_bw_bounds["APC"]['upper'])
        
        edge_attributes[(src,dst)] = link_throughput
        
    nx.set_edge_attributes(network, edge_attributes, name='link_speed')
    
    return # network

def geo_topology(n,seed, node_density):
    
    weights = np.array([.5,1,2.5])
    node_types = ["UGS", "PLT", "APC"]
    node_dist = [.7,.2,.1] # np.array([30,12,4])/46 # [.5,.3,.2]
    colors = ['red', 'green', 'yellow']
    sizes = [100,200,400]
    
    
    weight_to_node_type = {weights[x]:node_types[x] for x in range(len(weights))}
    weight_to_color = {weights[x]:colors[x] for x in range(len(weights))}
    weight_to_size = {weights[x]:sizes[x] for x in range(len(weights))}
    
    node_weights = np.random.choice(weights,p=node_dist,size=(n,))
    
    node_locations = {x: list(np.random.uniform(0,np.sqrt(n/node_density),2)) for x in range(n)}
    
    network = nx.geographical_threshold_graph(n, 
                                           theta=30, 
                                           weight=dict(enumerate(node_weights)),
                                           metric=None, 
                                           p_dist=None, 
                                           pos=node_locations,
                                           seed=seed)
    
    mapping = {}
    node_counter = {'UGS': 0, 'PLT': 0, 'APC': 0}
    nodes = []
    for node_num in network.nodes:
        node_type = weight_to_node_type[network.nodes('weight')[node_num]]
        node_name = node_type + f"-{node_counter[node_type]}"
        mapping[node_num] = node_name
        node_counter[node_type] += 1
        
          
    nx.relabel_nodes(network, mapping, copy=False)
    
    return network

def ER_topology(n,seed):
        
    return # network

def Barabasi_toplogy(n,seed):
    
    return

def write_to_csv(network):
    
    with open('mean_band.csv', 'w') as f:
        f.write('edge,bandwidth\n')
        for edge in network.edges():
            f.write(f'\"{edge[0]},{edge[1]}\",{network.get_edge_data(edge[0], edge[1])["link_speed"]}\n')
            
    with open('resource_query.csv', 'w') as f:
        f.write('cpu_available,has_gpu,memory_available,scenario_node\n')
        for nodedata in network.nodes(data=True):
            resources = nodedata[1]['resource']
            f.write(f'{resources.cpu},{resources.gpu},{resources.mem},{nodedata[0]}\n')     
            
    return

def create_random_network(n,network_algorithm,seed,save_network,draw_network, node_density):
    
    np.random.seed(seed)
    create_topology = NETWORK_OPTIONS[network_algorithm]
      
    network = create_topology(n,seed,node_density)
    assign_resources(network)
    
    if draw_network:
        nx.draw(network,
                pos=network.nodes('pos'), 
                node_color=dict(network.nodes.data('color')).values(),
                node_size=list(dict(network.nodes.data('size')).values()),
                with_labels=True
                )
    
    if save_network:
        write_to_csv(network)
  
    return network


NETWORK_OPTIONS = {
    "geographic": geo_topology,
    "geo_topology": geo_topology,
    "ER": ER_topology,
    "ER_topology": ER_topology,
    "BA": Barabasi_toplogy,
    "Barabasi_toplogy": Barabasi_toplogy
}

# network = create_random_network(num_of_nodes, network_algorithm, None, False, True, node_density)
    
