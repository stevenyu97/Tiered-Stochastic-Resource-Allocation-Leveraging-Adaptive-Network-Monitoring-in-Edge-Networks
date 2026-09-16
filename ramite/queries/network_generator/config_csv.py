# -*- coding: utf-8 -*-
"""
Created on Thu May  9 17:19:11 2024

@author: jake.b.perazzone

This is the config file where a few different options are set, such as random
network generation tehcnique, etc. as well as their settings.

"""
            
runs = 10
num_of_nodes = 30

node_density = 60 # nodes per square unit

random_network = True # False # 
draw_network = True # False # 

network_algorithm = 'geographic' # 'ER' # 'BA' # 

UGS_CPU_LOWER = 1000
UGS_CPU_UPPER = 2000
PLT_CPU_LOWER = 4000
PLT_CPU_UPPER = 6000
APC_CPU_LOWER = 6000
APC_CPU_UPPER = 10000

UGS_MEM = 3e9
PLT_MEM = 1.5e10
APC_MEM = 1.5e10

UGS_THROUGHPUT_LOWER = 1e6 # minimum for 1-hop offload = 5e6, and 1e7 for 2 hop
UGS_THROUGHPUT_UPPER = 1e8
PLT_THROUGHPUT_LOWER = 1e8
PLT_THROUGHPUT_UPPER = 1e8
APC_THROUGHPUT_LOWER = 1e8
APC_THROUGHPUT_UPPER = 1e8

NODE_COLORS = {
    "UGS": "brown",
    "PLT": "cornflowerblue",
    "APC": "darkblue",
    "CHQ": "blue",
}