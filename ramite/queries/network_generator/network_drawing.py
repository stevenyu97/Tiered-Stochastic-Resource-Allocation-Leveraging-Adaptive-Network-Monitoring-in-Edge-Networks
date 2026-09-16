# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 13:08:25 2023

@author: jake.b.perazzone
"""
import networkx as nx
import matplotlib.pyplot as plt

def draw_my_network(network, random_network):
    if random_network:
        plt.figure(1)
        nx.draw(network,
                pos=network.nodes('pos'), 
                node_color=dict(network.nodes.data('color')).values(),
                node_size=list(dict(network.nodes.data('size')).values()),
                with_labels=True
                )
    else:
        plt.figure(1)
        nx.set_node_attributes(network, nx.kamada_kawai_layout(network), name='pos')
        nx.draw(network,
                pos=network.nodes('pos'),
                node_color=dict(network.nodes.data('color')).values(),
                node_size=list(dict(network.nodes.data('size')).values()),
                with_labels=True)


