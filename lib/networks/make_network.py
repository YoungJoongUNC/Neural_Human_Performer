import os
import imp

def make_network(cfg):

    module = cfg.cross_transformer_network_module
    path = cfg.cross_transformer_network_path

    network = imp.load_source(module, path).Network()

    return network
