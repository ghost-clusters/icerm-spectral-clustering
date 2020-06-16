import networkx as nx
import gzip
import re
import sys

import matplotlib.pyplot as plt
from networkx import nx

'''
Examples copied from: https://networkx.github.io/documentation/stable/auto_examples/index.html#graph
'''

def load_karate_club():
    # small network graph from an anthropological study, see 
    # https://networkx.github.io/documentation/stable/auto_examples/graph/plot_karate_club.html

    return nx.karate_club_graph()


def load_roget_graph():
    """ Return the thesaurus graph from the roget.dat example in
    the Stanford Graph Base.
    """

    fh = open('data/roget_dat.txt', 'rb')

    G = nx.DiGraph()

    for line in fh.readlines():
        line = line.decode()
        if line.startswith("*"):  # skip comments
            continue
        if line.startswith(" "):  # this is a continuation line, append
            line = oldline + line
        if line.endswith("\\\n"):  # continuation line, buffer, goto next
            oldline = line.strip("\\\n")
            continue

        (headname, tails) = line.split(":")

        # head
        numfind = re.compile("^\d+")  # re to find the number of this word
        head = numfind.findall(headname)[0]  # get the number

        G.add_node(head)

        for tail in tails.split():
            if head == tail:
                pass # do not include self loops
            G.add_edge(head, tail)

    fh.close()
    return G.to_undirected()
