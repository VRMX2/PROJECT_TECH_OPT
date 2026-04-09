"""
engine/network_model.py
───────────────────────
Network topology management using NetworkX.

Models a computer network as a directed graph where:
  • Nodes = network devices / hosts
  • Edges = communication links
  • Node state: normal | attacked | defended | compromised

Provides:
  • NetworkModel class with add/remove/attack/defend/reset methods
  • Factory: build_default_network() for a quick demo topology
"""

import networkx as nx
import random
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Constants for node states
# ─────────────────────────────────────────────────────────────────────────────

STATE_NORMAL      = "normal"
STATE_ATTACKED    = "attacked"
STATE_DEFENDED    = "defended"
STATE_COMPROMISED = "compromised"

# Colour mapping used by the visualization layer
STATE_COLORS = {
    STATE_NORMAL:      "#007ACC",   # deeper blue
    STATE_ATTACKED:    "#FF003C",   # neon red
    STATE_DEFENDED:    "#39FF14",   # neon green
    STATE_COMPROMISED: "#FF9900",   # intense orange
}


class NetworkModel:
    """
    Wraps a NetworkX DiGraph to represent a cyber-network with game-state
    metadata on each node.
    """

    def __init__(self):
        self.G = nx.DiGraph()
        self._node_counter = 0

    # ─────────────────────────────────────────────────────────────────────
    # Node management
    # ─────────────────────────────────────────────────────────────────────

    def add_node(self, label: Optional[str] = None, node_type: str = "host") -> str:
        """
        Add a new node to the network.

        Args:
            label:     Human-readable name. Auto-generated if None.
            node_type: 'host', 'router', 'server', 'firewall'.

        Returns:
            The node ID string.
        """
        node_id = f"N{self._node_counter}"
        self._node_counter += 1
        label = label or node_id
        self.G.add_node(
            node_id,
            label=label,
            node_type=node_type,
            state=STATE_NORMAL,
            value=random.randint(1, 10),   # asset value (used in payoffs)
        )
        return node_id

    def remove_node(self, node_id: str):
        """Remove a node and all its edges."""
        if node_id in self.G:
            self.G.remove_node(node_id)

    def add_edge(self, src: str, dst: str, weight: float = 1.0):
        """Add a directed edge between two nodes."""
        if src in self.G and dst in self.G:
            self.G.add_edge(src, dst, weight=weight)

    # ─────────────────────────────────────────────────────────────────────
    # State transitions
    # ─────────────────────────────────────────────────────────────────────

    def attack_node(self, node_id: str):
        """Mark a node as under attack."""
        if node_id in self.G:
            current = self.G.nodes[node_id]["state"]
            if current == STATE_DEFENDED:
                # Attack blocked – stays defended
                pass
            elif current == STATE_ATTACKED:
                # Persistent attack → compromise
                self.G.nodes[node_id]["state"] = STATE_COMPROMISED
            else:
                self.G.nodes[node_id]["state"] = STATE_ATTACKED

    def defend_node(self, node_id: str):
        """Mark a node as defended (blocks current/future attacks)."""
        if node_id in self.G:
            self.G.nodes[node_id]["state"] = STATE_DEFENDED

    def compromise_node(self, node_id: str):
        """Forcibly compromise a node."""
        if node_id in self.G:
            self.G.nodes[node_id]["state"] = STATE_COMPROMISED

    def reset_node(self, node_id: str):
        """Reset a single node to normal state."""
        if node_id in self.G:
            self.G.nodes[node_id]["state"] = STATE_NORMAL

    def reset_all(self):
        """Reset all nodes to normal state."""
        for node_id in self.G.nodes:
            self.G.nodes[node_id]["state"] = STATE_NORMAL

    # ─────────────────────────────────────────────────────────────────────
    # Queries
    # ─────────────────────────────────────────────────────────────────────

    def get_nodes(self) -> list:
        """Return list of (node_id, attrs) tuples."""
        return list(self.G.nodes(data=True))

    def get_edges(self) -> list:
        """Return list of (src, dst, attrs) tuples."""
        return list(self.G.edges(data=True))

    def node_ids(self) -> list:
        return list(self.G.nodes())

    def state_summary(self) -> dict:
        """Return count of nodes in each state."""
        counts = {s: 0 for s in STATE_COLORS}
        for _, data in self.G.nodes(data=True):
            counts[data.get("state", STATE_NORMAL)] += 1
        return counts

    def total_value(self) -> int:
        """Sum of asset values across all nodes."""
        return sum(data.get("value", 0) for _, data in self.G.nodes(data=True))

    def compromised_value(self) -> int:
        """Sum of asset values of compromised nodes."""
        return sum(
            data.get("value", 0)
            for _, data in self.G.nodes(data=True)
            if data.get("state") == STATE_COMPROMISED
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory: demo network
# ─────────────────────────────────────────────────────────────────────────────

def build_default_network() -> NetworkModel:
    """
    Build a representative enterprise-style network topology:

      Internet → Firewall → [DMZ Server, Router]
      Router   → [WebServer, DB, WorkstationA, WorkstationB]
      DMZ      → WebServer
    """
    net = NetworkModel()

    fw  = net.add_node("Firewall",      "firewall")
    dmz = net.add_node("DMZ Server",    "server")
    rtr = net.add_node("Core Router",   "router")
    web = net.add_node("Web Server",    "server")
    db  = net.add_node("Database",      "server")
    ws1 = net.add_node("Workstation A", "host")
    ws2 = net.add_node("Workstation B", "host")
    srv = net.add_node("File Server",   "server")

    # Edges
    for src, dst in [
        (fw, dmz), (fw, rtr),
        (rtr, web), (rtr, db),
        (rtr, ws1), (rtr, ws2),
        (rtr, srv), (dmz, web),
    ]:
        net.add_edge(src, dst)

    return net
