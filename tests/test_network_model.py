"""
tests/test_network_model.py
───────────────────────────
Unit tests for engine/network_model.py.

Tests cover:
  • Node creation, removal, and edge management
  • State transitions: attack, defend, compromise, reset
  • Query helpers: state_summary, total_value, compromised_value
  • build_default_network() factory
"""

import pytest
from engine.network_model import (
    NetworkModel,
    build_default_network,
    STATE_NORMAL,
    STATE_ATTACKED,
    STATE_DEFENDED,
    STATE_COMPROMISED,
    STATE_COLORS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def empty_net():
    """Fresh NetworkModel with no nodes."""
    return NetworkModel()


@pytest.fixture
def simple_net():
    """Three-node network: fw → router → host."""
    net = NetworkModel()
    fw  = net.add_node("Firewall", "firewall")
    rtr = net.add_node("Router",   "router")
    hst = net.add_node("Host",     "host")
    net.add_edge(fw, rtr)
    net.add_edge(rtr, hst)
    return net, fw, rtr, hst


# ─────────────────────────────────────────────────────────────────────────────
# Node management
# ─────────────────────────────────────────────────────────────────────────────

class TestNodeManagement:

    def test_add_node_returns_unique_ids(self, empty_net):
        id1 = empty_net.add_node("A")
        id2 = empty_net.add_node("B")
        assert id1 != id2

    def test_add_node_increments_counter(self, empty_net):
        ids = [empty_net.add_node() for _ in range(5)]
        # All IDs must be unique
        assert len(set(ids)) == 5

    def test_add_node_auto_label(self, empty_net):
        nid = empty_net.add_node()
        nodes = dict(empty_net.get_nodes())
        assert nodes[nid]["label"] == nid

    def test_add_node_custom_label(self, empty_net):
        nid = empty_net.add_node(label="MyServer", node_type="server")
        nodes = dict(empty_net.get_nodes())
        assert nodes[nid]["label"] == "MyServer"
        assert nodes[nid]["node_type"] == "server"

    def test_add_node_default_state_is_normal(self, empty_net):
        nid = empty_net.add_node()
        nodes = dict(empty_net.get_nodes())
        assert nodes[nid]["state"] == STATE_NORMAL

    def test_add_node_has_asset_value(self, empty_net):
        nid = empty_net.add_node()
        nodes = dict(empty_net.get_nodes())
        assert 1 <= nodes[nid]["value"] <= 10

    def test_remove_node_removes_from_graph(self, empty_net):
        nid = empty_net.add_node("X")
        empty_net.remove_node(nid)
        node_ids = [n for n, _ in empty_net.get_nodes()]
        assert nid not in node_ids

    def test_remove_nonexistent_node_is_safe(self, empty_net):
        # Should not raise
        empty_net.remove_node("ghost")

    def test_node_ids_returns_all(self, simple_net):
        net, fw, rtr, hst = simple_net
        ids = net.node_ids()
        assert set(ids) == {fw, rtr, hst}


# ─────────────────────────────────────────────────────────────────────────────
# Edge management
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeManagement:

    def test_add_edge_appears_in_get_edges(self, simple_net):
        net, fw, rtr, hst = simple_net
        edges = [(s, d) for s, d, _ in net.get_edges()]
        assert (fw, rtr) in edges
        assert (rtr, hst) in edges

    def test_add_edge_for_missing_node_is_safe(self, empty_net):
        nid = empty_net.add_node("A")
        # dst "ghost" doesn't exist — should not raise
        empty_net.add_edge(nid, "ghost")
        assert len(empty_net.get_edges()) == 0

    def test_remove_node_also_removes_its_edges(self, simple_net):
        net, fw, rtr, hst = simple_net
        net.remove_node(rtr)
        edges = [(s, d) for s, d, _ in net.get_edges()]
        assert (fw, rtr) not in edges
        assert (rtr, hst) not in edges


# ─────────────────────────────────────────────────────────────────────────────
# State transitions
# ─────────────────────────────────────────────────────────────────────────────

class TestStateTransitions:

    def test_attack_normal_node_becomes_attacked(self, simple_net):
        net, fw, rtr, hst = simple_net
        net.attack_node(hst)
        nodes = dict(net.get_nodes())
        assert nodes[hst]["state"] == STATE_ATTACKED

    def test_attack_attacked_node_becomes_compromised(self, simple_net):
        net, fw, rtr, hst = simple_net
        net.attack_node(hst)
        net.attack_node(hst)   # second attack
        nodes = dict(net.get_nodes())
        assert nodes[hst]["state"] == STATE_COMPROMISED

    def test_attack_defended_node_stays_defended(self, simple_net):
        net, fw, rtr, hst = simple_net
        net.defend_node(hst)
        net.attack_node(hst)   # blocked
        nodes = dict(net.get_nodes())
        assert nodes[hst]["state"] == STATE_DEFENDED

    def test_defend_node_sets_state(self, simple_net):
        net, fw, rtr, hst = simple_net
        net.defend_node(fw)
        nodes = dict(net.get_nodes())
        assert nodes[fw]["state"] == STATE_DEFENDED

    def test_compromise_node_sets_state(self, simple_net):
        net, fw, rtr, hst = simple_net
        net.compromise_node(rtr)
        nodes = dict(net.get_nodes())
        assert nodes[rtr]["state"] == STATE_COMPROMISED

    def test_reset_node_restores_normal(self, simple_net):
        net, fw, rtr, hst = simple_net
        net.compromise_node(hst)
        net.reset_node(hst)
        nodes = dict(net.get_nodes())
        assert nodes[hst]["state"] == STATE_NORMAL

    def test_reset_all_restores_all_nodes(self, simple_net):
        net, fw, rtr, hst = simple_net
        net.attack_node(fw)
        net.compromise_node(rtr)
        net.defend_node(hst)
        net.reset_all()
        nodes = dict(net.get_nodes())
        for nid in [fw, rtr, hst]:
            assert nodes[nid]["state"] == STATE_NORMAL

    def test_attack_nonexistent_node_is_safe(self, empty_net):
        empty_net.attack_node("ghost")   # must not raise

    def test_defend_nonexistent_node_is_safe(self, empty_net):
        empty_net.defend_node("ghost")


# ─────────────────────────────────────────────────────────────────────────────
# Query helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryHelpers:

    def test_state_summary_all_normal_initially(self, simple_net):
        net, fw, rtr, hst = simple_net
        summary = net.state_summary()
        assert summary[STATE_NORMAL] == 3
        assert summary[STATE_ATTACKED] == 0
        assert summary[STATE_DEFENDED] == 0
        assert summary[STATE_COMPROMISED] == 0

    def test_state_summary_updates_after_transitions(self, simple_net):
        net, fw, rtr, hst = simple_net
        net.attack_node(fw)
        net.defend_node(rtr)
        net.compromise_node(hst)
        summary = net.state_summary()
        assert summary[STATE_NORMAL] == 0
        assert summary[STATE_ATTACKED] == 1
        assert summary[STATE_DEFENDED] == 1
        assert summary[STATE_COMPROMISED] == 1

    def test_total_value_positive(self, simple_net):
        net, *_ = simple_net
        assert net.total_value() > 0

    def test_compromised_value_zero_initially(self, simple_net):
        net, *_ = simple_net
        assert net.compromised_value() == 0

    def test_compromised_value_after_compromise(self, simple_net):
        net, fw, rtr, hst = simple_net
        nodes = dict(net.get_nodes())
        expected = nodes[fw]["value"] + nodes[rtr]["value"]
        net.compromise_node(fw)
        net.compromise_node(rtr)
        assert net.compromised_value() == expected

    def test_compromised_value_leq_total_value(self, simple_net):
        net, fw, rtr, hst = simple_net
        net.compromise_node(fw)
        assert net.compromised_value() <= net.total_value()

    def test_state_colors_has_all_four_states(self):
        assert STATE_NORMAL in STATE_COLORS
        assert STATE_ATTACKED in STATE_COLORS
        assert STATE_DEFENDED in STATE_COLORS
        assert STATE_COMPROMISED in STATE_COLORS


# ─────────────────────────────────────────────────────────────────────────────
# build_default_network factory
# ─────────────────────────────────────────────────────────────────────────────

class TestDefaultNetwork:

    def test_default_network_has_expected_node_count(self):
        net = build_default_network()
        assert len(net.node_ids()) == 8

    def test_default_network_has_edges(self):
        net = build_default_network()
        assert len(net.get_edges()) > 0

    def test_default_network_all_nodes_normal(self):
        net = build_default_network()
        summary = net.state_summary()
        assert summary[STATE_NORMAL] == 8

    def test_default_network_node_types_present(self):
        net = build_default_network()
        types = {data["node_type"] for _, data in net.get_nodes()}
        assert "firewall" in types
        assert "router" in types
        assert "server" in types
        assert "host" in types
