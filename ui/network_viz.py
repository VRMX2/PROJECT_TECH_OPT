"""
ui/network_viz.py
──────────────────
Interactive network graph visualization using PyVis.

Renders the NetworkModel as a drag-and-drop HTML network embedded in Streamlit.
Node colors reflect game state (normal / attacked / defended / compromised).

Provides:
  • render_network(net_model, height) → renders HTML component in Streamlit
  • build_pyvis_network(net_model)    → returns pyvis Network object
"""

import os
import tempfile
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network as PyvisNetwork

from engine.network_model import NetworkModel, STATE_COLORS


# ─────────────────────────────────────────────────────────────────────────────
# Icon / shape mapping per node type
# ─────────────────────────────────────────────────────────────────────────────

NODE_SHAPES = {
    "firewall": "diamond",
    "router":   "dot",
    "server":   "square",
    "host":     "ellipse",
}

NODE_SIZES = {
    "firewall": 40,
    "router":   35,
    "server":   30,
    "host":     25,
}


def build_pyvis_network(net_model: NetworkModel, dark: bool = True) -> PyvisNetwork:
    """
    Convert a NetworkModel into a PyVis network for interactive rendering.

    Args:
        net_model: NetworkModel instance.
        dark:      If True, use dark background styling.

    Returns:
        Configured pyvis.Network object.
    """
    bg_color = "#0e1117" if dark else "#ffffff"
    font_color = "#ffffff" if dark else "#000000"

    pv_net = PyvisNetwork(
        height="500px",
        width="100%",
        bgcolor=bg_color,
        font_color=font_color,
        directed=True,
        notebook=False,
    )

    # Configure physics for better layout
    pv_net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 120,
          "springConstant": 0.04,
          "damping": 0.09
        }
      },
      "edges": {
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.7 } },
        "color": { "color": "#4A90D9", "highlight": "#74b3f5" },
        "width": 1.5,
        "smooth": { "type": "curvedCW", "roundness": 0.1 }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true
      }
    }
    """)

    # Add nodes
    for node_id, data in net_model.get_nodes():
        state     = data.get("state", "normal")
        ntype     = data.get("node_type", "host")
        color     = STATE_COLORS.get(state, "#4A90D9")
        shape     = NODE_SHAPES.get(ntype, "ellipse")
        size      = NODE_SIZES.get(ntype, 25)
        label     = data.get("label", node_id)
        value_str = f"Asset Value: {data.get('value', '?')}"
        tooltip   = f"{label}\nType: {ntype}\nState: {state}\n{value_str}"

        # Add glow-like border when attacked/compromised
        border_width = 3 if state in ("attacked", "compromised") else 1
        border_color = "#FF0000" if state == "attacked" else (
                       "#FF6600" if state == "compromised" else color)

        pv_net.add_node(
            node_id,
            label=label,
            title=tooltip,
            color={"background": color, "border": border_color,
                   "highlight": {"background": color, "border": "#ffffff"}},
            shape=shape,
            size=size,
            borderWidth=border_width,
            font={"size": 12, "color": font_color},
        )

    # Add edges
    for src, dst, edata in net_model.get_edges():
        pv_net.add_edge(src, dst, title=f"{src} → {dst}")

    return pv_net


def render_network(net_model: NetworkModel, height: int = 520):
    """
    Render the network as an interactive HTML component inside Streamlit.

    Args:
        net_model: NetworkModel instance.
        height:    Height of the component in pixels.
    """
    pv_net = build_pyvis_network(net_model)

    # Save to a temp HTML file and load it
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as f:
        tmp_path = f.name
        pv_net.save_graph(tmp_path)

    with open(tmp_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    components.html(html_content, height=height, scrolling=False)
