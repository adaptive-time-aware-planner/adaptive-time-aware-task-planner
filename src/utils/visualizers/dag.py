# utils/viz/dag.py
import textwrap
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx


def wrap_label(label, width=10):
    """
    너무 긴 라벨을 일정 길이로 줄바꿈하기 위한 유틸 함수
    """
    return "\n".join(textwrap.wrap(str(label), width=width))


def visualize_graph(
    G: nx.DiGraph,
    save_path: Optional[str] = None,
    is_display: bool = False,
):
    """
    NetworkX DiGraph를 Graphviz layout(dot)로 시각화.
    - 간선 라벨(Interval, IsCritical)
    - 노드 라벨(줄바꿈)
    - 색상(노드/엣지)
    """

    # dot 레이아웃
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    plt.figure(figsize=(10, 8))

    # 노드 라벨
    node_labels = {}
    node_colors = []
    color_map = {"Monitoring": "pink", "Interaction": "lightblue"}

    for node in G.nodes():
        nd_label = G.nodes[node].get("label", str(node))
        node_labels[node] = wrap_label(nd_label, width=10)

        # Node 별 subtask_type에서 색 결정
        st_type = G.nodes[node].get("subtask_type", "")
        node_colors.append(color_map.get(st_type, "gray"))

    # 간선 라벨/색
    edge_labels = {}
    edge_colors = []
    for u, v, d in G.edges(data=True):
        info = d.get("info", {})
        interval = info.get("Interval", None)
        is_crit = info.get("IsCritical", False)

        edge_labels[(u, v)] = f"{round(interval, 2)}" if interval else ""
        edge_colors.append("red" if is_crit else "blue")

    # 노드/엣지 그리기
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_size=1500,
        node_color=node_colors,
        edge_color=edge_colors,
        arrows=True,
    )

    # 노드 라벨
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight="bold")

    # 간선 라벨
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_color="black", rotate=False
    )

    # 범례
    red_edge = plt.Line2D([0], [0], color="red", lw=2)
    blue_edge = plt.Line2D([0], [0], color="blue", lw=2)
    plt.legend(
        [red_edge, blue_edge], ["Critical", "Not Critical"], loc="best", frameon=True
    )
    plt.title("Directed Acyclic Graph (DAG)")

    if save_path:
        plt.savefig(save_path, dpi=300)

    if is_display:
        plt.show()
    else:
        plt.close()
