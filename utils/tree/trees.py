"""Core tree data structures and utilities for tree-based probe experiments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

# =============================================================================
# Public data structures and configuration
# =============================================================================

__all__ = [
    "Node",
    "TreeExample",
    "DistanceProbeConfig",
    "build_graph_and_depths",
    "tree_distance_matrix",
    "annotate_examples",
    "build_toy_examples",
    "build_traversal_examples",
    "tree_depth",
    "tree_distance",
    "pairwise_tree_distances",
]


@dataclass
class Node:
    """Single node in a toy reasoning tree."""

    nid: int
    text: str
    parent: Optional[int]


@dataclass
class TreeExample:
    """Container for toy reasoning trees used in the prototypes."""

    name: str
    nodes: List[Node]
    depth: Optional[Dict[int, int]] = None
    dist_mat: Optional[np.ndarray] = None


@dataclass
class DistanceProbeConfig:
    """Training configuration for the distance probe."""

    proj_dim: int = 16
    lr: float = 1e-2
    weight_decay: float = 1e-4
    steps: int = 1_500
    seed: int = 42
    fit_geometry: Literal["euclidean", "hyperbolic"] = "euclidean"
    pair_weighting: Literal["none", "inverse_freq"] = "inverse_freq"


def build_graph_and_depths(nodes: Sequence[Node]) -> Tuple[nx.Graph, Dict[int, int]]:
    """Return a NetworkX graph and BFS depths for the provided nodes."""

    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node.nid)
    root_id: Optional[int] = None
    for node in nodes:
        if node.parent is None:
            root_id = node.nid
            continue
        graph.add_edge(node.nid, node.parent)
    if root_id is None:
        raise ValueError("Tree requires a root node with parent=None")
    depths: Dict[int, int] = {root_id: 0}
    frontier: List[int] = [root_id]
    while frontier:
        next_frontier: List[int] = []
        for head in frontier:
            for child in graph.neighbors(head):
                if child in depths:
                    continue
                depths[child] = depths[head] + 1
                next_frontier.append(child)
        frontier = next_frontier
    return graph, depths


def tree_distance_matrix(graph: nx.Graph, ids: Sequence[int]) -> np.ndarray:
    """Compute the pairwise shortest-path distance matrix for the node ids."""

    n = len(ids)
    distances = np.zeros((n, n), dtype=np.float32)
    for i, source in enumerate(ids):
        lengths = nx.shortest_path_length(graph, source=source)
        for j, target in enumerate(ids):
            distances[i, j] = lengths[target]
    return distances


def annotate_examples(examples: Iterable[TreeExample]) -> List[TreeExample]:
    """Populate depth maps and distance matrices for tree examples."""

    annotated: List[TreeExample] = []
    for example in examples:
        graph, depths = build_graph_and_depths(example.nodes)
        example.depth = depths
        node_ids = [node.nid for node in example.nodes]
        example.dist_mat = tree_distance_matrix(graph, node_ids)
        annotated.append(example)
    return annotated


# =============================================================================
# Binary tree depth/distance helpers reused across scripts
# =============================================================================


def tree_depth(node_id: int) -> int:
    import math

    return int(math.floor(math.log2(node_id + 1))) if node_id >= 0 else 0


def tree_distance(a: int, b: int) -> int:
    if a == b:
        return 0
    da, db = tree_depth(a), tree_depth(b)
    u, v = a, b
    while da > db:
        u = (u - 1) // 2
        da -= 1
    while db > da:
        v = (v - 1) // 2
        db -= 1
    while u != v:
        u = (u - 1) // 2
        v = (v - 1) // 2
    lca_depth = tree_depth(u)
    return tree_depth(a) + tree_depth(b) - 2 * lca_depth


def pairwise_tree_distances(node_ids: Sequence[int]) -> np.ndarray:
    n = len(node_ids)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            dist = tree_distance(int(node_ids[i]), int(node_ids[j]))
            mat[i, j] = mat[j, i] = float(dist)
    return mat


# =============================================================================
# Binary traversal generators
# =============================================================================


def _describe_binary_path(path: str) -> str:
    mapping = {"L": "left", "R": "right"}
    if not path:
        return "root"
    return " -> ".join(mapping[ch] for ch in path)


def _generate_binary_traversal_nodes(depth: int, *, traversal: str = "bfs") -> List[Node]:
    """Create a full binary tree of a given depth with descriptive text."""

    traversal = traversal.lower()
    if traversal not in {"bfs", "dfs"}:
        raise ValueError(f"Unsupported traversal mode: {traversal}")

    nodes: List[Node] = []
    agenda: deque[Tuple[str, Optional[int]]] = deque([("", None)])
    while agenda:
        if traversal == "bfs":
            path, parent_id = agenda.popleft()
        else:  # dfs uses LIFO order to mimic pre-order expansion
            path, parent_id = agenda.pop()
        node_id = len(nodes)
        level = len(path)
        if level == 0:
            text = f"Task: Traverse a depth-{depth} binary tree from the root."
        else:
            direction_desc = _describe_binary_path(path)
            if level == depth:
                base = f"Leaf at {direction_desc}."
            else:
                base = f"Subtree at {direction_desc}."
            text = base
        nodes.append(Node(node_id, text, parent_id))
        if level < depth:
            left_child = (path + "L", node_id)
            right_child = (path + "R", node_id)
            if traversal == "bfs":
                agenda.append(left_child)
                agenda.append(right_child)
            else:
                agenda.append(right_child)
                agenda.append(left_child)
    return nodes


def build_toy_examples() -> List[TreeExample]:
    """Instantiate reusable toy reasoning trees."""

    examples: List[TreeExample] = []
    for name, specs in _STATIC_TOY_NODE_SPECS:
        nodes = [Node(nid, text, parent) for nid, text, parent in specs]
        examples.append(TreeExample(name=name, nodes=nodes))

    for name, depth, traversal in _TRAVERSAL_PRESETS:
        nodes = _generate_binary_traversal_nodes(depth, traversal=traversal)
        examples.append(TreeExample(name=name, nodes=nodes))

    return annotate_examples(examples)


# -----------------------------------------------------------------------------
# Stored example specifications
# -----------------------------------------------------------------------------

_STATIC_TOY_NODE_SPECS: Sequence[Tuple[str, Sequence[Tuple[int, str, Optional[int]]]]] = (
    (
        "binary_traversal",
        (
            (0, "Task: Perform in-order traversal (left, root, right) on the given binary tree.", None),
            (1, "Left subtree root: handles Steps 1, 3, and 5 (descend left, return, prepare to ascend).", 0),
            (2, "Left subtree left child: Step 2 visit.", 1),
            (3, "Left subtree right child: Step 4 visit.", 1),
            (4, "Right subtree root: Steps 6, 7, and 9 (return to root, then descend right).", 0),
            (5, "Right subtree left child: Step 8 visit.", 4),
            (6, "Right subtree right child: Step 10 visit.", 4),
        ),
    ),
    (
        "argument_structure",
        (
            (0, "Claim: The algorithm is correct for all binary search trees.", None),
            (1, "Sub-argument A: The base case holds for a single-node tree.", 0),
            (2, "Evidence A1: A single node is visited once in-order.", 1),
            (3, "Sub-argument B: The inductive step holds when attaching a child.", 0),
            (4, "Evidence B1: Left subtree is fully processed before root.", 3),
            (5, "Evidence B2: Right subtree is processed after root.", 3),
        ),
    ),
    (
        "decision_tree",
        (
            (0, "Question: Should we use in-order traversal here?", None),
            (1, "Consider sub-question: Are we extracting a sorted list?", 0),
            (2, "If yes, in-order preserves BST ordering.", 1),
            (3, "If not, consider pre-order or post-order for different purposes.", 1),
            (4, "Conclusion: Use in-order traversal when sorted output is desired.", 0),
        ),
    ),
    (
        "binary_traversal_large",
        (
            (0, "Task: Perform in-order traversal on a depth-3 binary tree (left, root, right).", None),
            (1, "Phase A: work through the full left subtree (Steps 1-4).", 0),
            (2, "Segment A1: descend into the left-left subtree root.", 1),
            (3, "Step 1: Visit the left-left-left leaf.", 2),
            (4, "Step 2: Visit the left-left-right leaf after returning.", 2),
            (5, "Segment A2: move to the left-right subtree root.", 1),
            (6, "Step 3: Visit the left-right-left leaf.", 5),
            (7, "Step 4: Visit the left-right-right leaf.", 5),
            (8, "Phase B: return to the tree root (Step 5) and transition to the right subtree.", 0),
            (9, "Segment B1: descend into the right-left subtree root.", 8),
            (10, "Step 6: Visit the right-left-left leaf.", 9),
            (11, "Step 7: Visit the right-left-right leaf.", 9),
            (12, "Segment B2: move to the right-right subtree root.", 8),
            (13, "Step 8: Visit the right-right-left leaf.", 12),
            (14, "Step 9: Visit the right-right-right leaf and conclude.", 12),
        ),
    ),
    (
        "argument_structure_large",
        (
            (0, "Claim: In-order traversal yields sorted output for any balanced BST.", None),
            (1, "Argument block A: Structure of the left half of the tree.", 0),
            (2, "Sub-argument A1: Base case for the left half.", 1),
            (3, "Evidence A1a: A single left leaf is emitted once.", 2),
            (4, "Evidence A1b: Returning to the parent preserves order.", 2),
            (5, "Sub-argument A2: Inductive step for attaching left children.", 1),
            (6, "Evidence A2a: Left branch completes before sibling nodes.", 5),
            (7, "Evidence A2b: No left value is skipped during ascent.", 5),
            (8, "Argument block B: Symmetric reasoning for the right half.", 0),
            (9, "Sub-argument B1: Base case for the right half.", 8),
            (10, "Evidence B1a: First right leaf appears after root.", 9),
            (11, "Evidence B1b: Parent revisit precedes right sibling.", 9),
            (12, "Sub-argument B2: Inductive step for right attachments.", 8),
            (13, "Evidence B2a: Right branch respects sorted ordering.", 12),
            (14, "Evidence B2b: Completion ensures no inversions remain.", 12),
        ),
    ),
    (
        "decision_tree_large",
        (
            (0, "Root decision: Pick a traversal strategy for the balanced BST.", None),
            (1, "Branch A: Do we need sorted output immediately?", 0),
            (2, "Check A1: Is stability across left subtrees required?", 1),
            (3, "Outcome A1a: Yes—output leftmost leaf first.", 2),
            (4, "Outcome A1b: No—document visit order but keep left-first bias.", 2),
            (5, "Check A2: Do we interleave summaries while traversing left children?", 1),
            (6, "Outcome A2a: Yes—emit annotations alongside each left leaf.", 5),
            (7, "Outcome A2b: No—collect left summaries before returning to root.", 5),
            (8, "Branch B: Focus on goals after returning to the root.", 0),
            (9, "Check B1: Do we pre-compute right-branch statistics?", 8),
            (10, "Outcome B1a: Yes—visit right-left leaves to gather data first.", 9),
            (11, "Outcome B1b: No—skip ahead to the right-right path.", 9),
            (12, "Check B2: Are we producing trace-style explanations?", 8),
            (13, "Outcome B2a: Yes—explain each right branch visit explicitly.", 12),
            (14, "Outcome B2b: No—summarize the final right leaf succinctly.", 12),
        ),
    ),
)

_TRAVERSAL_PRESETS: Sequence[Tuple[str, int, str]] = (
    ("binary_traversal_depth4", 4, "bfs"),
    ("binary_traversal_depth4_dfs", 4, "dfs"),
    ("binary_traversal_depth5", 5, "bfs"),
    ("binary_traversal_depth5_dfs", 5, "dfs"),
)


def build_traversal_examples(
    depths: Sequence[int],
    *,
    traversals: Sequence[str] = ("bfs", "dfs"),
) -> List[TreeExample]:
    """Generate binary tree traversal examples for the requested depths."""

    norm_traversals = tuple(dict.fromkeys(mode.lower() for mode in traversals))
    examples: List[TreeExample] = []
    for depth in depths:
        for traversal in norm_traversals:
            nodes = _generate_binary_traversal_nodes(depth, traversal=traversal)
            name = f"binary_traversal_depth{depth}_{traversal}"
            examples.append(TreeExample(name=name, nodes=nodes))
    return annotate_examples(examples)
