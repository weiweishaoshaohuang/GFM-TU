import json
import math
import tkinter as tk
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

MAX_VISIBLE_CHILDREN = 3      # 每個節點最多展示的子節點
MIN_VISITS_TO_SHOW = 2        # 低於此訪問數的子節點會被折疊
LEVEL_HEIGHT = 180
MARGIN_X = 160
MARGIN_Y = 110
CANVAS_MIN_WIDTH = 1400

ACTION_ABBR = {
    "SearchNode": "SN",
    "GetAllNeighbours": "GAN",
    "GetSharedNeighbours": "GSN",
    "Answer": "ANS",
}
LEGEND_TEXT = "Legend:\nSelect: 根節點\nSN: SearchNode\nGAN: GetAllNeighbours\nGSN: GetSharedNeighbours\nANS: Answer\n紅色 = 最佳路徑"


@dataclass
class NodeInfo:
    node_id: str
    parent_id: str | None = None
    label: str = ""
    visits: int = 0
    avg: float = 0.0
    depth: int = 0
    reward: float | None = None
    children: list[str] = field(default_factory=list)
    pos: tuple[float, float] = (0.0, 0.0)
    hidden_children: int = 0  # 用於顯示 +N 提示


class TreeCanvasViewer:
    def __init__(self, file_path: str, refresh_ms: int = 200):
        self.file_path = Path(file_path)
        self.refresh_ms = refresh_ms
        self.offset = 0
        self.nodes: dict[str, NodeInfo] = {}
        self.levels: defaultdict[int, list[str]] = defaultdict(list)
        self.best_path_nodes: set[str] = set()

        self.root = tk.Tk()
        self.root.title("MCTS 推理樹")
        self.canvas = tk.Canvas(self.root, width=CANVAS_MIN_WIDTH, height=720, bg="#f8f8ff")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.root.after(100, self.refresh)

    # 讀取新事件 -----------------------------------------------------------
    def refresh(self):
        try:
            file_size = self.file_path.stat().st_size
            if file_size < self.offset:  # 檔案被清空或覆寫
                self.offset = 0
                self._reset_tree()
        except FileNotFoundError:
            self.offset = 0
            self._reset_tree()
            return

        try:
            with self.file_path.open("r", encoding="utf-8") as f:
                f.seek(self.offset)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    event = json.loads(line)
                    self.handle_event(event)
                self.offset = f.tell()
        except FileNotFoundError:
            pass
        self.root.after(self.refresh_ms, self.refresh)

    # 事件處理 -------------------------------------------------------------
    def handle_event(self, event: dict):
        etype = event.get("type")
        node_id = str(event.get("node_id"))

        if etype == "root":
            self._reset_tree()
            node = NodeInfo(node_id=node_id, parent_id=None, label="Select")
            self.nodes[node_id] = node
            self._rebuild_levels()
        elif etype == "expand":
            parent_id = str(event.get("parent_id"))
            action = event.get("action", ["", []])
            label = self._short_label(action)
            node = NodeInfo(node_id=node_id, parent_id=parent_id, label=label)
            self.nodes[node_id] = node
            if parent_id in self.nodes:
                self.nodes[parent_id].children.append(node_id)
            self._rebuild_levels()
        elif etype == "simulate":
            node = self.nodes.get(node_id)
            if node:
                node.reward = event.get("reward")
        elif etype == "backprop":
            node = self.nodes.get(node_id)
            if node:
                node.visits = event.get("visits", node.visits)
                node.avg = event.get("avg", node.avg)
        elif etype == "best_path":
            path_ids = event.get("path", [])
            self.best_path_nodes = {str(pid) for pid in path_ids}

        self._redraw()

    # 排版與繪製 -----------------------------------------------------------
    def _reset_tree(self):
        self.nodes.clear()
        self.levels.clear()
        self.best_path_nodes.clear()

    def _rebuild_levels(self):
        self.levels.clear()
        if not self.nodes:
            return

        roots = [n for n in self.nodes.values() if n.parent_id is None]
        queue = list(roots)
        for root_node in roots:
            root_node.depth = 0
            self.levels[0].append(root_node.node_id)

        while queue:
            node = queue.pop(0)
            child_nodes = [self.nodes[cid] for cid in node.children if cid in self.nodes]

            child_nodes.sort(key=lambda n: (-(n.visits), -n.avg))
            filtered = [child for child in child_nodes if child.visits >= MIN_VISITS_TO_SHOW]
            if not filtered and child_nodes:
                filtered = child_nodes[:1]

            visible = filtered[:MAX_VISIBLE_CHILDREN]
            node.hidden_children = max(0, len(child_nodes) - len(visible))

            for child in visible:
                child.depth = node.depth + 1
                self.levels[child.depth].append(child.node_id)
                queue.append(child)

    def _redraw(self):
        self.canvas.delete("all")
        if not self.nodes:
            self._draw_legend()
            return

        canvas_width = max(self.canvas.winfo_width(), CANVAS_MIN_WIDTH)

        for depth, node_ids in self.levels.items():
            count = max(len(node_ids), 1)
            spacing = (canvas_width - 2 * MARGIN_X) / (count + 1)
            for idx, nid in enumerate(node_ids):
                x = MARGIN_X + spacing * (idx + 1)
                y = MARGIN_Y + depth * LEVEL_HEIGHT
                self.nodes[nid].pos = (x, y)

        for node in self.nodes.values():
            if node.parent_id and node.parent_id in self.nodes and node.node_id in self.levels.get(node.depth, []):
                parent = self.nodes[node.parent_id]
                self._draw_edge(parent, node)

        for node in self.nodes.values():
            if node.node_id in self.levels.get(node.depth, []):
                self._draw_node(node)

        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self._draw_legend()

    def _draw_edge(self, parent: NodeInfo, child: NodeInfo):
        x1, y1 = parent.pos
        x2, y2 = child.pos
        visits = max(child.visits, 0)
        width = 1 + math.log(visits + 1)
        highlight = parent.node_id in self.best_path_nodes and child.node_id in self.best_path_nodes
        color = "#d64545" if highlight else ("#5470d3" if visits > 0 else "#bbc4de")
        width = max(width, 2.4) if highlight else width
        self.canvas.create_line(x1, y1 + 30, x2, y2 - 30,
                                width=width, fill=color, capstyle=tk.ROUND)

    def _draw_node(self, node: NodeInfo):
        x, y = node.pos
        r = 32
        highlight = node.node_id in self.best_path_nodes
        fill = "#ffdfd9" if highlight else ("#ffe7ba" if node.parent_id is None else "#ffffff")
        outline = "#d64545" if highlight else ("#c2a47a" if node.parent_id is None else "#7a8ba8")
        width = 3 if highlight else 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                fill=fill, outline=outline, width=width)

        lines = [node.label or "Select",
                 f"v={node.visits}",
                 f"avg={node.avg:.2f}"]
        text = "\n".join(lines[:3])
        text_color = "#8c1a1a" if highlight else "#1f2a44"
        self.canvas.create_text(x, y, text=text,
                                font=("Microsoft YaHei", 9), fill=text_color)

        if node.reward is not None:
            self.canvas.create_text(x, y + r + 10,
                                    text=f"r={node.reward:.2f}",
                                    font=("Microsoft YaHei", 8), fill="#666666")

        if node.hidden_children > 0:
            self.canvas.create_text(x + r - 6, y + r - 6,
                                    text=f"+{node.hidden_children}",
                                    font=("Microsoft YaHei", 8), fill="#c55f2a")

    def _draw_legend(self):
        self.canvas.create_text(10, 10, anchor="nw",
                                text=LEGEND_TEXT,
                                font=("Microsoft YaHei", 10),
                                fill="#4a4a4a",
                                tags="legend")

    # 工具 ---------------------------------------------------------------
    def _short_label(self, action) -> str:
        if isinstance(action, (list, tuple)) and action:
            name = str(action[0])
            abbr = ACTION_ABBR.get(name, name[:3].upper())
            payload = ""
            if len(action) > 1 and action[1]:
                payload = self._format_payload(action[1])
            label = f"{abbr} {payload}".strip()
            return label[:14]
        return str(action)[:12]

    def _format_payload(self, params) -> str:
        if isinstance(params, (list, tuple)) and params:
            first = params[0]
            if isinstance(first, (list, tuple)) and len(first) >= 2:
                if isinstance(first[0], int) and isinstance(first[1], int):
                    return f"r{first[0]}c{first[1]}"
            if isinstance(first, str):
                return first.strip()[:6]
        elif isinstance(params, str):
            return params.strip()[:6]
        return ""

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    viewer = TreeCanvasViewer("mcts_events.jsonl")
    viewer.run()
