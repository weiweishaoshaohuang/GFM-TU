import json, pathlib, time

class JsonlLogger:
    def __init__(self, path="mcts_events.jsonl"):
        self.path = pathlib.Path(path)
        # if self.path.exists():
        #     self.path.unlink()
        self.file = self.path.open("a", encoding="utf-8", buffering=1)

    def write(self, event_type, **payload):
        payload["type"] = event_type
        payload["ts"] = time.time()
        self.file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.file.flush()
