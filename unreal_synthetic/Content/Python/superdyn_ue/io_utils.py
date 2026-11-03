import gzip, hashlib, json
from pathlib import Path


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def load_scene(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
