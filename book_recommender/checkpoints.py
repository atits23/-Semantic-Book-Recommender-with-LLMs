"""Content-addressed inference cache: interrupted runs can safely resume."""

import hashlib
import json

from .config import ARTIFACTS, make_directories


def cache_key(text, model, settings):
    payload = json.dumps([model, settings, text], sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def predict_cached(texts, model, settings, predict, stage, batch_size=16):
    make_directories()
    path = ARTIFACTS / "checkpoints" / f"{stage}.jsonl"
    cache = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                cache[item["key"]] = item["value"]
            except (ValueError, KeyError):
                continue  # A process may have stopped during its last write.
    keys = [cache_key(t, model, settings) for t in texts]
    missing = {key: text for key, text in zip(keys, texts, strict=True) if key not in cache}
    from tqdm import tqdm

    # Similar-length inputs waste fewer padded tokens in transformer batches.
    # Results are restored to original order using their content keys below.
    items = sorted(missing.items(), key=lambda item: len(item[1]))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")  # Isolate a possibly truncated previous record.
        for start in tqdm(range(0, len(items), batch_size), desc=stage):
            batch = items[start : start + batch_size]
            results = predict([text for _, text in batch])
            if len(results) != len(batch):
                raise ValueError("Model returned an unexpected number of predictions")
            for (key, _), result in zip(batch, results, strict=True):
                cache[key] = result
                handle.write(json.dumps({"key": key, "value": result}) + "\n")
            handle.flush()
    return [cache[key] for key in keys]
