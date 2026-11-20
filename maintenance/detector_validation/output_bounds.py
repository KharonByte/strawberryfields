"""Byte caps for a proposed bounded collection; not a collection release."""

START_BYTES = 16 * 1024
COMMAND_BYTES = 16 * 1024
EXIT_BYTES = 16 * 1024
ENV_BYTES = 64 * 1024
LOG_BYTES = 64 * 1024
SUMMARY_BYTES = 4 * 1024**2
SUMMARY_WRITES = 70
PROPOSED_CAPACITY = 768 * 1024**2
PARTIAL_BYTES = 480128 + 65536


def storage_plan():
    """All680 regular chunks plus280 secondary arrays and one terminal-failure allowance."""
    parts = {
        "680_regular_chunks": 680
        * (START_BYTES + COMMAND_BYTES + EXIT_BYTES + LOG_BYTES + 160128 + 262144),
        "280_secondary_arrays": 280 * 160128,
        "70_full_snapshots": SUMMARY_WRITES * SUMMARY_BYTES,
        "manifest_release_environment": 2 * SUMMARY_BYTES + ENV_BYTES,
        "index_reserve_and_header": 16 * 1024**2 + 4096,
        "terminal_invalid_unexpected_secondary_partial": 480128 + 160128 + PARTIAL_BYTES,
    }
    total = sum(parts.values())
    return {
        "parts": parts,
        "maximum_stored_bytes": total,
        "capacity": PROPOSED_CAPACITY,
        "headroom": PROPOSED_CAPACITY - total,
        "condition": "enforced member caps, at most70snapshots, terminal failure stops; no retry",
    }


def put_bounded(store, name, payload, maximum):
    if len(payload) > maximum:
        raise ValueError("member byte cap exceeded: " + name)
    store.put(name, payload)


class CollectionSnapshots:
    """Bound append-only full receipts; raw/exit and per-chunk checkpoints remain separate."""

    def __init__(self, store):
        self.store = store
        self.count = 0

    def put(self, payload):
        if self.count >= SUMMARY_WRITES:
            raise ValueError("full collection snapshot count exceeded")
        put_bounded(self.store, "collection.json", payload, SUMMARY_BYTES)
        self.count += 1
