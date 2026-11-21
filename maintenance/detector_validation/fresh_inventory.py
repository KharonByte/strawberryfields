"""Reconcile fresh terminal bookkeeping before any raw rows receive credit."""

from .provenance import parse_json

ROOT_MEMBERS = {
    "manifest.json",
    "release.json",
    "source.json",
    "runtime.json",
    "ledger.json",
    "attempt-start.json",
    "collection.json",
}
COMMAND_MEMBERS = {"command.json", "started.json", "exit.json", "stdout.txt", "stderr.txt"}
RAW_MEMBERS = {"samples.npy", "metadata.json"}


def reconcile(reader, manifest, collection, release_sha):
    """Return successfully recorded coordinates, never credit an unledgered attempt.

    Incomplete attempts may retain a final command without an exit/ledger entry.
    Those bytes remain diagnostic evidence, not accepted sample coverage.
    """
    identifier = manifest["experiment_id"]
    status = collection.get("status")
    if status not in ("incomplete", "collected_not_accepted") or reader.state != {
        "status": status,
        "experiment_id": identifier,
    }:
        raise ValueError("terminal checkpoint/collection state mismatch")
    if any(
        collection.get(k) != value
        for k, value in {
            "schema": "detector-fresh-collection-v7",
            "diagnostic_reference": manifest.get("diagnostic_reference"),
            "experiment_id": identifier,
            "profile": manifest["profile"],
            "release_sha256": release_sha,
        }.items()
    ):
        raise ValueError("collection identity mismatch")
    start = parse_json(reader.read_bytes("attempt-start.json"))
    if start != {
        "schema": "detector-fresh-start-v7",
        "experiment_id": identifier,
        "release_sha256": release_sha,
        "status": "started_completion_unknown",
        "single_attempt": True,
    }:
        raise ValueError("attempt start identity mismatch")
    cases = manifest["profile_definition"]["cases"]
    coordinates = [(c["id"], i) for c in cases for i in range(c["chunk_count"])]
    by_id = {c["id"]: c for c in cases}
    ledger = collection.get("cases")
    if not isinstance(ledger, list) or len(ledger) > len(cases):
        raise ValueError("invalid collection case ledger")
    recorded = {}
    for position, row in enumerate(ledger):
        if row.get("id") != cases[position]["id"] or not isinstance(row.get("chunks"), list):
            raise ValueError("collection case ledger order mismatch")
        for index, entry in enumerate(row["chunks"]):
            coordinate = (row["id"], index)
            if type(entry.get("index")) is not int or entry["index"] != index:
                raise ValueError("collection chunk ledger order mismatch")
            if len(recorded) >= len(coordinates) or coordinate != coordinates[len(recorded)]:
                raise ValueError("collection chunk ledger is not an execution prefix")
            recorded[coordinate] = {k: v for k, v in entry.items() if k != "index"}
    inventory = set(reader.inventory())
    if not ROOT_MEMBERS <= inventory:
        raise ValueError("missing fresh provenance members")
    prefixes = {f"{cid}/{i:02d}": (cid, i) for cid, i in coordinates}
    attempted = set()
    for name in inventory - ROOT_MEMBERS:
        parts = name.split("/")
        if len(parts) != 4 or parts[0] not in ("commands", "raw"):
            raise ValueError("unexpected fresh evidence member")
        prefix = "/".join(parts[1:3])
        if prefix not in prefixes:
            raise ValueError("unexpected fresh evidence coordinate")
        allowed = (
            (COMMAND_MEMBERS | {"partial-frame.bin"})
            if parts[0] == "commands"
            else (RAW_MEMBERS | {"secondary.npy", "returned-invalid.npy"})
        )
        if parts[3] not in allowed:
            raise ValueError("unexpected fresh evidence member")
        attempted.add(prefixes[prefix])
    expected_attempts = set(coordinates[: len(recorded)])
    if status == "incomplete" and len(recorded) < len(coordinates):
        expected_attempts.add(coordinates[len(recorded)])
    if not set(recorded) <= attempted or not attempted <= expected_attempts:
        raise ValueError("attempted inventory differs from collection ledger")
    accepted = set()
    for coordinate in attempted:
        cid, index = coordinate
        prefix = f"{cid}/{index:02d}"
        if f"commands/{prefix}/command.json" not in inventory:
            raise ValueError("attempt without command binding")
        if coordinate not in recorded:
            continue
        exit_record = parse_json(reader.read_bytes(f"commands/{prefix}/exit.json"))
        if exit_record != recorded[coordinate]:
            raise ValueError("collection ledger differs from worker exit record")
        if exit_record.get("exit_code") != 0 or exit_record.get("limit_failure") is not None:
            if coordinate != coordinates[len(recorded) - 1]:
                raise ValueError("collection continued after a failed worker")
            continue
        required = {f"commands/{prefix}/{n}" for n in COMMAND_MEMBERS}
        required |= {f"raw/{prefix}/{n}" for n in RAW_MEMBERS}
        if by_id[cid]["path"] == "local_engine_gaussian":
            required.add(f"raw/{prefix}/secondary.npy")
        present = {n for n in inventory if n.startswith((f"commands/{prefix}/", f"raw/{prefix}/"))}
        if present != required:
            raise ValueError("successful chunk has missing or diagnostic-only members")
        accepted.add(coordinate)
    if status == "collected_not_accepted" and accepted != set(coordinates):
        raise ValueError("complete collection lacks complete successful ledger coverage")
    return accepted
