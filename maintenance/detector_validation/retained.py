import stat

"""Adapted frozen assessor raw-byte/count/statistical audit; not a new independent implementation. No simulator or analyser imports.
Run only after root's terminal notification. Frozen primary analytic certificates
are inputs, not candidate-frequency summaries. No collection/repair/resampling.
"""

import argparse, ast, hashlib, json, math, os, resource, signal, struct
from collections import Counter
from decimal import Decimal as D, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction as F
from pathlib import Path


from .storage import read_bounded as bounded_bytes


def jload(data):
    from .provenance import parse_json

    return parse_json(data)


def cdf(n, x, q):
    if x < 0:
        return F(0), F(0)
    if x >= n or q == 0:
        return F(1), F(1)
    if q == 1:
        return F(0), F(0)
    if x > n // 2:
        l, u = cdf(n, n - x - 1, 1 - q)
        return 1 - u, 1 - l
    ans = []
    for mode in (ROUND_FLOOR, ROUND_CEILING):
        with localcontext() as ctx:
            ctx.prec = 75
            ctx.rounding = mode
            z = D(q.numerator) / D(q.denominator)
            assert F(z) == q
            a = 1 - z
            # Integer power through repeated directed products.
            term = D(1)
            b = a
            k = n
            while k:
                if k & 1:
                    term = term * b
                k //= 2
                if k:
                    b = b * b
            total = term
            for k in range(1, x + 1):
                term = term * D(n - k + 1) / D(k) * z / a
                total += term
            ans.append(F(total))
    return max(F(0), ans[0]), min(F(1), ans[1])


def root(n, x, t):
    l, u = F(0), F(1)
    while u - l > F(1, 10**10):
        m = (l + u) / 2
        a, b = cdf(n, x, m)
        if a > t:
            l = m
        elif b < t:
            u = m
        else:
            raise ValueError("independent binomial bracket unresolved")
    return l, u


def cp(n, x):
    a = F(3, 68000)
    lo = (F(0), F(0)) if x == 0 else root(n, x - 1, 1 - a)
    hi = (F(1), F(1)) if x == n else root(n, x, a)
    return lo[0], hi[1]


def npy(data, n, m):
    assert data[:8] == b"\x93NUMPY\x01\x00"
    size = struct.unpack("<H", data[8:10])[0]
    assert size <= 1024
    h = ast.literal_eval(data[10 : 10 + size].decode("ascii").strip())
    assert h == {"descr": "<i8", "fortran_order": False, "shape": (n, m)}
    body = data[10 + size :]
    assert len(body) == 8 * n * m
    return list(struct.iter_unpack("<" + "q" * m, body))


def audit(args):
    resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    signal.alarm(120)
    path = Path(args.container)
    if any(p.is_symlink() for p in (path.absolute(), *path.absolute().parents)):
        raise ValueError("symlink container")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("nonregular container")

    expected_container_sha256 = "e184d9e7636ae221060a2afe131f99bad1850c2173cf86613eeee21a876efec1"
    try:

        container_hasher = hashlib.sha256()
        offset = 0
        if os.fstat(fd).st_size != 1073741824:
            raise ValueError("original retained container size mismatch")
        while offset < 1073741824:
            data = os.pread(fd, min(1024**2, 1073741824 - offset), offset)
            if not data:
                raise ValueError("short retained container")
            container_hasher.update(data)
            offset += len(data)
        if container_hasher.hexdigest() != expected_container_sha256:
            raise ValueError("original retained container hash mismatch")

        def read(n, o):
            d = os.pread(fd, n, o)
            assert len(d) == n
            return d

        magic, capacity, reserve = struct.unpack(">8sQQ", read(24, 0))
        assert magic == b"DTVST001" and capacity == path.stat().st_size and capacity <= 2 * 1024**3
        assert 8192 <= reserve <= 16 * 1024**2 and reserve % 2 == 0
        candidates = []
        for k in range(2):
            o = capacity - reserve + k * (reserve // 2)
            sm, g, n, sha = struct.unpack(">8sQQ32s", read(56, o))
            if sm != b"DTVIX001" or n > reserve // 2 - 64:
                continue
            data = read(n, o + 64)
            if hashlib.sha256(data).digest() != sha:
                continue
            idx = jload(data)
            if idx["generation"] == g:
                candidates.append(idx)
        idx = max(candidates, key=lambda x: x["generation"])
        entries = idx["files"]
        assert len(entries) <= 8192
        ranges = []
        digests = {}

        def member(name):
            r = entries[name]
            o, n = r["offset"], r["bytes"]
            assert (
                type(o) is int
                and type(n) is int
                and 4096 <= o
                and 0 <= n <= 4 * 1024**2
                and o + n <= capacity - reserve
            )
            data = read(n, o)
            assert hashlib.sha256(data).hexdigest() == r["sha256"]
            return data

        for name, rec in entries.items():
            assert not name.startswith("/") and all(
                x not in ("", "..", ".") for x in name.split("/")
            )
            data = member(name)
            ranges.append((rec["offset"], rec["offset"] + len(data)))
            digests[name] = rec["sha256"]
        ranges.sort()
        assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))
        receipt = jload(member("collection.json"))
        assert receipt["status"] in (
            "complete",
            "incomplete",
            "error",
            "failed",
        ), "not a terminal checkpoint"
        if (
            hashlib.sha256(bounded_bytes(args.manifest)).hexdigest()
            != "5caba5fa7d5301583755ae4c4878f6d39c82fa088b4475a7ea9ea5fb0e0cae8a"
        ):
            raise ValueError("retained analysis requires exact original frozen manifest")
        manifest = jload(bounded_bytes(args.manifest))
        assert len(manifest["cases"]) == 68
        assert hashlib.sha256(bounded_bytes(args.release)).hexdigest() == args.release_sha256
        assert hashlib.sha256(member("release.json")).hexdigest() == args.release_sha256
        assert member("manifest.json") == bounded_bytes(args.manifest)
        release = jload(bounded_bytes(args.release))
        certreceipt = Path(args.references) / "controls-receipt.json"
        assert (
            hashlib.sha256(bounded_bytes(certreceipt)).hexdigest()
            == "cac2853b25f74101aff0c3ddc4eb2c4ac01dc7ca3e4ccda563df52f8e0679bb7"
        )
        certfiles = {f["path"]: f for f in jload(bounded_bytes(certreceipt))["files"]}
        assert (
            release["manifest_sha256"] == hashlib.sha256(bounded_bytes(args.manifest)).hexdigest()
        )
        reports = {}
        scalar = []
        tv = []
        tails = []
        fatal = []
        with localcontext() as ctx:
            ctx.prec = 80
            e = F(((D(1840) / D(".004")).ln() / D(200000)).sqrt()) + F(1, 10**60)
            for i, case in enumerate(manifest["cases"]):
                cid = case["id"]
                rows = []
                counts = Counter()
                modes = len(case["output_mode_order"])
                missing = []
                for c in range(10):
                    name = f"raw/{cid}/{c:02d}/samples.npy"
                    if name not in entries:
                        missing.append(c)
                        continue
                    block = npy(member(name), 10000, modes)
                    metaname = f"raw/{cid}/{c:02d}/metadata.json"
                    if metaname not in entries:
                        fatal.append([cid, c, "missing_metadata"])
                    else:
                        meta = jload(member(metaname))
                        expected = {
                            "case": case,
                            "case_id": cid,
                            "chunk_index": c,
                            "seed": case["chunk_seeds"][c],
                            "rows": 10000,
                            "status": "captured_not_accepted",
                            "output_mode_order": case["output_mode_order"],
                            "requested_mode_order": case["measured_modes"],
                        }
                        if any(meta.get(k) != v for k, v in expected.items()):
                            fatal.append([cid, c, "metadata_binding"])
                        if meta.get("measurement_execution", {}).get("status") != "returned":
                            fatal.append([cid, c, "no_return_marker"])
                        if not meta.get("imports") or meta.get("imports") != meta.get(
                            "imports_after"
                        ):
                            fatal.append([cid, c, "imports_identity"])
                    exitname = f"commands/{cid}/{c:02d}/exit.json"
                    if exitname not in entries:
                        fatal.append([cid, c, "missing_exit"])
                    else:
                        ex = jload(member(exitname))
                        if ex.get("exit_code") != 0 or ex.get("limit_failure") is not None:
                            fatal.append([cid, c, "chunk_exit"])
                    for row in block:
                        if any(v < 0 for v in row) or (
                            case["detector"] == "threshold" and any(v > 1 for v in row)
                        ):
                            fatal.append([cid, c, "raw_domain"])
                            continue
                        key = tuple(min(v, 5) for v in row) if case["detector"] == "pnr" else row
                        counts["cell:" + ",".join(map(str, key))] += 1
                        if "odd_total_count" in case["additional_events"] and sum(row) % 2:
                            counts["event:odd_total_count"] += 1
                        if "unequal_counts" in case["additional_events"] and len(set(row)) != 1:
                            counts["event:unequal_counts"] += 1
                        if case["family"] == "vacuum" and any(row):
                            fatal.append([cid, c, "vacuum_support"])
                        if any(
                            case["eta"][mode] == 0 and row[k]
                            for k, mode in enumerate(case["output_mode_order"])
                        ):
                            fatal.append([cid, c, "zero_efficiency_support"])
                        if (
                            case["detector"] == "pnr"
                            and case["family"] == "squeezed_vacuum"
                            and case["eta"] == [1]
                            and row[0] % 2
                        ):
                            fatal.append([cid, c, "squeezed_support"])
                        if (
                            case["detector"] == "pnr"
                            and case["family"] == "two_mode_squeezed"
                            and all(x == 1 for x in case["eta"])
                            and row[0] != row[1]
                        ):
                            fatal.append([cid, c, "pair_support"])
                    if case["path"] == "local_engine_gaussian":
                        sec = f"raw/{cid}/{c:02d}/secondary.npy"
                        if sec not in entries:
                            fatal.append([cid, c, "missing_secondary"])
                        else:
                            other = npy(member(sec), 10000, modes)
                            order = [
                                case["measured_modes"].index(m) for m in case["output_mode_order"]
                            ]
                            assert all(
                                tuple(q[k] for k in order) == r for q, r in zip(other, block)
                            )
                    rows.extend(block)
                rep = {
                    "rows": len(rows),
                    "missing_chunks": missing,
                    "counts": dict(counts),
                    "status": "incomplete",
                }
                reports[cid] = rep
                if missing:
                    continue
                exempt = case["family"] == "vacuum" or all(
                    case["eta"][m] == 0 for m in case["output_mode_order"]
                )
                if not exempt:
                    for period in (1, 10, 100, 1000, 10000):
                        if all(rows[j] == rows[j % period] for j in range(period, len(rows))):
                            fatal.append([cid, "replay", period])
                            break
                certpath = Path(args.references) / f"correct_{i:02d}" / "oracle-certificate.json"
                certbytes = bounded_bytes(certpath, 1024**2)
                expected = certfiles[f"correct_{i:02d}/oracle-certificate.json"]
                assert (
                    len(certbytes) == expected["bytes"]
                    and hashlib.sha256(certbytes).hexdigest() == expected["sha256"]
                )
                cert = jload(certbytes)["comparisons"]
                assert len(cert) == case["comparison_count"]
                refs = {k: F(v["reference_point_decimal"]) for k, v in cert.items()}
                assert all(F(v["certified_error_upper"]) <= F(1, 10**8) for v in cert.values())
                b = F(1, 10**8)
                case_status = []
                metrics = {}
                for k, p in refs.items():
                    d = abs(F(counts[k], 100000) - p)
                    upper = d + e + b
                    lower = d - e - b
                    state = (
                        "accepted"
                        if upper <= F(17, 1000)
                        else "defect" if lower > F(17, 1000) else "inconclusive"
                    )
                    scalar.append(state)
                    case_status.append(state)
                    metrics[k] = {"upper": str(upper), "lower": str(max(0, lower)), "status": state}
                names = [k for k in refs if k.startswith("cell:")]
                m = len(names)
                r = F(((D(m) * D(2).ln() + (D(68) / D(".003")).ln()) / D(200000)).sqrt()) + F(
                    1, 10**60
                )
                d = sum((abs(F(counts[k], 100000) - refs[k]) for k in names), F(0)) / 2
                upper = d + r + m * b / 2
                lower = d - r - m * b / 2
                state = (
                    "accepted"
                    if upper <= F(3, 100)
                    else "defect" if lower > F(3, 100) else "inconclusive"
                )
                tv.append(state)
                case_status.append(state)
                rep["joint_tv"] = {
                    "upper": str(upper),
                    "lower": str(max(0, lower)),
                    "status": state,
                }
                if case["detector"] == "pnr":
                    tailnames = [k for k in names if "5" in k[5:].split(",")]
                    x = sum(counts[k] for k in tailnames)
                    p = sum((refs[k] for k in tailnames), F(0))
                    # Sum certified errors explicitly: b is a whole-tail allowance, not eachcell.
                    err = sum((F(cert[k]["certified_error_upper"]) for k in tailnames), F(0))
                    assert err <= b
                    idealhi = sum((F(cert[k]["upper"]) for k in tailnames), F(0))
                    assert idealhi <= F(6, 1000)
                    L, U = cp(100000, x)
                    upper = max(abs(L - p), abs(U - p)) + b
                    lower = max(L - p - b, p - U - b, F(0))
                    state = (
                        "accepted"
                        if upper <= F(3, 1000)
                        else "defect" if lower > F(3, 1000) else "inconclusive"
                    )
                    tails.append(state)
                    case_status.append(state)
                    rep["tail"] = {
                        "count": x,
                        "L": str(L),
                        "U": str(U),
                        "upper": str(upper),
                        "lower": str(lower),
                        "status": state,
                        "ideal_tail_upper": str(idealhi),
                        "full_pnr_tv_target_plus_ideal_tail": str(F(3, 100) + idealhi),
                    }
                rep["comparisons"] = metrics
                rep["status"] = (
                    "defect"
                    if "defect" in case_status
                    else "accepted" if set(case_status) == {"accepted"} else "inconclusive"
                )
        return {
            "implementation": "frozen-prefix-formulas-v6",
            "container": str(path),
            "checkpoint_generation": idx["generation"],
            "checkpoint_state": idx["state"],
            "collection_status": receipt["status"],
            "release_sha256": args.release_sha256,
            "verified_member_count": len(entries),
            "verified_member_hashes": digests,
            "case_reports": reports,
            "criteria_statuses": {
                "cell": dict(Counter(scalar)),
                "joint_tv": dict(Counter(tv)),
                "tail": dict(Counter(tails)),
            },
            "integrity_or_support_findings": fatal[:100],
            "affected_case_ids": sorted({f[0] for f in fatal}),
            "finding_count": len(fatal),
            "scientific_raw_verdict": (
                "incomplete"
                if receipt["status"] != "complete"
                or any(r["missing_chunks"] for r in reports.values())
                else (
                    "not_accepted"
                    if fatal or any(r["status"] != "accepted" for r in reports.values())
                    else "accepted_probability_and_raw_rules"
                )
            ),
            "scope": "adapted frozen assessor byte/count audit against frozen primary analytic certificates; production provenance is separately QA reviewed; no simulator imports/resampling",
            "analysis_resources": {
                "max_cpu_seconds": 60,
                "max_wall_seconds": 120,
                "user_cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                "system_cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_stime,
            },
            "prefix_rule": "original v3 families/radii and N100000 applied only to complete ten-chunk cases; incomplete study cannot pass",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    finally:
        os.close(fd)


def retained_report(args):
    report = audit(args)
    cases = report["case_reports"]
    coverage = {"completed": 0, "partial": 0, "untested": 0}
    for case in cases.values():
        state = (
            "completed" if not case["missing_chunks"] else "partial" if case["rows"] else "untested"
        )
        case["coverage"] = state
        coverage[state] += 1
    manifest = jload(bounded_bytes(args.manifest))
    from itertools import product

    for definition in manifest["cases"]:
        case = cases[definition["id"]]
        case["scientific_status"] = {
            "accepted": "PASS",
            "defect": "FAIL",
            "inconclusive": "INCOMPLETE",
        }.get(case["status"], "UNTESTED" if not case["rows"] else "INCOMPLETE")
        labels = [
            "cell:" + ",".join(map(str, row))
            for row in product(
                range(6 if definition["detector"] == "pnr" else 2),
                repeat=len(definition["output_mode_order"]),
            )
        ]
        labels += ["event:" + event for event in definition["additional_events"]]
        labels += ["joint_tv"] + (["tail"] if definition["detector"] == "pnr" else [])
        evaluated = dict(case.get("comparisons", {}))
        for key in ("joint_tv", "tail"):
            if key in case:
                evaluated[key] = case[key]
        case["criterion_states"] = {
            key: {"accepted": "PASS", "defect": "FAIL", "inconclusive": "INCOMPLETE"}.get(
                evaluated.get(key, {}).get("status"),
                "UNTESTED" if not case["rows"] else "INCOMPLETE",
            )
            for key in labels
        }
        case["criterion_evaluated"] = {key: key in evaluated for key in labels}
    for cid, case in cases.items():
        findings = [f for f in report["integrity_or_support_findings"] if f[0] == cid]
        invalid = cid in report["affected_case_ids"]
        case["integrity_status"] = "INVALID" if invalid else "VERIFIED"
        case["integrity_findings"] = findings
        if invalid:
            case["scientific_status"] = "INCOMPLETE"
            case["criterion_evaluated"] = {key: False for key in case["criterion_states"]}
            case["criterion_states"] = {key: "INCOMPLETE" for key in case["criterion_states"]}
        case["reason"] = (
            "invalid_evidence"
            if invalid
            else (
                "statistical_inconclusive"
                if case["status"] == "inconclusive"
                else "missing_chunks" if case["missing_chunks"] else "original_full_case_evaluated"
            )
        )
    failed = any(c["status"] == "defect" for c in cases.values())
    report.update(
        schema="detector-test-report-v6",
        mode="retained",
        execution_status="completed",
        integrity_status="INVALID" if report["finding_count"] else "VERIFIED",
        retained_container_sha256="e184d9e7636ae221060a2afe131f99bad1850c2173cf86613eeee21a876efec1",
        scientific_verdict=(
            "INCOMPLETE"
            if report["finding_count"]
            else (
                "FAIL"
                if failed
                else (
                    "INCOMPLETE"
                    if coverage["completed"] != 68
                    else (
                        "PASS"
                        if all(c["status"] == "accepted" for c in cases.values())
                        else "INCOMPLETE"
                    )
                )
            )
        ),
        coverage_status="COMPLETE" if coverage["completed"] == 68 else "INCOMPLETE",
        coverage=coverage,
        synthetic=False,
        acceptance_claim=False,
        source_formula_origin="preparation/attempt2-review/prefix_audit.py; original frozen criteria unchanged",
    )
    return report
