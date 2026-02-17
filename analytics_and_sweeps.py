from collections import defaultdict
import itertools
import random

from problem import SLOT_LIMITS, VLEN


def sweep_params(param_grid: dict = None, seed: int = 123, top_n: int = 10):
    """
    Sweep over parameter combinations and report the best results.

    param_grid: dict of param_name -> list of values to try.
                Only specified params are swept; others use DEFAULT_PARAMS.
    """
    if param_grid is None:
        param_grid = {
            "level1_guard": [0, 1, 8],
            "level2_guard": [0, 4, 8],
            "level3_guard": [0, 8],
            "packer_short_threshold": [4, 8, 12],
            "packer_medium_threshold": [8, 14, 20],
            "packer_random_prob": [0.0, 0.5, 0.8, 1.0],
            "groups_per_chunk_cap": [16, 18, 19],
        }

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combos = list(itertools.product(*values))
    print(f"Sweeping {len(combos)} combinations across {keys}")

    results = []
    best_so_far = float("inf")
    for idx, combo in enumerate(combos):
        overrides = dict(zip(keys, combo))
        params = {**DEFAULT_PARAMS, **overrides}
        try:
            cycles = do_kernel_test(10, 16, 256, seed=seed, params=params, quiet=True)
        except Exception as e:
            print(f"  [{idx+1}/{len(combos)}] FAIL {overrides}: {e}")
            continue
        if cycles < best_so_far:
            best_so_far = cycles
            marker = " ***"
        else:
            marker = ""
        results.append((cycles, idx, overrides))
        if (idx + 1) % 50 == 0 or marker:
            print(f"  [{idx+1}/{len(combos)}] {cycles} cycles {overrides}{marker}")

    results.sort(key=lambda r: r[0])
    print(f"\n=== Top {top_n} results ===")
    for i, (cycles, _, overrides) in enumerate(results[:top_n]):
        delta = {k: v for k, v in overrides.items() if v != DEFAULT_PARAMS.get(k)}
        print(f"  {i+1}. {cycles} cycles  (speedup {BASELINE/cycles:.1f}x)  delta={delta}")

    print(f"\n=== Bottom 3 ===")
    for cycles, _, overrides in results[-3:]:
        delta = {k: v for k, v in overrides.items() if v != DEFAULT_PARAMS.get(k)}
        print(f"  {cycles} cycles  delta={delta}")

    return results


def random_sweep(n_samples: int = 500, seed: int = 42, top_n: int = 15):
    """
    Random sampling over a large parameter space including per-group
    vselect/valu cutoffs (in multiples of VLEN=8).
    """
    rng = random.Random(seed)
    batch_size = 256
    # Cutoff values: multiples of 8 from 0 to 256 inclusive
    cutoff_choices = list(range(0, batch_size + 1, VLEN))  # 0,8,16,...,256

    results = []
    best_so_far = float("inf")
    print(f"Random sweep: {n_samples} samples")

    for idx in range(n_samples):
        overrides = {
            "depth1_vselect_cutoff": rng.choice(cutoff_choices),
            "depth2_vselect_cutoff": rng.choice(cutoff_choices),
            "level1_guard": rng.choice([0, 8, 16]),
            "level2_guard": rng.choice([0, 8, 16]),
            "level3_guard": rng.choice([0, 8, 16]),
            "packer_short_threshold": rng.randint(4, 20),
            "packer_medium_threshold": rng.randint(8, 28),
            "packer_random_prob": rng.choice([0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0]),
            "groups_per_chunk_cap": rng.randint(14, 22),
        }
        params = {**DEFAULT_PARAMS, **overrides}
        try:
            cycles = do_kernel_test(10, 16, 256, seed=123, params=params, quiet=True)
        except Exception as e:
            print(f"  [{idx+1}/{n_samples}] FAIL: {e}")
            continue
        if cycles < best_so_far:
            best_so_far = cycles
            marker = " ***"
        else:
            marker = ""
        results.append((cycles, idx, overrides))
        if (idx + 1) % 100 == 0 or marker:
            print(f"  [{idx+1}/{n_samples}] {cycles} cycles{marker}")

    results.sort(key=lambda r: r[0])
    print(f"\n=== Top {top_n} results ===")
    for i, (cycles, _, ov) in enumerate(results[:top_n]):
        print(f"  {i+1}. {cycles} cycles  (speedup {BASELINE/cycles:.1f}x)  {ov}")

    print(f"\n=== Bottom 5 ===")
    for cycles, _, ov in results[-5:]:
        print(f"  {cycles} cycles  {ov}")

    return results


def print_analytics(packed, stats):
    print("\n=== Cross-packer analytics ===")
    print(f"Total packed cycles: {len(packed)}")
    for engine in ("load", "valu", "alu", "store"):
        empty = stats["empty_slots"][engine]
        total = len(packed) * SLOT_LIMITS[engine]
        pct = 100 * empty / total if total else 0
        blocked = stats["blocked_behind_head"][engine]
        print(f"  {engine:5s}: {empty:5d}/{total:5d} slots empty ({pct:4.1f}%), " f"{blocked} stream-cycles blocked behind head")
        # Show what's blocking
        for eng2 in ("load", "valu", "alu", "store", "flow"):
            b = stats["blocked_by"].get((engine, eng2), 0)
            if b:
                print(f"         blocked by residual {eng2}: {b}")
    print()

    # Slot budget analysis
    NON_EXPANDABLE = {"multiply_add", "vbroadcast"}
    total_slots = defaultdict(int)
    non_exp_valu = 0
    exp_valu = 0
    per_round_slots = defaultdict(lambda: defaultdict(int))  # round -> engine -> count
    per_round_non_exp = defaultdict(int)
    per_round_exp = defaultdict(int)

    for result in packed:
        for engine, ops in result.items():
            if engine.startswith("_"):
                continue
            total_slots[engine] += len(ops)

    # Count from original streams (before packing) for per-round breakdown
    # Re-derive from the packed meta
    for result in packed:
        meta = result.get("_meta", {})
        for engine, ops in result.items():
            if engine.startswith("_"):
                continue
            for i, op in enumerate(ops):
                entry = meta.get((engine, i))
                rnd = entry[2] if entry else None
                if engine == "valu":
                    if op[0] in NON_EXPANDABLE:
                        non_exp_valu += 1
                        if rnd is not None:
                            per_round_non_exp[rnd] += 1
                    else:
                        exp_valu += 1
                        if rnd is not None:
                            per_round_exp[rnd] += 1
                if rnd is not None:
                    per_round_slots[rnd][engine] += 1

    C = len(packed)
    N = 19  # streams
    print("=== Slot budget ===")
    print(f"  valu: {total_slots['valu']} total ({non_exp_valu} non-expandable, {exp_valu} expandable)")
    print(f"  alu:  {total_slots['alu']}")
    print(f"  load: {total_slots['load']}")
    print(f"  store:{total_slots['store']}")

    # Theoretical minimum with valu<->alu expansion
    # x = expandable valu kept as valu
    # Constraint 1: non_exp + x <= 6*C  (valu capacity)
    # Constraint 2: (exp - x)*8 + alu <= 12*C  (alu capacity)
    # Constraint 3: load <= 2*C
    # Balance: (non_exp + x)/6 = ((exp - x)*8 + alu)/12
    total_alu = total_slots["alu"]
    total_load = total_slots["load"]
    # 12*(non_exp + x) = 6*((exp - x)*8 + alu)
    # 12*non_exp + 12x = 6*8*exp - 48x + 6*alu
    # 60x = 48*exp + 6*alu - 12*non_exp
    if exp_valu > 0:
        x_balanced = (48 * exp_valu + 6 * total_alu - 12 * non_exp_valu) / 60
        x_balanced = max(0, min(exp_valu, x_balanced))
        c_valu_alu = (non_exp_valu + x_balanced) / 6
    else:
        c_valu_alu = non_exp_valu / 6
    c_load = total_load / 2
    c_store = total_slots["store"] / 2
    c_min = max(c_valu_alu, c_load, c_store)
    bottleneck = "valu/alu" if c_valu_alu >= c_load and c_valu_alu >= c_store else "load" if c_load >= c_store else "store"

    print(f"\n  Theoretical min (perfect packing + valu<->alu):")
    print(f"    valu/alu balanced: {c_valu_alu:.0f} cycles (x={x_balanced:.0f} exp kept as valu)")
    print(f"    load bound:        {c_load:.0f} cycles")
    print(f"    store bound:       {c_store:.0f} cycles")
    print(f"    => C_min = {c_min:.0f} ({bottleneck}-bound)")
    print(f"    Actual: {C} cycles ({C/c_min:.2f}x theoretical)")

    # Per-round breakdown
    all_rounds = sorted(per_round_slots.keys())
    print(f"\n  Per-round slot counts (summed across all streams):")
    print(f"  {'rnd':>4s} {'valu':>5s} {'(ne)':>5s} {'(ex)':>5s} {'alu':>5s} {'load':>5s} {'store':>5s}")
    for r in all_rounds:
        s = per_round_slots[r]
        ne = per_round_non_exp.get(r, 0)
        ex = per_round_exp.get(r, 0)
        print(f"  {r:4d} {s.get('valu',0):5d} {ne:5d} {ex:5d} {s.get('alu',0):5d} {s.get('load',0):5d} {s.get('store',0):5d}")
