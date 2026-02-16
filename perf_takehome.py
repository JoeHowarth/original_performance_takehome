"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
from dataclasses import dataclass
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    cdiv,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)

class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.v_const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, list[tuple]]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: slot})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def vscratch_const(self, val, name=None):
        if val not in self.v_const_map:
            addr = self.alloc_scratch(name, VLEN)
            self.add("load", ("const", addr, val))
            self.add("valu", ("vbroadcast", addr, addr))
            self.v_const_map[val] = addr
        return self.v_const_map[val]

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, vtmp1, vtmp2, round, batch_base):
        slots = []
        HASH_STAGES = [
            ("+", 0x7ED55D16, "+", "<<", 12),
            ("^", 0xC761C23C, "^", ">>", 19),
            ("+", 0x165667B1, "+", "<<", 5),
            ("+", 0xD3A2646C, "^", "<<", 9),
            ("+", 0xFD7046C5, "+", "<<", 3),
            ("^", 0xB55A4F09, "^", ">>", 16),
        ]

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if hi % 2 == 0:
                vsht3 = self.vscratch_const((1 << val3) + 1)
                slots.append({"valu": [("multiply_add", val_hash_addr, val_hash_addr, vsht3, self.vscratch_const(val1))]})
            else:
                slots.append(
                    {"valu": [(op1, vtmp1, val_hash_addr, self.vscratch_const(val1)), (op3, vtmp2, val_hash_addr, self.vscratch_const(val3))]}
                )
                # using some alu doesn't help here
                # if (hi ^ batch_base) % 2 == 0:
                # if hi == 1 or hi == 3:
                #     slots.append({"alu": [(op2, val_hash_addr + j, vtmp1 + j, vtmp2 + j) for j in range(VLEN)]})
                # else:
                #     slots.append({"valu": [(op2, val_hash_addr, vtmp1, vtmp2)]})
                slots.append({"valu": [(op2, val_hash_addr, vtmp1, vtmp2)]})
            # slots.append({"debug": [("vcompare", val_hash_addr, [(round, batch_base + j, "hash_stage", hi) for j in range(VLEN)])]})
        return slots

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """

        self.forest_height = forest_height

        tmp1 = self.alloc_scratch(f"tmp1")

        init_streams = []
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        self.alloc_scratch() # padding
        # for i, v in enumerate(init_vars):
            # self.add("load", ("const", self.scratch[v], i))
            # self.add("load", ("load", self.scratch[v], tmp1))
        self.add("load", ("vload", self.scratch["rounds"], 0))

        # self.alloc_scratch("zeros", VLEN)
        # self.vscratch_const(1, "ones")
        # self.alloc_scratch("twos", VLEN)


        # zero_const = self.scratch_const(0)
        # one_const = self.scratch_const(1)
        # two_const = self.scratch_const(2)

        # zeros = self.vscratch_const(0, "zeros")
        # ones = self.vscratch_const(1, "ones")
        # twos = self.vscratch_const(2, "twos")
        # vn_nodes = self.alloc_scratch("vn_nodes", VLEN)
        # self.add("valu", ("vbroadcast", vn_nodes, self.scratch["n_nodes"]))



        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        instrs: list[dict[str, list]] = []
        for (_, val1, _,_, val3) in HASH_STAGES:
            self.vscratch_const(val1)
            self.vscratch_const(val3)

        def dbg(x):
            print(x)
            return x

        short_forest_vals = self.alloc_scratch("short_forest_vals", VLEN)
        short_forest_vec_vals = self.alloc_scratch("short_forest_vec_vals", VLEN * 8)
        instrs.append({"load": [("vload", short_forest_vals, self.scratch["forest_values_p"])]})
        instrs.append({"valu": dbg([("vbroadcast", short_forest_vec_vals + i * VLEN, short_forest_vals + i) for i in range(6)])})
        instrs.append({"valu": dbg([("vbroadcast", short_forest_vec_vals + i * VLEN, short_forest_vals + i) for i in range(6,8)])})

        groups_per_chunk = (SCRATCH_SIZE - self.scratch_ptr) // 70 # 65 + padding
        print(f"groups_per_chunk {groups_per_chunk}")

        chunk_vars = [[ self.alloc_scratch(f"vtmp1_{i}", VLEN),
            self.alloc_scratch(f"vtmp2_{i}", VLEN),
            self.alloc_scratch(f"vtmp3_{i}", VLEN),
            self.alloc_scratch(f"tmp_val_{i}", VLEN),
            self.alloc_scratch(f"tmp_node_val_{i}", VLEN),
            self.alloc_scratch(f"tmp_idx_{i}", VLEN),
            self.alloc_scratch(f"tmp_addr_{i}", VLEN),
            self.alloc_scratch(f"tmp_addr2_{i}", VLEN),
        ] for i in range(groups_per_chunk)]

        n_groups = batch_size // VLEN
        groups = defaultdict(list)
        for chunk in range(cdiv(n_groups , groups_per_chunk)):
            start = chunk * groups_per_chunk
            to_do = min(n_groups - start, groups_per_chunk)
            for i in range(to_do):
                batch_num = start + i
                group_instrs = self.build_group_load(batch_num, 0, *chunk_vars[i])
                for instr in group_instrs:
                    instr["_batch"] = batch_num
                    instr["_round"] = -1
                for round in range(rounds):
                    g = self.build_group(batch_num, round, *chunk_vars[i])
                    for instr in g:
                        instr["_batch"] = batch_num
                        instr["_round"] = round
                    group_instrs.extend(g)
                store_instrs = self.build_group_store(batch_num, *chunk_vars[i])
                for instr in store_instrs:
                    instr["_batch"] = batch_num
                    instr["_round"] = rounds
                group_instrs.extend(store_instrs)
                groups[i].extend(group_instrs)
        # instrs.extend(pack(groups.values()))
        instrs.extend(cross_packer(list(groups.values())))
        
        self.instrs.extend(instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

    def build_group_load(
        self, batch: int, round, 
        vtmp1, vtmp2, vtmp3, tmp_val, tmp_node_val, tmp_idx, tmp_addr, tmp_addr2
    ) -> list[dict]:
        instrs = []

        batch_base = batch * VLEN
        i_batch_const = self.scratch_const(batch_base)
        zeros = self.vscratch_const(0, "zeros")

        # idx = mem[inp_indices_p + i]
        # val = mem[inp_values_p + i]
        instrs.append(
            {
                "alu": [
                    # ("+", tmp_addr, self.scratch["inp_indices_p"], i_batch_const),
                    ("+", tmp_addr2, self.scratch["inp_values_p"], i_batch_const),
                ]
            }
        )
        instrs.append({"load": [("vload", tmp_val, tmp_addr2)], "valu": [("vbroadcast", tmp_idx, zeros)]})
        # instrs.append({"debug": [("vcompare", tmp_idx, [(round, batch_base + j, "idx") for j in range(VLEN)])]})
        # instrs.append({"debug": [("vcompare", tmp_val, [(round, batch_base + j, "val") for j in range(VLEN)])]})

        return instrs

    def build_group_store(
        self, batch: int,  
        vtmp1, vtmp2, vtmp3, tmp_val, tmp_node_val, tmp_idx, tmp_addr, tmp_addr2
    ) -> list[dict]:
        instrs = []

        batch_base = batch * VLEN
        i_batch_const = self.scratch_const(batch_base)

        # mem[inp_indices_p + i] = idx
        # mem[inp_values_p + i] = val
        instrs.append(
            {
                "valu": [
                    ("+", tmp_addr, self.scratch["inp_indices_p"], i_batch_const),
                    ("+", tmp_addr2, self.scratch["inp_values_p"], i_batch_const),
                ]
            }
        )
        instrs.append({"store": [("vstore", tmp_addr2, tmp_val)]})
        # instrs.append({"store": [("vstore", tmp_addr, tmp_idx), ("vstore", tmp_addr2, tmp_val)]})

        return instrs


    def build_group(
        self, batch: int, round, 
        vtmp1, vtmp2, vtmp3, tmp_val, tmp_node_val, tmp_idx, tmp_addr, tmp_addr2
    ) -> list[dict]:
        # 292 valu per batch
        instrs = []

        batch_base = batch * VLEN
        zeros = self.vscratch_const(0, "zeros")
        ones = self.vscratch_const(1, "ones")
        twos = self.vscratch_const(2, "twos")

        # node_val = mem[forest_values_p + idx]

        if round % (self.forest_height + 1) == 0:
            # 1 valu (can be 0 for round 0)
            vvals = self.scratch["short_forest_vec_vals"]
            instrs.append({"valu": [("+", tmp_node_val, zeros, vvals)]})
            # if batch_base % 2 == 0:
            #     instrs.append({"valu": [("+", tmp_node_val, zeros, vvals)]})
            # else:
            #     instrs.append({"alu": [("+", tmp_node_val + j, zeros +j, vvals + j) for j in range(VLEN)]})
        elif round % (self.forest_height + 1) == 1 and not (round == 1 and batch_base < 4):
            # 3 valu
            vvals = self.scratch["short_forest_vec_vals"]
            offset = vtmp1
            diff = vtmp2
            vf1 = vvals + VLEN * 1
            vf2 = vvals + VLEN * 2
            instrs.append({"valu": [
                ("-", offset, tmp_idx, ones), 
                ("-", diff, vf2, vf1)
            ]})
            instrs.append({"valu": [("multiply_add", tmp_node_val, diff, offset, vf1)]})

            # instrs.append({"valu": ["%", vtmp1, ]})

            # Option A: flow-heavy (1 flow + 2 valu)
            # valu:  vmod tmp, val, 2
            # valu:  veq  mask, tmp, 0  
            # flow:  vselect child, mask, one_vec, two_vec   # bottleneck: 1 flow slot
        elif round % (self.forest_height + 1) == 2 and not (round == 2 and batch_base < 4 ):
        # elif round % (self.forest_height + 1) == 2 :
            # 9 valu
            vvals = self.scratch["short_forest_vec_vals"]
            vec_3 = self.vscratch_const(3, "threes")

            vf3 = vvals + VLEN * 3
            vf4 = vvals + VLEN * 4
            vf5 = vvals + VLEN * 5
            vf6 = vvals + VLEN * 6

            offset = vtmp1
            bit0   = vtmp2
            bit1   = vtmp3
            d01    = tmp_addr      # reuse
            d23    = tmp_addr2     # reuse

            # Cycle 1: offset + level-1 diffs [3 valu]
            instrs.append({"valu": [
                ("-", offset, tmp_idx, vec_3), 
                ("-", d01, vf4, vf3),
                ("-", d23, vf6, vf5),
            ]})
            # if batch % 2 == 0:
            #     instrs.append({"valu": [
            #         ("-", offset, tmp_idx, vec_3), 
            #         ("-", d01, vf4, vf3),
            #         ("-", d23, vf6, vf5),
            #     ]})
            # else:
            #     instrs.append({
            #         "valu": [("-", offset, tmp_idx, vec_3), ("-", d01, vf4, vf3)],
            #         "alu": [("-", d23 + j, vf6 + j, vf5 + j) for j in range(VLEN)]
            #     })

            # Cycle 2: bit extraction [2 valu]
            instrs.append({"valu": [
                ("&",  bit0, offset, ones),
                (">>", bit1, offset, ones),   # max val 1, no mask needed
            ]})

            # Cycle 3: level-1 selects [2 valu]
            instrs.append({"valu": [
                ("multiply_add", d01, d01, bit0, vf3),    # r01
                ("multiply_add", d23, d23, bit0, vf5),    # r23
            ]})

            # Cycle 4: level-2 diff [1 valu]
            da = offset  # dead, reuse
            instrs.append({"valu": [
                ("-", da, d23, d01),
            ]})

            # Cycle 5: level-2 select [1 valu]
            instrs.append({"valu": [
                ("multiply_add", tmp_node_val, da, bit1, d01),
            ]})
        else:
            # 8 alu (1 valu equiv)
            # Bad scalar loads and counter incrs to handle non-contiguous value loads
            instrs.append({"alu": [("+", vtmp3 + j, self.scratch["forest_values_p"], tmp_idx + j) for j in range(8)]})
            for j in range(0, 8, 2):
                instrs.append({"load": [("load", tmp_node_val + j, vtmp3 + j), ("load", tmp_node_val + j + 1, vtmp3 + j + 1)]})

        # instrs.append({"debug": [("vcompare", tmp_node_val, [(round, batch_base + j, "node_val") for j in range(VLEN)])]})

        # val = myhash(val ^ node_val)
        # 13 valu
        instrs.append({"valu": [("^", tmp_val, tmp_val, tmp_node_val)]})
        instrs.extend(self.build_hash(tmp_val, vtmp1, vtmp2, round, batch_base))

        # instrs.append({"debug": [("vcompare", tmp_val, [(round, batch_base + j, "hashed_val") for j in range(VLEN)])]})

        # idx = 2*idx + (1 if val % 2 == 0 else 2)
        #     = 2*idx + 1 + (val & 1)
        # 3 valudf
        instrs.append({"valu": [("&", vtmp1, tmp_val, ones)]})
        instrs.append({"valu": [("+", vtmp1, vtmp1, ones)]})
        instrs.append({"valu": [("multiply_add", tmp_idx, tmp_idx, twos, vtmp1)]})

        # instrs.append({"debug": [("vcompare", tmp_idx, [(round, batch_base + j, "next_idx") for j in range(VLEN)])]})

        # idx = 0 if idx >= n_nodes else idx
        if round == 10:
            instrs.append({"valu": [("+", tmp_idx, zeros, zeros)]})

        # instrs.append({"debug": [("vcompare", tmp_idx, [(round, batch_base + j, "wrapped_idx") for j in range(VLEN)])]})

        return instrs

def head_has_work(instr: dict) -> bool:
    return any(instr.get(engine) for engine in SLOT_LIMITS.keys())

def trim_stream(stream: list[dict]) -> None:
    while stream and not head_has_work(stream[0]):
        stream.pop(0)

def pick_frontier_with_engine(
    frontier: list[tuple[int, dict]], start: int, engine: str,
    streams_by_len: list[tuple[int,int]]
) -> int | None:
    n = len(frontier)

    l = len(streams_by_len)
    if l < 8 or (l < 14 and random.random() > 0.8) :
        for _,si in streams_by_len:
            _, head = frontier[si]
            if head.get(engine):
                return si

    for off in range(n):
        i = (start + off) % n
        si, head = frontier[i]
        if head.get(engine):
            return i
    return None

def cross_packer(streams: list[list[dict]]):
    packed = []
    engine_order = ("load", "valu", "alu", "flow", "store")
    streams = [pack_debug(s) for s in streams if s]
    # rr = 0

    stats = {
        "empty_slots": defaultdict(int),
        "blocked_behind_head": defaultdict(int),
        "blocked_by": defaultdict(int),
    }
    stable_ids = list(range(len(streams)))
    while True:
        paired = [(s, sid) for s, sid in zip(streams, stable_ids) if s]
        streams = [s for s, _ in paired]
        stable_ids = [sid for _, sid in paired]
        for s in streams:
            trim_stream(s)
        paired = [(s, sid) for s, sid in zip(streams, stable_ids) if s]
        streams = [s for s, _ in paired]
        stable_ids = [sid for _, sid in paired]
        if not streams:
            break

        # Freeze frontier: si is index into current streams array
        frontier = [(si, streams[si][0]) for si in range(len(streams))]

        instr = defaultdict(list)
        meta = {}  # (engine, slot_idx) -> stable_stream_id
        progressed = True
        rr = 0
        while progressed:
            progressed = False
            for engine in engine_order:
                goal = SLOT_LIMITS[engine] - len(instr[engine])
                while goal > 0:
                    streams_by_len = sorted((-len(stream), si) for si,stream in enumerate(streams))
                    fi = pick_frontier_with_engine(frontier, rr, engine, streams_by_len)
                    if fi is None:
                        break

                    si, head = frontier[fi]
                    stable_id = stable_ids[si]
                    batch_id = head.get("_batch")
                    round_id = head.get("_round")
                    slots = head[engine]
                    take = min(goal, len(slots))

                    base = len(instr[engine])
                    instr[engine].extend(slots[:take])
                    for j in range(take):
                        meta[(engine, base + j)] = (stable_id, batch_id, round_id)
                    del slots[:take]
                    if not slots:
                        head.pop(engine, None)

                    rr = (fi + 1) % len(frontier)
                    goal -= take
                    progressed = True
                # Try valu->alu expansion
        EXPANDABLE_OPS = {"+", "-", "*", "^", "&", "|", "<<", ">>", "%", "<", "==", "//", "cdiv", "!="}

        alu_remaining = SLOT_LIMITS["alu"] - len(instr["alu"])
        while alu_remaining > 0:
            expanded = False
            for fi_idx in range(len(frontier)):
                si, head = frontier[fi_idx]
                stable_id = stable_ids[si]
                batch_id = head.get("_batch")
                round_id = head.get("_round")
                if not head.get("valu"):
                    continue
                for vi, vop in enumerate(head["valu"]):
                    if len(vop) == 4 and vop[0] in EXPANDABLE_OPS:
                        op, dest, a1, a2 = vop
                        head["valu"].pop(vi)
                        if not head["valu"]:
                            head.pop("valu", None)

                        all_8 = [(op, dest+j, a1+j, a2+j) for j in range(VLEN)]
                        take = min(alu_remaining, VLEN)
                        base = len(instr["alu"])
                        instr["alu"].extend(all_8[:take])
                        for j in range(take):
                            meta[("alu", base + j)] = (stable_id, batch_id, round_id)
                        if take < VLEN:
                            head.setdefault("alu", []).extend(all_8[take:])
                        alu_remaining -= take
                        expanded = True
                        break
                if expanded:
                    break
            if not expanded:
                break

        result = {e: ops for e, ops in instr.items() if ops}
        if meta:
            result["_meta"] = meta
        packed.append(result)

        # Analytics: check for blocked work behind partially-consumed heads
        for engine in engine_order:
            empty = SLOT_LIMITS[engine] - len(instr.get(engine, []))
            if empty > 0:
                stats["empty_slots"][engine] += empty
                # Check which streams have this engine's work queued behind a stuck head
                for si, head in frontier:
                    if head and len(streams[si]) > 1:
                        # Head is stuck (not empty), check next instruction
                        next_instr = streams[si][1]
                        if next_instr.get(engine):
                            stats["blocked_behind_head"][engine] += 1
                            # What's blocking the head from advancing?
                            for blocking_eng in engine_order:
                                if head.get(blocking_eng):
                                    stats["blocked_by"][(engine, blocking_eng)] += 1

        # Advance at most one head per stream after the cycle.
        for si, head in frontier:
            if not head_has_work(head):
                streams[si].pop(0)

    # Print analytics
    print("\n=== Cross-packer analytics ===")
    print(f"Total packed cycles: {len(packed)}")
    for engine in ("load", "valu", "alu", "store"):
        empty = stats["empty_slots"][engine]
        total = len(packed) * SLOT_LIMITS[engine]
        pct = 100 * empty / total if total else 0
        blocked = stats["blocked_behind_head"][engine]
        print(f"  {engine:5s}: {empty:5d}/{total:5d} slots empty ({pct:4.1f}%), "
              f"{blocked} stream-cycles blocked behind head")
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

    return packed

def remove_empty_lists(lists: list[list]) -> list[list]:
    return list(filter(lambda x: len(x) != 0, lists))

def pack(batches: list[list[dict]]):
    assert len(batches) > 0

    packed = [{}]
    idx = -1
    for i,batch in enumerate(batches):
        if i == 0:
            drain_list(packed, batch, debug=True)
        drain_list(packed, batch)
    return packed

def drain_list(into_list: list[dict[str, list]], from_list: list[dict[str,list[tuple]]], debug = False):
    # if debug:
    #     print_instrs(from_list[:50], no_debug=False)
    from_list = pack_debug(from_list)
    # if debug:
    #     print_instrs(from_list[:50], no_debug=False)
    if len(into_list) == 0:
        into_list.append({})
    idx = -1
    # if debug:
    #     print_instrs(from_list[:50])
    for from_ in from_list:
        idx = next_with_slots(into_list, idx)
        drain(into_list[idx], from_)
        while not instr_empty(from_):
            idx = next_with_slots(into_list, idx)
            drain(into_list[idx], from_)
    # if debug:
    #     print_instrs(into_list[:50])
            

def print_instrs(xs: list[dict], no_debug = True):
    print("")
    out = []
    for x in xs:
        x = {name:len(slots) for name, slots in x.items()}
        if "debug" in x and no_debug:
            del x["debug"]
        out.append(x)
    print(out)


def next_with_slots(l: list[dict], i:int) -> int:
    for i in range(i+1, len(l)):
        if not instr_full(l[i]):
            return i
    l.append({})
    return len(l) -1


def instr_empty(instr: dict) -> bool: 
    if len(instr) == 0:
        return True
    for ops in instr.values():
        if len(ops) == 0:
            return True
    return False


def instr_full(instr: dict) -> bool: 
    for slot, limit in SLOT_LIMITS.items():
        if slot not in instr:
            return False
        if len(instr[slot]) < limit:
            return False
    return True

def pack_debug(instrs: list[dict]):
    no_debug = []
    dbgs = []
    for instr in instrs:
        if len(instr) == 1 and "debug" in instr:
            dbgs.extend(instr["debug"])
        else:
            if len(dbgs) > 0:
                if "debug" not in instr:
                    instr["debug"] = dbgs 
                else: 
                    instr["debug"].extend(dbgs)
                dbgs = []
            no_debug.append(instr)
    return no_debug


def drain(into: dict[str, list[tuple]], from_: dict[str,list[tuple]]):
    from_empty = True
    for slot, limit in SLOT_LIMITS.items():
        if slot not in from_:
            continue
        if slot not in into:
            into[slot] = []
        available = limit - len(into[slot]) 

        assert available >= 0, "instr packed beyond limit"
        if available == 0:
            continue
        if available >= len(from_[slot]):
            into[slot].extend(from_[slot])
            del from_[slot]
        else:
            into[slot].extend(from_[slot][:available])
            from_[slot] = from_[slot][available:]
            from_empty = False
    instr_full = all(( len(ops) == SLOT_LIMITS[slot] for slot, ops in into.items()))
    return (instr_full, from_empty)


BASELINE = 147_734


def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)] == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_pack(self):
        vtmp1, tmp_val, twos, zeros, ones, vtmp3, tmp_idx = 1,2,3,4,5,6,7
        import copy 
        instrs = []
        instrs.append({"load": [("%", vtmp1, tmp_val, twos)]})
        instrs.append({"valu": [("==", vtmp1, vtmp1, zeros)]})
        instrs.append({"valu": [("!=", vtmp1, vtmp1, zeros)]})
        # instrs.append({"valu": [("*", tmp_idx, tmp_idx, twos)]})
        # instrs.append({"valu": [("+", tmp_idx, tmp_idx, vtmp3)]})

        batches = [copy.deepcopy(instrs) for i in range(3)]

        packed = pack(batches)
        for j, instr in enumerate(packed):
            print(f"\nInstr {j}")
            for name,slot in instr.items():
                print(f"{name}: {len(slot)}")
                print(slot)


        # packed = pack(batches)
        # print("=== Done ===")
        # for instr in packed:
        #     print(instr)
        #     print([(name, len(slot)) for name,slot in instr.items()])



    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)
        # do_kernel_test(2, 2, 64, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
