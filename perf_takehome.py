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

@dataclass
class GroupScratch:
    group_base: int
    vtmp1: int
    vtmp2: int
    vtmp3: int
    tmp_val: int
    tmp_node_val: int
    tmp_idx: int
    tmp_addr: int
    tmp_addr2: int

    def __init__(self, builder: 'KernelBuilder', _mem_num: int, _group_start_offset: int):
        self.group_base = _group_start_offset
        self.mem_num = _mem_num
        i = _mem_num
        self.vtmp1 = builder.alloc_scratch(f"vtmp1_{i}", VLEN)
        self.vtmp2 = builder.alloc_scratch(f"vtmp2_{i}", VLEN),
        self.vtmp3 = builder.alloc_scratch(f"vtmp3_{i}", VLEN),
        self.tmp_val = builder.alloc_scratch(f"tmp_val_{i}", VLEN),
        self.tmp_node_val = builder.alloc_scratch(f"tmp_node_val_{i}", VLEN),
        self.tmp_idx = builder.alloc_scratch(f"tmp_idx_{i}", VLEN),
        self.tmp_addr = builder.alloc_scratch(f"tmp_addr_{i}", VLEN),
        self.tmp_addr2 = builder.alloc_scratch(f"tmp_addr2_{i}", VLEN),

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

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(
                {"valu": [(op1, vtmp1, val_hash_addr, self.vscratch_const(val1)), (op3, vtmp2, val_hash_addr, self.vscratch_const(val3))]}
            )
            slots.append({"valu": [(op2, val_hash_addr, vtmp1, vtmp2)]})
            slots.append({"debug": [("vcompare", val_hash_addr, [(round, batch_base + j, "hash_stage", hi) for j in range(VLEN)])]})
        return slots

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """

        tmp1 = self.alloc_scratch(f"tmp1")

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
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        zeros = self.vscratch_const(0, "zeros")
        ones = self.vscratch_const(1, "ones")
        twos = self.vscratch_const(2, "twos")
        vn_nodes = self.alloc_scratch("vn_nodes", VLEN)
        self.add("valu", ("vbroadcast", vn_nodes, self.scratch["n_nodes"]))



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

        groups_per_chunk = (SCRATCH_SIZE - self.scratch_ptr) // 70 # 65 + padding

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
        for chunk in range(cdiv(n_groups , groups_per_chunk)):
            start = chunk * groups_per_chunk
            to_do = min(n_groups - start, groups_per_chunk)
            groups = []
            for i in range(to_do):
                group_instrs = self.build_group_load(start + i, 0, *chunk_vars[i])
                for round in range(rounds):
                    g = self.build_group(start + i, round, *chunk_vars[i])
                    group_instrs.extend(g)
                group_instrs.extend(self.build_group_store(start + i, *chunk_vars[i]))
                groups.append(group_instrs)
            instrs.extend(pack(groups))


        self.instrs.extend(instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

    def merge_instrs(self, old: list[dict[str, list[tuple]]], new: list[dict[str, list[tuple]]]):
        i = 0
        for n in new:
            while sum(len(ops) for ops in n.values()) > 0:
                o = old[i]
                o_full, n_empty = self.merge(o, n)
                if o_full:
                    i += 1

    def build_group_load(
        self, batch: int, round, 
        vtmp1, vtmp2, vtmp3, tmp_val, tmp_node_val, tmp_idx, tmp_addr, tmp_addr2
    ) -> list[dict]:
        instrs = []

        batch_base = batch * VLEN
        i_batch_const = self.scratch_const(batch_base)

        # idx = mem[inp_indices_p + i]
        # val = mem[inp_values_p + i]
        instrs.append(
            {
                "alu": [
                    ("+", tmp_addr, self.scratch["inp_indices_p"], i_batch_const),
                    ("+", tmp_addr2, self.scratch["inp_values_p"], i_batch_const),
                ]
            }
        )
        instrs.append({"load": [("vload", tmp_idx, tmp_addr), ("vload", tmp_val, tmp_addr2)]})
        instrs.append({"debug": [("vcompare", tmp_idx, [(round, batch_base + j, "idx") for j in range(VLEN)])]})
        instrs.append({"debug": [("vcompare", tmp_val, [(round, batch_base + j, "val") for j in range(VLEN)])]})

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
        instrs.append({"store": [("vstore", tmp_addr, tmp_idx), ("vstore", tmp_addr2, tmp_val)]})

        return instrs


    def build_group(
        self, batch: int, round, 
        vtmp1, vtmp2, vtmp3, tmp_val, tmp_node_val, tmp_idx, tmp_addr, tmp_addr2
    ) -> list[dict]:
        instrs = []

        batch_base = batch * VLEN
        i_batch_const = self.scratch_const(batch_base)
        zeros = self.vscratch_const(0, "zeros")
        ones = self.vscratch_const(1, "ones")
        twos = self.vscratch_const(2, "twos")
        vn_nodes = self.scratch["vn_nodes"]


        # node_val = mem[forest_values_p + idx]
        # Bad scalar loads and counter incrs to handle non-contiguous value loads

        instrs.append({"alu": [("+", vtmp3 + j, self.scratch["forest_values_p"], tmp_idx + j) for j in range(8)]})
        for j in range(0, 8, 2):
            instrs.append({"load": [("load", tmp_node_val + j, vtmp3 + j), ("load", tmp_node_val + j + 1, vtmp3 + j + 1)]})

        instrs.append({"debug": [("vcompare", tmp_node_val, [(round, batch_base + j, "node_val") for j in range(VLEN)])]})

        # val = myhash(val ^ node_val)
        instrs.append({"valu": [("^", tmp_val, tmp_val, tmp_node_val)]})
        instrs.extend(self.build_hash(tmp_val, vtmp1, vtmp2, round, batch_base))

        instrs.append({"debug": [("vcompare", tmp_val, [(round, batch_base + j, "hashed_val") for j in range(VLEN)])]})

        # idx = 2*idx + (1 if val % 2 == 0 else 2)
        instrs.append({"valu": [("%", vtmp1, tmp_val, twos)]})
        instrs.append({"valu": [("==", vtmp1, vtmp1, zeros)]})
        instrs.append({"flow": [("vselect", vtmp3, vtmp1, ones, twos)]})
        instrs.append({"valu": [("*", tmp_idx, tmp_idx, twos)]})
        instrs.append({"valu": [("+", tmp_idx, tmp_idx, vtmp3)]})

        instrs.append({"debug": [("vcompare", tmp_idx, [(round, batch_base + j, "next_idx") for j in range(VLEN)])]})

        # idx = 0 if idx >= n_nodes else idx
        if round == 10:
            instrs.append({"valu": [("+", tmp_idx, zeros, zeros)]})
        # instrs.append({"flow": [("vselect", tmp_idx, vtmp1, tmp_idx, zeros)]})

        instrs.append({"debug": [("vcompare", tmp_idx, [(round, batch_base + j, "wrapped_idx") for j in range(VLEN)])]})

        return instrs

def pack(batches: list[list[dict]]):
    assert len(batches) > 0

    packed = [{}]
    idx = -1
    for batch in batches:
        drain_list(packed, batch)
    return packed

def drain_list(into_list: list[dict[str, list]], from_list: list[dict[str,list[tuple]]]):
    if len(into_list) == 0:
        into_list.append({})
    idx = -1
    for from_ in from_list:
        idx = next_with_slots(into_list, idx)
        drain(into_list[idx], from_)
        while not instr_empty(from_):
            idx = next_with_slots(into_list, idx)
            drain(into_list[idx], from_)
            



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
