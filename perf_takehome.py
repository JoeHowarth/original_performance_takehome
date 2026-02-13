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

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append({"valu": [(op1, vtmp1, val_hash_addr, self.vscratch_const(val1))]})
            slots.append({"valu": [(op3, vtmp2, val_hash_addr, self.vscratch_const(val3))]})
            slots.append({"valu": [(op2, val_hash_addr, vtmp1, vtmp2)]})
            slots.append({"debug": [("vcompare", val_hash_addr, [(round, batch_base + j, "hash_stage", hi) for j in range(VLEN)])]})
        return slots


    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

        vtmp1 = self.alloc_scratch("vtmp1", VLEN)
        vtmp2 = self.alloc_scratch("vtmp2", VLEN)
        vtmp3 = self.alloc_scratch("vtmp3", VLEN)

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

        body = []  # array of slots

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx", VLEN)
        tmp_val = self.alloc_scratch("tmp_val", VLEN)
        tmp_node_val = self.alloc_scratch("tmp_node_val", VLEN)
        tmp_addr = self.alloc_scratch("tmp_addr", VLEN)


        instrs: list[dict[str, list]] = []

        for round in range(rounds):
            for i in range(batch_size // 8):
                batch_base = i * VLEN
                i_batch_const = self.scratch_const(batch_base)

                # idx = mem[inp_indices_p + i]
                instrs.append({"alu": [("+", tmp_addr, self.scratch["inp_indices_p"], i_batch_const)]})
                instrs.append({"load": [("vload", tmp_idx, tmp_addr)]})
                instrs.append({"debug": [("vcompare", tmp_idx, [(round, batch_base + j, "idx") for j in range(VLEN)])]})
                # val = mem[inp_values_p + i]
                instrs.append({"alu": [("+", tmp_addr, self.scratch["inp_values_p"], i_batch_const)]})
                instrs.append({"load": [("vload", tmp_val, tmp_addr)]})
                instrs.append({"debug": [("vcompare", tmp_val, [(round, batch_base + j, "val") for j in range(VLEN)])]})

                # node_val = mem[forest_values_p + idx]
                # Bad scalar loads and counter incrs to handle non-contiguous value loads
                for j in range(8):
                    instrs.append({"alu": [("+", tmp3, self.scratch["forest_values_p"], tmp_idx + j)]})
                    instrs.append({"load": [("load", tmp_node_val + j, tmp3)]})

                instrs.append({"debug": [("vcompare", tmp_node_val, [(round, batch_base + j, "node_val") for j in range(VLEN)])]})
                # val = myhash(val ^ node_val)
                instrs.append({"valu": [("^", tmp_val, tmp_val, tmp_node_val)]})
                instrs.extend(self.build_hash(tmp_val, vtmp1, vtmp2, round, batch_base))
                instrs.append({"debug": [("vcompare", tmp_val, [(round, batch_base + j, "hashed_val") for j in range(VLEN)])]})
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                instrs.append({"valu": [("%", tmp1, tmp_val, twos)]})
                instrs.append({"valu": [("==", tmp1, tmp1, zeros)]})
                instrs.append({"flow": [("vselect", tmp3, tmp1, ones, twos)]})
                instrs.append({"valu": [("*", tmp_idx, tmp_idx, twos)]})
                instrs.append({"valu": [("+", tmp_idx, tmp_idx, tmp3)]})
                instrs.append({"debug": [("vcompare", tmp_idx, [(round, batch_base + j, "next_idx") for j in range(VLEN)])]})
                # idx = 0 if idx >= n_nodes else idx
                instrs.append({"valu": [("<", tmp1, tmp_idx, vn_nodes)]})
                instrs.append({"flow": [("vselect", tmp_idx, tmp1, tmp_idx, zeros)]})
                instrs.append({"debug": [("vcompare", tmp_idx, [(round, batch_base + j, "wrapped_idx") for j in range(VLEN)])]})
                # mem[inp_indices_p + i] = idx
                instrs.append({"valu": [("+", tmp_addr, self.scratch["inp_indices_p"], i_batch_const)]})
                instrs.append({"store": [("vstore", tmp_addr, tmp_idx)]})
                # mem[inp_values_p + i] = val
                instrs.append({"valu": [("+", tmp_addr, self.scratch["inp_values_p"], i_batch_const)]})
                instrs.append({"store": [("vstore", tmp_addr, tmp_val)]})

        # Scalar 
        # for round in range(rounds):
        #     for i in range(batch_size):
        #         i_const = self.scratch_const(i)
        #         # idx = mem[inp_indices_p + i]
        #         body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
        #         body.append(("load", ("load", tmp_idx, tmp_addr)))
        #         body.append(("debug", ("compare", tmp_idx, (round, i, "idx"))))
        #         # val = mem[inp_values_p + i]
        #         body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
        #         body.append(("load", ("load", tmp_val, tmp_addr)))
        #         body.append(("debug", ("compare", tmp_val, (round, i, "val"))))
        #         # node_val = mem[forest_values_p + idx]
        #         body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
        #         body.append(("load", ("load", tmp_node_val, tmp_addr)))
        #         body.append(("debug", ("compare", tmp_node_val, (round, i, "node_val"))))
        #         # val = myhash(val ^ node_val)
        #         body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
        #         body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
        #         body.append(("debug", ("compare", tmp_val, (round, i, "hashed_val"))))
        #         # idx = 2*idx + (1 if val % 2 == 0 else 2)
        #         body.append(("alu", ("%", tmp1, tmp_val, two_const)))
        #         body.append(("alu", ("==", tmp1, tmp1, zero_const)))
        #         body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
        #         body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
        #         body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
        #         body.append(("debug", ("compare", tmp_idx, (round, i, "next_idx"))))
        #         # idx = 0 if idx >= n_nodes else idx
        #         body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
        #         body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
        #         body.append(("debug", ("compare", tmp_idx, (round, i, "wrapped_idx"))))
        #         # mem[inp_indices_p + i] = idx
        #         body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
        #         body.append(("store", ("store", tmp_addr, tmp_idx)))
        #         # mem[inp_values_p + i] = val
        #         body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
        #         body.append(("store", ("store", tmp_addr, tmp_val)))

        # body_instrs = self.build(body)
        self.instrs.extend(instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

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
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
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
        # do_kernel_test(10, 16, 256, trace=True, prints=False)
        do_kernel_test(2, 1, 8, trace=True, prints=False)

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
