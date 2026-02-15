I've gotten to 1727
- full valu utilizaton from cycles ~160 to ~1300, then (164+236+287+347+370+390)/390/6 = 76.6% utilization after cycle 1300
- After initialization, loads are 1302 / 1645 => 80% util
- startup costs ~80 cycles, this can be compressed a lot through better packing and using flow to load consts too
- alu is pretty under utilized, but not 0% either

I think we need to get to 100% of load engine after initialization period

There's a few directions the packer can go:
1. Prioritize loads somehow
    a. Maybe instead of greedy placement by group stream (chunk idx? it's 1 or 2 groups per stream), we look over all eligible instrs and pick loads. 
    b. Then we fill valu, alu, flow and store. If gaps left in alu, take more valu from 'ready' frontier and decompose into alu slots.

