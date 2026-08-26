# Built-in length datasets

The MoonConv CSV files contain only `arrival_time_ms` and `input_length`
columns. They were derived from the public Hugging Face dataset
`ShwStone/moonconv-wildchat-v4-flash-prefill` at revision
`284a2326bbef3d5107995f52e38eeee9d0ccdb45`.

- Each `*-trace.csv` corresponds to one of `formal_0`, `formal_1`, `formal_2`,
  or `screening`. `arrival_time_ms` is copied from
  `base_arrival_offset_ms`, while `input_length` is copied from
  `actual_input_length`, in `sequence_index` order.

No request text, token IDs, candidate IDs, source trace indices, or request IDs
are included. The simulator preserves zero-gap bursts and all relative arrival
intervals, then scales the complete interval pattern to the configured mean
QPS. The pattern repeats when warmup plus measurement duration exceeds one
trace window.

The source dataset is released under ODC-By 1.0 and contains transformed
information from WildChat-4.8M. See the source dataset card and its
`THIRD_PARTY_NOTICES.md` for the complete attribution requirements.

SHA-256:

- formal 0: `780245505d7c3a8f468ffee546250f49b94025e666a9c0afb0b0af2d806da908`
- formal 1: `8e21a96a5a348ff833afd918e8d572eeb66849872f261bac5f0907fc9b4073e6`
- formal 2: `ec47051b0098cccec2bf1ff2999b8991765022ba8a59ab79f814e3188e445fa0`
- screening: `e66f16ddd11c713390cb46e8e0da666fa4ebf957bfad6aabafccb86bbef1863f`
