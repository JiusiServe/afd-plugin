# AFD Recipes

This directory contains reproducible deployment and benchmark recipes for the
AFD connectors supported by this repository.

## Directory layout

Recipes are organized by hardware backend, connector, and model:

```text
recipe/
├── gpu/
│   └── p2p_nccl/
│       └── deepseek_v2_lite/
└── npu/
    ├── cam_async/
    │   └── deepseek_v3_2/
    └── camp2p/
        └── deepseek_v3_2/
```

Directory names use lowercase snake case:

- Hardware backend: `gpu` or `npu`.
- Connector: the short connector name, such as `p2p_nccl`, `camp2p`, or
  `cam_async`.
- Model: the model family or variant, such as `deepseek_v2_lite` or
  `deepseek_v3_2`.

## Available recipes

| Hardware | Connector | Model | Recommended stage | Recipe |
| --- | --- | --- | --- | --- |
| GPU | `P2pNcclAFDConnector` | DeepSeek-V2-Lite | Decode | [Launch examples](gpu/p2p_nccl/deepseek_v2_lite/README.md) |
| Ascend NPU | `CAMP2pAFDConnector` | DeepSeek-V3.2 | Decode | [Synchronous decode](npu/camp2p/deepseek_v3_2/README.md) |
| Ascend NPU | `CAMAsyncAFDConnector` | DeepSeek-V3.2 | Prefill | [Asynchronous prefill](npu/cam_async/deepseek_v3_2/README.md) |

Open the model-level README before running a recipe. It documents the required
hardware and container image, topology, environment variables, launch order,
and known limitations. Unless a recipe says otherwise, run its commands from
the repository root.

## Adding a recipe

Add new content under `recipe/<hardware>/<connector>/<model>/`. Each model
directory should contain a `README.md` describing prerequisites, topology,
launch commands, validation steps, and limitations. Keep model-specific launch
scripts, configuration files, and result images in the same directory so the
recipe can be moved or linked as one unit.
