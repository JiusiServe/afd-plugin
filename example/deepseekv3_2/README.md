# DeepSeek-V3.2 AFD Examples

End-to-end launch scripts for running DeepSeek-V3.2 with the AFD
(Attention-FFN Disaggregation) plugin on vLLM `v0.19.1`.

## Prerequisites

- NPUs (tested against Ascend A3). The physical machines must be located on the same WLAN, with network connectivity. All NPUs must be interconnected. Intra-node connectivity is via HCCS, and inter-node connectivity is via RDMA.
- vLLM `v0.19.1` and the `afd-plugin` package installed in the same
  environment (see repository root `AGENTS.md`).
- DeepSeek-V3.2 weights on disk. All scripts default to
  `/path/model_weights/dsv3_2`; override with
  `MODEL=...` when launching.

## prefill decode disaggregation

We can run the following scripts to launch a server on the prefiller/decoder node, respectively.

### Start the service

```bash
# on 190.0.0.1
python launch_dp.py --template ./p0.sh --dp-size 2 --tp-size 8 --dp-size-local 2 --dp-rank-start 0 --dp-address 190.0.0.1 --dp-rpc-port 18432 --vllm-start-port 8006
# on 190.0.0.2
python launch_dp.py --template ./p1.sh --dp-size 2 --tp-size 8 --dp-size-local 2 --dp-rank-start 0 --dp-address 190.0.0.2 --dp-rpc-port 18432 --vllm-start-port 8006
# on 190.0.0.3
python launch_dp.py --template ./d0.sh --dp-size 32 --tp-size 1 --dp-size-local 16 --dp-rank-start 0 --dp-address 190.0.0.3 --dp-rpc-port 18432 --vllm-start-port 8006
# on 190.0.0.4
python launch_dp.py --template ./d1.sh --dp-size 32 --tp-size 1 --dp-size-local 16 --dp-rank-start 16 --dp-address 190.0.0.4 --dp-rpc-port 18432 --vllm-start-port 8006
```

### Example Proxy for Deployment

```bash
python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --port 1999 \
  --host 190.0.0.1 \
  --prefiller-hosts \
    190.0.0.1 \
    190.0.0.1 \
    190.0.0.2 \
    190.0.0.2 \
  --prefiller-ports \
    8006 8007 8006 8007 \
  --decoder-hosts \
    190.0.0.3 \
    190.0.0.3 \
    190.0.0.3 \
    190.0.0.3 \
    190.0.0.3 \
    190.0.0.3 \
    190.0.0.3 \
    190.0.0.3 \
    190.0.0.3 \
    190.0.0.3 \
    190.0.0.3 \
    190.0.0.3 \
    190.0.0.3 \
    190.0.0.3 \
    190.0.0.3 \
    190.0.0.3 \
    190.0.0.4 \
    190.0.0.4 \
    190.0.0.4 \
    190.0.0.4 \
    190.0.0.4 \
    190.0.0.4 \
    190.0.0.4 \
    190.0.0.4 \
    190.0.0.4 \
    190.0.0.4 \
    190.0.0.4 \
    190.0.0.4 \
    190.0.0.4 \
    190.0.0.4 \
    190.0.0.4 \
    190.0.0.4 \
  --decoder-ports  \
    8006 8007 8008 8009 8010 8011 8012 8013 8014 8015 8016 8017 8018 8019 8020 8021 \
    8006 8007 8008 8009 8010 8011 8012 8013 8014 8015 8016 8017 8018 8019 8020 8021
```

## prefill decode colocated

We can run the following scripts to launch

### Start the service

```bash
cd /path/afd-plugin/example/deepseekv3_2/prefill_decode_colocated
# on 190.0.0.1
bash 16a16f_graph_dp16tp1_attn.sh
# on 190.0.0.2
bash 16a16f_graph_dp16tp1_ffn.sh
```

### Eager mode

Add `--enforce-eager` to the `vllm serve` call in `16a16f_graph_dp16tp1_attn.sh` and `16a16f_graph_dp16tp1_ffn.sh` respectively:

```bash
# on 190.0.0.1
bash 16a16f_graph_dp16tp1_attn.sh   # vllm serve ... --enforce-eager
# on 190.0.0.2
bash 16a16f_graph_dp16tp1_ffn.sh    # vllm serve ... --enforce-eager
```

## Benchmark

We recommend use aisbench tool to assess performance. [aisbench](https://github.com/AISBench/benchmark.git) Execute the following commands to install aisbench

```shell
git clone https://github.com/AISBench/benchmark.git
cd benchmark/
pip3 install -e ./
```

You need to cancel the http proxy before assessing performance, as following

```shell
# unset proxy
unset http_proxy
unset https_proxy
```

- You can place your datasets in the dir: `benchmark/ais_bench/datasets`
- You can change the configuration in the dir :`benchmark/ais_bench/benchmark/configs/models/vllm_api` Take the ``vllm_api_stream_chat.py`` for examples

```python
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="/path/model_weights/dsv3_2",
        model="dsv3_2",
        request_rate = 14,
        retry = 2,
        host_ip = "190.0.0.1", # Proxy service host IP
        host_port = 8000,  # Proxy service Port
        max_out_len = 10,
        batch_size=768,
        trust_remote_code=True,
        generation_kwargs = dict(
            temperature = 0,
            seed = 1024,
            ignore_eos=False,
        )
    )
]
```

- Take synthetic dataset for example, execute the following commands to assess performance.

```shell
ais_bench --models vllm_api_stream_chat --datasets synthetic_gen_string --debug --mode perf
```
