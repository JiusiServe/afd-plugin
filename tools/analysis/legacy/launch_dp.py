#!/usr/bin/env python3
"""Per-node launcher: spawn N DP vllm instances using a node-specific template.
相当于官方 examples/external_online_dp/launch_online_dp.py，多了 --template 参数，
这样每个节点可以直接用自己的 p0/p1/d0/d1 模板，不必都改名为 run_dp_template.sh。

用法见同目录脚本头注释 / README。
"""
import argparse
import multiprocessing
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="./run_dp_template.sh",
                        help="本节点的模板脚本路径，例如 ./p0.sh")
    parser.add_argument("--dp-size", type=int, required=True, help="全局 DP 大小。")
    parser.add_argument("--tp-size", type=int, default=1, help="每实例的 TP 大小。")
    parser.add_argument("--dp-size-local", type=int, default=-1, help="本节点实例数。默认=dp-size。")
    parser.add_argument("--dp-rank-start", type=int, default=0, help="本节点起始 dp_rank。")
    parser.add_argument("--dp-address", type=str, required=True, help="DP broker 的 IP。")
    parser.add_argument("--dp-rpc-port", type=str, default="12345", help="DP broker 的 RPC 端口。")
    parser.add_argument("--vllm-start-port", type=int, default=9000, help="vllm 起始端口。")
    return parser.parse_args()


args = parse_args()
dp_size = args.dp_size
tp_size = args.tp_size
dp_size_local = dp_size if args.dp_size_local == -1 else args.dp_size_local
dp_rank_start = args.dp_rank_start
dp_address = args.dp_address
dp_rpc_port = args.dp_rpc_port
vllm_start_port = args.vllm_start_port
template_path = args.template

if not os.path.exists(template_path):
    print(f"Template file {template_path} does not exist.")
    sys.exit(1)

num_cards = dp_size_local * tp_size


def run_command(visible_devices, dp_rank, vllm_engine_port):
    command = [
        "bash", template_path,
        visible_devices,
        str(vllm_engine_port),
        str(dp_size),
        str(dp_rank),
        dp_address,
        dp_rpc_port,
        str(tp_size),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    processes = []
    for i in range(dp_size_local):
        dp_rank = dp_rank_start + i
        vllm_engine_port = vllm_start_port + i
        visible_devices = ",".join(str(x) for x in range(i * tp_size, (i + 1) * tp_size))
        process = multiprocessing.Process(
            target=run_command,
            args=(visible_devices, dp_rank, vllm_engine_port),
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
