name=${1:?usage: run_itask.sh <task-name> <model-image>}
model=${2:?usage: run_itask.sh <task-name> <model-image>}
user=wb02363348
#image=hcr.met:a-wulan01.hw-wulan.local/antsys/vllm:release_0.18.0_0415_202603221557_aarch64
image=hcr.meta-wulan01.hw-wulan.local/antsys/vllm:nightly-main-a3-openeuler-20260801230444_aarch64
itask create --name ${name} --user ${user} --model ${model} --image ${image} --hostnet --16card --skip-sync --type a3 --workdir /a3_inference/itask/workdir/wb02363348/bjf_afd/code
# itask create --name ${name} --user ${user} --image ${image} --hostnet --16card --skip-sync --type a3 --workdir /a3_inference/itask/workdir/hk02335263/jcz_afd_100/code
# itask create --name ${name} --user ${user} --image ${image} --hostnet --16card --skip-sync --type a3
