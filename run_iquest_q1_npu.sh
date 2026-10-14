#!/usr/bin/env bash
# IQuest-Q1 on Ascend NPU — v0 启动脚本
#
# 基线：sglang @ 27b68303（全量克隆），改动见 python/sglang/srt/hardware_backend/npu/
#       attention/ascend_backend.py 和 python/sglang/srt/models/iquest_q1.py
#
# 与官方 cookbook（H200 / fa3）的差异，逐条说明见文件末尾。

set -euo pipefail

# ---------------------------------------------------------------------------
# 1) 路径 / 端口
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/model/IQuest-Q1}"     # 官方 checkpoint（BF16，175 分片）
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
SERVED_NAME="IQuest-Q1"

# ---------------------------------------------------------------------------
# 2) 并行度
# ---------------------------------------------------------------------------
# 权重 320B × 2B = 640GB，TP=8 时每卡 ~80GB。
# 96GB 卡 → 占 83%，很紧；上下文必须压低（见 max-model-len）。
TP="${TP:-8}"
# 注意：--ep-size 在 sglang 里不存在（我在 server_args.py 里查过，没有这个字段）。
# 专家并行的开关名请用下面这条确认后再加，别猜：
#   python -m sglang.launch_server --help | grep -iE "ep|expert|moe"
# NPU 上常见的是 --enable-ep-moe 一类；不传也能跑，只是 MoE 走 TP 分片。

# ---------------------------------------------------------------------------
# 3) 显存
# ---------------------------------------------------------------------------
# 权重就吃掉 ~83%，留给 KV / 激活 / 框架的余量很小。
# 先给 0.8 试；OOM 就往下调，或者把 max-model-len 再压低。
MEM_FRACTION="${MEM_FRACTION:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"   # 先别上 512K，KV 预算没规划过

# ---------------------------------------------------------------------------
# 4) 环境
# ---------------------------------------------------------------------------
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# 我们的 learned-sink 路径不依赖 ASCEND_USE_FIA/FIA2 分支（模型每层都显式
# 传 return_lse=True，直接走新增的 _iquest_attn_with_lse），所以这两个开关
# 保持默认即可。如果排查别的问题需要开 FIA，再打开下面两行：
# export ASCEND_USE_FIA=1

# triton-ascend 的 deprecation 警告会刷屏，屏蔽掉
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::UserWarning}"

# ---------------------------------------------------------------------------
# 5) 启动
# ---------------------------------------------------------------------------
# ★ --disable-cuda-graph 是必须的 ★
#   forward_decode_graph 没有 return_lse 处理，图模式下的 decode 会绕开
#   我们的路径，拿不到 lse → sink 修正失效（甚至报错）。v0 一律禁图。
#
# --disable-prefill-cuda-graph 沿用官方配方（官方也禁了 prefill 图）。
#
# --enable-torch-compile / --attention-backend fa3 是 CUDA 专属，NPU 不用。

python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --tp "${TP}" \
  --mem-fraction-static "${MEM_FRACTION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --disable-cuda-graph \
  --disable-prefill-cuda-graph \
  --reasoning-parser iquest_q1 \
  --tool-call-parser iquest_q1 \
  --host "${HOST}" \
  --port "${PORT}"

# ---------------------------------------------------------------------------
# 备注 / 排障
# ---------------------------------------------------------------------------
# [1] 没开 MTP。
#     官方 low-latency 配方带 EAGLE 递归 draft（mtp/ 子目录）。NPU 上
#     iquest_q1_mtp.py 的 attention 同样需要 lse，但走的是 draft 模型那条
#     链路，我们还没验证。v0 先把 target 跑通，MTP 后面单独做。
#
# [2] 如果报 "mixed" 相关的错（forward_mixed 没有 return_lse），
#     说明命中了 NPU 的 mixed prefill+decode 批，加：
#       --chunked-prefill-size -1
#
# [3] 显存：640GB 权重在 8×96GB 上占 83%，很容易被激活值顶爆。
#     真的 OOM 就先加卡（16 卡 = 42%），或者自己量化。
#
# [4] 冒烟测试：
#   curl http://127.0.0.1:30000/v1/chat/completions \
#     -H 'Content-Type: application/json' \
#     -d '{"model":"IQuest-Q1","messages":[{"role":"user","content":"Who are you?"}],
#          "max_tokens":64,"temperature":0}'
#   思维链应该被 reasoning parser 切到 reasoning_content 字段，
#   content 里只剩正文（这是 vLLM 那边没做到的）。
#
# [5] 验证 sink 真的生效（最关键的一步）：
#     把 iquest_q1.py 里 enable_sink_attention 强制置 False 再跑同一个
#     temperature=0 的 prompt，输出必须不一样。一样就说明 sink 没起作用。
