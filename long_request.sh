#!/usr/bin/env bash
# 长序列 + 滑窗对照测试
#
# 前提：--context-length 必须 > 滑窗(4096)，比如 8192。
#       如果两者相等，滑窗永远不驱逐，测不出东西。
#
# 设计：跑两个请求，needle 位置不同
#   Case A  needle 放在**窗口内**（末尾附近）→ 滑窗层和全注意力层都能看到
#   Case B  needle 放在**窗口外**（最开头）  → 只有全注意力层能看到
#
# 判读：
#   A 答对 + B 答对  → 正常。B 是靠全注意力层检索的，说明长程检索没坏
#   A 答对 + B 答错  → 也正常。正好符合滑窗语义（滑窗层看不到，全注意力层没兜住）
#   A 答错           → 有问题。窗口内的信息都取不到，滑窗或 sink 大概率不对
#   A、B 答案一样且都答错 → 更可疑，像是根本没在看 KV

set -euo pipefail

URL="${URL:-http://localhost:30000/v1/chat/completions}"
TARGET_TOKENS="${TARGET_TOKENS:-7000}"

python - "$URL" "$TARGET_TOKENS" <<'PYEOF'
import json, sys, urllib.request

url, target = sys.argv[1], int(sys.argv[2])

filler = (
    "人工智能模型在长上下文场景下的表现，取决于注意力机制能否在很长的序列里"
    "仍然准确定位到关键信息。滑窗注意力通过限制每个位置能看到的键值范围来控制"
    "计算量和显存占用，但这也意味着超出窗口的历史会被丢弃，除非有额外的机制补偿。"
)
reps = max(1, int(target * 1.6 / len(filler)))
body = filler * reps
n = len(body)

NEEDLE = "代号是「蓝鲸七号」。"
Q = ("问题：上面这段资料里提到的代号是什么？只回答代号本身。")

def run(label, prompt):
    payload = {"model": "IQuest-Q1",
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 64, "temperature": 0}
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.load(r)
    ch = d["choices"][0]
    print("=" * 60)
    print("[%s] usage=%s finish=%s" % (label, json.dumps(d.get("usage")), ch.get("finish_reason")))
    print("content:", (ch["message"].get("content") or "").strip())
    print("reason:", (ch["message"].get("reasoning_content") or "").strip()[:300])

# Case A：needle 在末尾附近，位于窗口内
prompt_a = ("下面是一段资料。\n\n" + body + "\n" + NEEDLE + "\n资料结束。\n\n" + Q)
# Case B：needle 在最开头，对末尾的提问而言已在窗口之外
prompt_b = ("下面是一段资料。\n\n" + NEEDLE + "\n" + body + "\n资料结束。\n\n" + Q)

print("filler chars: %d  (~%d tokens)  窗口=4096" % (n, int(n / 1.6)))
print("A: needle 在末尾（窗口内） | B: needle 在开头（窗口外）")
run("A 窗口内", prompt_a)
run("B 窗口外", prompt_b)
PYEOF
