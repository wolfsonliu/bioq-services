#!/usr/bin/env bash
# 预取 BindCraft2 需要的 7 个 AlphaFold 检查点（上游称 ~5.3 GB，实测归档
# 5,587,968,000 B ≈ 5.59 GB），只解出这 7 个文件，其余丢弃。
#
# 默认落到 services/bindcraft2-server/weights/（stage 目录）；
# 正式部署直接下到 NAS：
#
#   WEIGHTS_DST=/data/models/bindcraft2/alphafold \
#       ./services/bindcraft2-server/scripts/fetch_weights.sh
#
# 若 NAS 上 /data/models/alphafold 已由 alphafold-server 就位，则不必跑本脚本：
#   ln -s /data/models/alphafold /data/models/bindcraft2/alphafold
# （上游查找顺序：<dir>/params/params_<model>.npz 或 <dir>/params_<model>.npz。）
#
# 本脚本**不**处理 ProteinMPNN 权重：它们随包分发（vendor.sh 已校验）。
#
# 归档布局（实测 alphafold_params_2022-12-06.tar）：16 个成员——15 个
# params_model_*.npz + LICENSE，全部在归档**顶层**（没有 params/ 前缀），
# 每个 npz 约 356 MiB，远高于上游 100 MiB 的"未完成"下限。

set -euo pipefail

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="${WEIGHTS_DST:-$PROJECT_ROOT/services/bindcraft2-server/weights/alphafold}"
URL="${ALPHAFOLD_PARAMS_URL:-https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar}"

MODELS=(
    model_1_multimer_v3 model_2_multimer_v3 model_3_multimer_v3
    model_4_multimer_v3 model_5_multimer_v3
    model_1_ptm model_2_ptm
)

# 目标已存在但不是目录：立刻失败，避免白下 5.59 GB 才在 mkdir 处报错。
if [ -e "$DST" ] && [ ! -d "$DST" ]; then
    echo "ERROR: WEIGHTS_DST exists but is not a directory: $DST" >&2
    exit 1
fi

TMP="$(mktemp -d -t bc2-weights.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

echo "Downloading AlphaFold parameters -> $TMP"
echo "  $URL"
mkdir -p "$TMP/params"

# -C - 断点续传（对齐兄弟服务的 wget -c）；归档支持 Range（accept-ranges: bytes）。
for attempt in 1 2 3; do
    if curl -fL -C - --retry 3 --retry-delay 5 -o "$TMP/alphafold_params.tar" "$URL"; then
        break
    fi
    [ "$attempt" = "3" ] && { echo "ERROR: download failed after 3 attempts" >&2; exit 1; }
    echo "  download failed, retrying ..."
    sleep $((attempt * 10))
done

# 只解出需要的 7 个成员。成员名在归档顶层，故用精确名匹配；**不加** `*/` 前缀
# 候选——GNU tar 对任一未命中的模式都会以退出码 2 失败（实测），会让下面的
# `||` 分支在下载完全成功时也误触发。
members=()
for model in "${MODELS[@]}"; do
    members+=(--wildcards "params_${model}.npz")
done

echo "Extracting 7 of the archive's checkpoints ..."
tar -xf "$TMP/alphafold_params.tar" -C "$TMP/params" "${members[@]}" 2>/dev/null || {
    echo "ERROR: none of the expected members matched; archive layout changed?" >&2
    tar -tf "$TMP/alphafold_params.tar" | head -20 >&2
    exit 1
}

# tar 会把 members 解到各自的目录层级；统一摊平到 params/。
find "$TMP/params" -mindepth 2 -name 'params_*.npz' -exec mv -f {} "$TMP/params/" \;
find "$TMP/params" -mindepth 1 -type d -exec rm -rf {} +

mkdir -p "$DST/params"
for model in "${MODELS[@]}"; do
    src="$TMP/params/params_${model}.npz"
    [ -f "$src" ] || { echo "ERROR: missing params_${model}.npz in archive" >&2; exit 1; }
    # 上游用 100 MB 下限判"未完成"——小于此值的文件视为损坏
    size="$(stat -c%s "$src")"
    [ "$size" -ge $((100 << 20)) ] || { echo "ERROR: ${model} truncated ($size bytes)" >&2; exit 1; }
    mv -f "$src" "$DST/params/params_${model}.npz"
done

echo "Done. 7 checkpoints at $DST/params/"
du -sh "$DST"
