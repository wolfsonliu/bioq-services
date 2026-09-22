#!/usr/bin/env bash
# 把上游 BindCraft2 源码 vendor 到 services/bindcraft2-server/upstream/，
# 固定 SHA，使 `docker build` 不访问网络。每次构建前跑一次（升级时重跑）：
#
#   ./services/bindcraft2-server/scripts/vendor.sh
#
# CN 网络下可用镜像：
#   BINDCRAFT2_REPO=https://ghproxy.cn/https://github.com/PacesaLab/BindCraft2 \
#       ./services/bindcraft2-server/scripts/vendor.sh
#
# 升级 pin：改 BINDCRAFT2_SHA。
#
# 注意：**不要**排除 bindcraft/weights/proteinmpnn/ —— 三个变体的 .npz 是
# 上游 pyproject 的 package-data（~20 MB），随包分发；排除它们会让 design 在
# ProteinMPNN 重设计阶段才失败。

set -euo pipefail

BINDCRAFT2_REPO="${BINDCRAFT2_REPO:-https://github.com/PacesaLab/BindCraft2}"
BINDCRAFT2_SHA="${BINDCRAFT2_SHA:-5342aefa18dedad653f7a5f6dbee1e566ca24d8f}"  # v1.0.1

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/bindcraft2-server/upstream"
TMP="$(mktemp -d -t bc2-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$BINDCRAFT2_REPO" "$TMP/repo"; then
        break
    fi
    [ "$i" = "5" ] && {
        echo "ERROR: git clone failed after 5 attempts" >&2
        exit 1
    }
    echo "  clone failed, retrying in $((i*10))s ..."
    sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$BINDCRAFT2_SHA"
actual="$(git rev-parse HEAD)"
if [[ "$actual" != "$BINDCRAFT2_SHA" ]]; then
    echo "ERROR: HEAD mismatch after checkout (got $actual, expected $BINDCRAFT2_SHA)" >&2
    exit 1
fi
rm -rf .git

# --delete 清掉上一次 vendor 的陈旧文件。docs/ 与 containers/ 不进镜像
# （Dockerfile 会再剪一次），但留在 upstream/ 便于本地比对上游文档。
rsync -a --delete \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv' --exclude='results' \
    "$TMP/repo/" "$DST/"

# 自检：ProteinMPNN 权重必须在（少了会在重设计阶段才炸）
for variant in neutral negative positive; do
    f="$DST/bindcraft/weights/proteinmpnn/weights_${variant}/v_48_020.npz"
    [ -f "$f" ] || { echo "ERROR: missing shipped ProteinMPNN weights: $f" >&2; exit 1; }
done
# 自检：运行期需要的 preset 与 scaffold 树必须在
for d in settings/core settings/modality settings/property settings/target scaffolds; do
    [ -d "$DST/$d" ] || { echo "ERROR: missing upstream tree: $DST/$d" >&2; exit 1; }
done

echo "Vendored $BINDCRAFT2_REPO @ $BINDCRAFT2_SHA"
echo "  -> $DST"
du -sh "$DST"
