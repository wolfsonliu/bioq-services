#!/usr/bin/env bash
# `python -m bindcraft.cli` 的离线替身：不碰 GPU，只写各子命令本该写出的产物。
# 由 tests/conftest.py 复制到 tmp_path 并 chmod +x，通过 BINDCRAFT2_PYTHON 指进来。
set -euo pipefail

if [ "${1:-}" = "-c" ]; then
  # /healthz/detail 的 GPU 探针：第一行 backend，第二行 devices。
  echo gpu
  echo cuda:0
  exit 0
fi

sub="${1:-}"
shift || true

if [ "$sub" = "design" ]; then
  json="$1"
  out="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_folder"])' "$json")"
  mkdir -p "$out/3_Ranked" "$out/1_Trajectories"
  printf 'design,i_pDAE\ndemo_1,0.72\n' > "$out/3_Ranked/!_Ranked.csv"
  printf 'design,terminated\ndemo_1,\n' > "$out/1_Trajectories/!_Trajectories.csv"
  printf 'scope,metric,samples\ncampaign,final,1\n' > "$out/summary.csv"
  printf '{"source_revision":"fake"}\n' > "$out/campaign_metadata.json"
  exit 0
fi

if [ "$sub" = "rank" ] || [ "$sub" = "filter" ]; then
  out=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --output) out="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  [ -n "$out" ] || { echo "fake_bindcraft: --output is required" >&2; exit 2; }
  mkdir -p "$(dirname "$out")"
  printf 'design,rank\n' > "$out"
  exit 0
fi

echo "fake_bindcraft: unknown subcommand '${sub}'" >&2
exit 2
