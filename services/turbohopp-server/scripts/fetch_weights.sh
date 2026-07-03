#!/usr/bin/env bash
# TurboHopp consistency-model checkpoint staging (placeholder).
#
# Unlike diffusion-hopping-server (whose upstream repo ships 4 .ckpts in
# checkpoints/), TurboHopp's upstream repository does NOT publish a public
# checkpoint.  The consistency-model .ckpt must be obtained by:
#
#   1. Training via upstream's train_consistency.py using a DiffHopp teacher.
#      See `services/turbohopp-server/upstream/train_consistency.py` and
#      `configs/config_consistency.yaml`.  Reference training paths in
#      upstream configs point at /data/aigen/consistency/training/checkpoints/
#      — the paper's authors' internal layout.
#
#   2. Requesting from the authors (Yoo et al. 2024, NeurIPS 2024).
#
# Once you have a checkpoint, stage it to NAS:
#
#   rsync -av path/to/turbohopp_consistency.ckpt \
#     <nas>:/data/models/turbohopp/checkpoints/v1/
#
# After that FC + /healthz/detail reports `weights_loaded: true, files_found: 1`.

set -euo pipefail

cat >&2 <<'EOF'
ERROR: TurboHopp upstream does not publish a public consistency-model .ckpt.

Once you have obtained one (train via train_consistency.py or from paper
authors), stage it to NAS:

  rsync -av path/to/turbohopp_consistency.ckpt \
    <nas>:/data/models/turbohopp/checkpoints/v1/

/healthz/detail will report weights_loaded=false until then.
EOF
exit 1
