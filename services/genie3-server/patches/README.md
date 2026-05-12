# genie3 patches

Patches applied to `opensource/genie3/` before building the server image.
Applied automatically by the Dockerfile via `git apply` during the build.

| Patch | Upstream issue | Why we need it |
|---|---|---|
| `0001-fix-binder-framework-prot-rep-mode.patch` | `create_np_features_from_target_config` calls `create_np_features_from_motif_config` without the required `prot_rep_mode` argument when `binder_framework` is set in the problem JSON | Enables the `binder_framework` code path (framework-constrained binder design, e.g. nanobody CDR scaffolding). Without it the call crashes with `TypeError`. |

These patches should be reported upstream once we've validated they work end-to-end.
