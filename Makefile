PLATFORM := linux/amd64

# Default registry. Override with REGISTRY=... if pushing somewhere else.
# Login once with: docker login harbor.ruosheng.bio
REGISTRY ?= harbor.ruosheng.bio/aliyun_fc

# Auto-discover buildable images across the tiers: compute workers under
# services/, the control-plane gateway/, and edge/ components (jwt, MCP adapter).
# framework/ has no Dockerfile so it is never discovered (it is a library, not a
# deployable image). The image name is the LAST path segment (workers keep their
# -server suffix, so image names & live FC references are unchanged).
SERVICE_DOCKERFILES := $(wildcard services/*/Dockerfile) $(wildcard gateway/Dockerfile) $(wildcard edge/*/Dockerfile)
SERVICES := $(notdir $(patsubst %/Dockerfile,%,$(SERVICE_DOCKERFILES)))

# Resolve a service name to its source directory across the tiers. Candidates,
# in order: services/<name>/ (workers), gateway/ (top-level), edge/<name>/.
svc_dir = $(patsubst %/Dockerfile,%,$(firstword $(wildcard services/$(1)/Dockerfile $(1)/Dockerfile edge/$(1)/Dockerfile)))

# Per-service versioning. Each service has its own release cadence; image tags
# are NEVER coordinated globally. Priority for which tag to use:
#
#   1. CLI override (`make push-<svc> TAG=v0.0.5`) — wins over everything
#   2. <svc-dir>/VERSION file (one line, e.g. "v0.0.5") — the normal case
#   3. Git describe fallback — for unversioned local builds only
#
# Note: TAG is left unset by default (no `?=`) so the priority chain above
# actually evaluates instead of being short-circuited by a global default.
GIT_TAG  := $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
service_version = $(shell cat $(call svc_dir,$(1))/VERSION 2>/dev/null || echo $(GIT_TAG))

# SIF output directory for Apptainer images.
SIF_DIR ?= sif

.PHONY: help build push clean list version login-harbor bump sif

# Keep intermediate pattern targets around (no auto-rm after the recipe runs).
.PRECIOUS: build-% tag-%

help:
	@echo "Versioning model: each service has its own image tag (services evolve"
	@echo "independently). The tag comes from <svc-dir>/VERSION; bump that"
	@echo "file when you cut a new release. `TAG=...` on the CLI overrides for"
	@echo "one invocation."
	@echo ""
	@echo "Build:"
	@echo "  make build-<service>         Build one service image (uses its VERSION)"
	@echo "  make build-<svc> TAG=v0.0.5  Override tag for this build"
	@echo "  make build                   Build all (each at its own VERSION)"
	@echo ""
	@echo "Push to harbor (REGISTRY=$(REGISTRY)):"
	@echo "  make push-<service>          Build + tag + push one service"
	@echo "  make push-<svc> TAG=v0.0.5   Push a specific version"
	@echo "  make tag-<service>           Tag locally, no push (inspect first)"
	@echo "  make push                    Push all (rarely needed)"
	@echo ""
	@echo "Apptainer SIF (SIF_DIR=$(SIF_DIR)):"
	@echo "  make sif-<service>           Docker build → Apptainer SIF"
	@echo "  make sif                     Build all SIF images"
	@echo "  make clean-sif-<service>     Remove one SIF"
	@echo "  make clean-sif               Remove all SIF images"
	@echo ""
	@echo "Versioning:"
	@echo "  make bump-<service>          Bump patch version (v0.0.5 → v0.0.6)"
	@echo "  make version                 Show every service's current tag"
	@echo ""
	@echo "Misc:"
	@echo "  make clean                   Remove all local service images"
	@echo "  make clean-<service>         Remove one"
	@echo "  make list                    List discovered services"
	@echo "  make login-harbor            docker login harbor.ruosheng.bio"
	@echo ""
	@echo "Current state:"
	@$(foreach svc,$(SERVICES),echo "  $(svc): $(call service_version,$(svc))";)

version:
	@$(foreach svc,$(SERVICES),echo "$(svc): $(call service_version,$(svc))";)

list:
	@echo $(SERVICES)

login-harbor:
	docker login harbor.ruosheng.bio

# --- Bump patch version ---

bump-%:
	@f=$(call svc_dir,$*)/VERSION; \
	if [ ! -f "$$f" ]; then echo "error: $$f not found"; exit 1; fi; \
	old=$$(cat "$$f"); \
	prefix=$${old%.*}; \
	patch=$${old##*.}; \
	new="$$prefix.$$((patch + 1))"; \
	echo "$$new" > "$$f"; \
	echo "→ $*: $$old → $$new"

# --- Build ---

build: $(addprefix build-,$(SERVICES))

build-%:
	$(eval V := $(if $(TAG),$(TAG),$(call service_version,$*)))
	@echo "→ building $*:$(V)"
	docker build --platform $(PLATFORM) \
		-t $*:$(V) -t $*:latest \
		-f $(call svc_dir,$*)/Dockerfile .

# --- Tag (no push) ---
# Useful when you want to verify the registry path before pushing, or push
# manually via `docker push` after inspection.
tag-%: build-%
	$(eval V := $(if $(TAG),$(TAG),$(call service_version,$*)))
	docker tag $*:$(V) $(REGISTRY)/$*:$(V)
	docker tag $*:$(V) $(REGISTRY)/$*:latest
	@echo "→ tagged $(REGISTRY)/$*:$(V) and $(REGISTRY)/$*:latest"

# --- Push to harbor ---
# Convenience flow matching the v0.2 service deploy:
#   docker build → docker tag → docker push (both :TAG and :latest)
# Inlines the tag step so we don't depend on `tag-%` (which would otherwise
# get auto-rm'd as an intermediate pattern target).
push: $(addprefix push-,$(SERVICES))

push-%: build-%
	$(eval V := $(if $(TAG),$(TAG),$(call service_version,$*)))
	docker tag $*:$(V) $(REGISTRY)/$*:$(V)
	docker tag $*:$(V) $(REGISTRY)/$*:latest
	docker push $(REGISTRY)/$*:$(V)
	docker push $(REGISTRY)/$*:latest
	@echo "→ pushed $(REGISTRY)/$*:$(V)"
	@echo "  next: in Alibaba FC console, update the function image to $(REGISTRY)/$*:$(V)"

# --- Apptainer SIF ---
# Convert Docker images to Apptainer SIF for Slurm/HPC clusters.
# Requires `apptainer` on PATH. SIF files land in $(SIF_DIR)/.
#
# Usage:
#   make sif-dockq-server              # single service
#   make sif                           # all services
#   make sif-dockq-server SIF_DIR=/shared/images  # custom output dir

sif: $(addprefix sif-,$(SERVICES))

sif-%: build-%
	$(eval V := $(if $(TAG),$(TAG),$(call service_version,$*)))
	@command -v apptainer >/dev/null 2>&1 || { echo "error: apptainer not found on PATH"; exit 1; }
	@mkdir -p $(SIF_DIR)
	apptainer build $(SIF_DIR)/$*.sif docker-daemon://$*:$(V)
	@echo "→ $(SIF_DIR)/$*.sif (from $*:$(V))"

# --- Clean ---

clean: $(addprefix clean-,$(SERVICES))

clean-%:
	$(eval V := $(if $(TAG),$(TAG),$(call service_version,$*)))
	docker rmi $*:$(V) $*:latest 2>/dev/null || true
	docker rmi $(REGISTRY)/$*:$(V) $(REGISTRY)/$*:latest 2>/dev/null || true

clean-sif: $(addprefix clean-sif-,$(SERVICES))

clean-sif-%:
	rm -f $(SIF_DIR)/$*.sif
