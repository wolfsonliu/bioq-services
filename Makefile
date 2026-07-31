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

# --- Local dev (kind + OpenFaaS) ---
# One-command local deploy via deploy/openfaas/local-up.sh. All state (kubeconfig,
# downloaded tools, and the shared volume holding gateway.db + job dirs) lives
# under BIOQ_WORKDIR. These are exported so `make local-*` and the scripts agree.
BIOQ_WORKDIR ?= $(HOME)/.cache/bioq-local
BIOQ_API_KEY ?= bioq-local-secret
BIOQ_GATEWAY_PORT ?= 9000
export BIOQ_WORKDIR BIOQ_API_KEY BIOQ_GATEWAY_PORT
# LOCAL_SERVICES: services to start (space-separated). LOCAL_SVC: which service's
# logs `make local-logs` tails (use "gateway" for the bioq gateway itself).
LOCAL_SERVICES ?= dockq-server
LOCAL_SVC ?= dockq-server
KUBECTL := KUBECONFIG=$(BIOQ_WORKDIR)/kubeconfig PATH="$(BIOQ_WORKDIR)/bin:$$PATH" kubectl

.PHONY: help build push clean list version login-harbor bump sif \
	local-up local-down local-purge local-status local-logs local-test \
	local-info local-forward local-user

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
	@echo "Local dev (kind + OpenFaaS):"
	@echo "  make local-up                Start local deploy (LOCAL_SERVICES=\"$(LOCAL_SERVICES)\")"
	@echo "  make local-up LOCAL_SERVICES=\"dockq-server plip-server\"   Pick services"
	@echo "  make local-status            Show local pods + services"
	@echo "  make local-logs [LOCAL_SVC=..]  Tail logs (LOCAL_SVC=gateway for the gateway)"
	@echo "  make local-test              Run the dockq functional test vs the local deploy"
	@echo "  make local-user ACCOUNT=bob [ADMIN=1]  Create a user + API key (ADMIN=1 for console access)"
	@echo "  make local-info              Print gateway URL / API key / paths"
	@echo "  make local-forward           (Re)establish the gateway port-forward"
	@echo "  make local-down              Tear down (make local-purge also wipes $(BIOQ_WORKDIR))"
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

# --- Local dev (kind + OpenFaaS) ---
# Thin wrappers over deploy/openfaas/local-up.sh / local-down.sh + kubectl queries.
# Config: LOCAL_SERVICES / LOCAL_SVC / BIOQ_* (see the top of this file).

local-up:
	deploy/openfaas/local-up.sh $(LOCAL_SERVICES)

local-down:
	deploy/openfaas/local-down.sh

local-purge:
	deploy/openfaas/local-down.sh --purge

local-status:
	@$(KUBECTL) get pods -A 2>/dev/null | grep -E 'NAMESPACE|bioq|openfaas' \
		|| echo "cluster not up — run: make local-up"
	@echo
	@$(KUBECTL) get svc -A 2>/dev/null | grep -E 'NAMESPACE|bioq|openfaas' || true

local-logs:
	@if [ "$(LOCAL_SVC)" = gateway ]; then \
		$(KUBECTL) -n bioq logs deploy/bioq-gateway -f; \
	else \
		$(KUBECTL) -n openfaas-fn logs -l faas_function=$(LOCAL_SVC) -f; \
	fi

local-forward:
	$(KUBECTL) -n bioq port-forward svc/bioq-gateway $(BIOQ_GATEWAY_PORT):9000 --address 127.0.0.1

local-info:
	@echo "gateway URL : http://127.0.0.1:$(BIOQ_GATEWAY_PORT)   (header X-API-Key: $(BIOQ_API_KEY))"
	@echo "kubeconfig  : $(BIOQ_WORKDIR)/kubeconfig"
	@echo "shared dir  : $(BIOQ_WORKDIR)/shared   (jobs/<acct>-<id>/, users/<acct>/<id>/; pgdata/ or gateway.db)"
	@echo "gateway db  : postgres (svc bioq-postgres, ns bioq) by default; BIOQ_DB_BACKEND=sqlite for the old single-file DB"
	@echo "weights dir : $(BIOQ_WORKDIR)/shared/models -> /data/models (put GPU weights in <dir>/<svc>/; BIOQ_GPU=1 to schedule a GPU)"

local-test:
	cd gateway && GATEWAY_BASE_URL=http://127.0.0.1:$(BIOQ_GATEWAY_PORT) \
		GATEWAY_API_KEY=$(BIOQ_API_KEY) RUN_LOCAL_TESTS=1 \
		uv run --with pytest --with pytest-asyncio python -m pytest tests/test_local_openfaas.py -v

# Create a gateway user + API key in the local deploy. Runs seed_key.py inside
# the gateway pod, which reads GATEWAY_DB_URL — so it works the same whether the
# DB is postgres or sqlite. The secret is printed ONCE (only its hash is stored);
# pass SECRET=.. for a fixed one, else a random secret is generated.
#   make local-user ACCOUNT=alice
#   make local-user ACCOUNT=alice SECRET=s3cret KEY_ID=gk_alice DISPLAY_NAME="Alice"
#   make local-user ACCOUNT=root ADMIN=1        # grant the admin role (console access)
local-user:
	@if [ -z "$(ACCOUNT)" ]; then \
		echo "usage: make local-user ACCOUNT=<name> [SECRET=..] [KEY_ID=..] [DISPLAY_NAME=..] [ADMIN=1]"; \
		exit 2; \
	fi
	@$(KUBECTL) -n bioq exec deploy/bioq-gateway -- /opt/gateway/.venv/bin/python \
		/opt/gateway/scripts/seed_key.py --account-id "$(ACCOUNT)" \
		$(if $(SECRET),--secret "$(SECRET)") \
		$(if $(KEY_ID),--key-id "$(KEY_ID)") \
		$(if $(DISPLAY_NAME),--display-name "$(DISPLAY_NAME)") \
		$(if $(ADMIN),--admin)
