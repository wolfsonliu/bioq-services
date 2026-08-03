#!/usr/bin/env bash
# Create/update a Keycloak user in realm `bioq`. Runs INSIDE the keycloak pod:
#   kubectl -n bioq exec -i deploy/keycloak -- bash -s -- <user> <pass> [admin] < kc-user.sh
# ADMIN (3rd arg, non-empty) adds the user to the `bioq-admins` group → admin role.
set -euo pipefail
USER="$1"; PASS="$2"; ADMIN="${3:-}"
K=/opt/keycloak/bin/kcadm.sh

"$K" config credentials --server http://localhost:8080 --realm master \
  --user admin --password admin >/dev/null

"$K" create users -r bioq -s "username=$USER" -s enabled=true >/dev/null 2>&1 \
  || echo "(user $USER may already exist; updating)"
"$K" set-password -r bioq --username "$USER" --new-password "$PASS"

if [ -n "$ADMIN" ]; then
  uid="$("$K" get users -r bioq -q exact=true -q "username=$USER" \
        --fields id --format csv --noquotes | head -1)"
  gid="$("$K" get groups -r bioq -q search=bioq-admins \
        --fields id --format csv --noquotes | head -1)"
  "$K" update "users/$uid/groups/$gid" -r bioq \
    -s realm=bioq -s "userId=$uid" -s "groupId=$gid" -n
  echo "keycloak user '$USER' ready (admin — in bioq-admins)"
else
  echo "keycloak user '$USER' ready (normal user)"
fi
