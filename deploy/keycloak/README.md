# keycloak (OIDC auth spike)

Local [Keycloak](https://www.keycloak.org/) as an OIDC IdP, used to **verify that
the gateway's existing JWT/JWKS auth layer accepts tokens from a real third-party
IdP** — with no gateway code changes, only config. Groundwork for OIDC login
(browser console + `bioq` CLI device flow). See the design decision in the
`bioagent` monorepo: `engineering/decisions/2026-08-03-oidc-authentication.md`.

## What it sets up

`realm-export.json` imports realm **`bioq`** with:
- public client **`bioq-gateway`** — direct access grants (password grant, for the
  spike) + an **audience mapper** emitting `aud=gateway-server` (matches the
  gateway's `jwt_audience` default) + a **group-membership mapper** (`groups` claim)
- user **`alice` / `alice`** in group **`bioq-admins`**

## Run

```bash
./verify.sh          # up Keycloak + prove verification end-to-end
./verify.sh --down   # tear Keycloak down
```

`verify.sh`:
1. brings up Keycloak (`docker compose up -d`, realm auto-imported)
2. gets an access token via the password grant
3. **Step A** — calls the gateway's own `verify_jwt()` against Keycloak's JWKS
4. **Step B** — runs the gateway locally (`BYPASS_VPC=false`, `JWT_JWKS_URL`=Keycloak)
   and hits auth-gated `/v1/services`: no token → 401, valid Bearer → 200

## Point the gateway at Keycloak (manual)

```bash
export GATEWAY_AUTH__JWT_JWKS_URL=http://localhost:8080/realms/bioq/protocol/openid-connect/certs
export GATEWAY_AUTH__JWT_AUDIENCE=gateway-server
# then a request with `Authorization: Bearer <keycloak access token>` authenticates
# as account_id = token `sub`.
```

## Notes

- Dev mode (`start-dev`) keeps everything in-memory — nothing persists across
  `docker compose down`. **Not for production.**
- The `:z` suffix on the realm-file volume mount is required on SELinux hosts
  (Fedora/RHEL), else Keycloak can't read the mounted file and imports nothing.
- This is a **spike**, not a deployment. Production OIDC = a managed/HA IdP (which
  can front LDAP/AD), the gateway configured with its JWKS URL, plus `sub`→account
  and `groups`→role mapping. See the decision doc.
