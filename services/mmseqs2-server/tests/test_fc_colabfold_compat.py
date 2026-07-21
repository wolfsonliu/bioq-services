"""FC tests: wire-protocol compatibility with the ColabFold / boltz MSA client.

The whole point of mmseqs2-server is to be a drop-in replacement for
``https://api.colabfold.com`` — clients that already talk ColabFold's MSA
protocol (upstream ``colabfold.colabfold.run_mmseqs2`` and Boltz's fork at
``opensource/boltz/src/boltz/data/msa/mmseqs2.py``) should be able to point
their ``host_url`` at us and just work.

This module speaks that raw HTTP protocol directly (mirroring what
``run_mmseqs2`` sends over the wire) so we do NOT have to import upstream's
huge dep stack (matplotlib / tqdm / jax) into the test environment.

The two hard contracts we verify:

1. **Submit format**: multi-record FASTA (``>101\\nseq1\\n>102\\nseq2\\n``)
   with form fields ``q`` + ``mode`` posted to
   ``/ticket/{msa,pair}``.  This is what run_mmseqs2 emits (colabfold.py
   line 80-84 and boltz mmseqs2.py line 67-71); NOT the ColabFold offline
   ``:``-joined single-record complex format.

2. **Tarball layout**: after
   ``requests.get(f"{host}/result/download/{ID}").content`` the tarball
   MUST contain these exact filenames — that's what the client's
   ``a3m_files = [f"{path}/uniref.a3m"]`` (or ``pair.a3m``) list looks for
   (colabfold.py line 244-248, boltz mmseqs2.py line 258-262):

     * Monomer: ``uniref.a3m``  (+ ``bfd.mgnify30.metaeuk30.smag30.a3m``
                                    when ``use_env=True``)
     * Paired:  ``pair.a3m``

Marked ``@pytest.mark.fc``, skipped by default::

    pytest -m fc services/mmseqs2-server/tests/test_fc_colabfold_compat.py -v
"""

from __future__ import annotations

import io
import os
import tarfile
import time
from pathlib import Path

import httpx
import pytest

from bioq_service.fc_testing import fc_url

SERVICE = "mmseqs2-server"

# Upstream reference MSA server (used by Section 3's side-by-side compat
# test).  Set MMSEQS2_COMPARE_UPSTREAM=1 to opt in — off by default because
# hitting api.colabfold.com is (a) slow, (b) subject to public rate limits,
# and (c) adds a hard external network dependency to the FC suite.
UPSTREAM_HOST = "https://api.colabfold.com"
UPSTREAM_ENABLED = os.environ.get("MMSEQS2_COMPARE_UPSTREAM") == "1"
# api.colabfold.com now soft-warns without a User-Agent and threatens to make
# it a hard error; send a distinct one so their ops can attribute traffic.
UPSTREAM_UA = "bioagent-mmseqs2-server-compat-test/0.0.4 (github.com/bioagent)"

# The ColabFold client applies N=101 as the numeric-header base (see
# colabfold.py line 80).  We mirror that to keep the tarball's internal
# ``>101`` block references consistent with what a real client would send.
COLABFOLD_HEADER_BASE = 101

# Small monomer (~50 aa) — keeps the MSA cost bounded on the GPU subset DB.
SHORT_MONOMER = "QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVN"

# Short heterodimer for the paired test.
PAIRED_CHAIN_B = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDY"
)

# MSA polling parameters — mmseqs on GPU subset DB is 3-10 min for these
# short seqs; give 40 min to absorb cold start.  Longer than the framework
# default because ``/ticket/<id>`` polls burn one FC instance per hit at
# capacity (no session affinity across GET), so we back off aggressively.
POLL_TIMEOUT_S = 2400
POLL_INTERVAL_S = 20

TIMEOUT = httpx.Timeout(connect=30, read=300, write=60, pool=30)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url(SERVICE, start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str):
    with httpx.Client(base_url=base_url, timeout=TIMEOUT) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers (wire protocol replicas of run_mmseqs2's submit/status/download)
# ---------------------------------------------------------------------------


def _colabfold_query(seqs: list[str], n_base: int = COLABFOLD_HEADER_BASE) -> str:
    """Serialize sequences to the multi-record FASTA that run_mmseqs2 emits.

    Mirrors ``colabfold.py:80-84``::

        for seq in seqs:
            query += f">{n}\\n{seq}\\n"
            n += 1
    """
    return "".join(f">{n_base + i}\n{s}\n" for i, s in enumerate(seqs))


def _submit_ticket(
    client: httpx.Client, endpoint: str, q: str, mode: str,
) -> str:
    """POST /ticket/{msa,pair}; return the ticket id.

    Matches run_mmseqs2's ``submit()`` inner function (colabfold.py 80-104):
    form-data POST with ``q`` + ``mode``.
    """
    r = client.post(endpoint, data={"q": q, "mode": mode})
    assert r.status_code == 200, f"submit failed: {r.status_code} {r.text!r}"
    body = r.json()
    assert body.get("status") == "PENDING", (
        f"expected PENDING on submit; got {body!r}"
    )
    tid = body.get("id")
    assert tid, f"submit body missing 'id': {body!r}"
    return tid


def _poll_ticket_to_complete(
    client: httpx.Client, ticket_id: str,
    *,
    session: dict[str, str] | None = None,
) -> None:
    """GET /ticket/<id> in a loop until status is COMPLETE.

    Uses session-affinity header (bioagent-session-id = job_id) so polls
    route back to the same FC instance and don't burn one per request.
    Matches run_mmseqs2's ``status()`` inner function (colabfold.py 112-134).
    """
    hdrs = session or {"bioagent-session-id": ticket_id}
    deadline = time.monotonic() + POLL_TIMEOUT_S
    last_body: dict = {}
    consecutive_429 = 0
    while time.monotonic() < deadline:
        r = client.get(f"/ticket/{ticket_id}", headers=hdrs)
        if r.status_code == 429:
            consecutive_429 += 1
            assert consecutive_429 < 30, (
                f"gave up after {consecutive_429} consecutive 429s on ticket poll"
            )
            time.sleep(POLL_INTERVAL_S)
            continue
        consecutive_429 = 0
        assert r.status_code == 200, f"ticket poll failed: {r.status_code} {r.text!r}"
        last_body = r.json()
        status = last_body.get("status", "")
        if status == "COMPLETE":
            return
        if status == "ERROR":
            raise AssertionError(f"ticket terminal ERROR: {last_body!r}")
        # Still PENDING / RUNNING / UNKNOWN — keep polling.
        time.sleep(POLL_INTERVAL_S)
    raise AssertionError(
        f"ticket {ticket_id!r} did not COMPLETE within {POLL_TIMEOUT_S}s; "
        f"last body: {last_body!r}"
    )


def _download_tarball(
    client: httpx.Client, ticket_id: str,
    *,
    session: dict[str, str] | None = None,
) -> bytes:
    """GET /result/download/<id>; return the tar.gz bytes.

    Matches run_mmseqs2's ``download()`` (colabfold.py 136-153).
    """
    hdrs = session or {"bioagent-session-id": ticket_id}
    r = client.get(f"/result/download/{ticket_id}", headers=hdrs)
    assert r.status_code == 200, f"download failed: {r.status_code} {r.text!r}"
    return r.content


def _tarball_names(tarball_bytes: bytes) -> list[str]:
    """Extract member names from an in-memory tar.gz."""
    with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode="r:gz") as tf:
        return tf.getnames()


# ---------------------------------------------------------------------------
# Section 1: Wire-protocol basics — submit → poll → download for monomer
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def monomer_tarball(client: httpx.Client) -> tuple[str, bytes]:
    """End-to-end run against /ticket/msa in ColabFold wire format.

    Uses ``mode=all`` (UniRef30 only, no env DB) so this passes even when
    ``colabfold_envdb_202108_db`` is not yet staged on NAS.  When envdb is
    available, add a variant that submits ``mode=env`` and checks for
    ``bfd.mgnify30.metaeuk30.smag30.a3m`` in the tarball too.

    Returns (ticket_id, tarball_bytes).
    """
    q = _colabfold_query([SHORT_MONOMER])
    tid = _submit_ticket(client, "/ticket/msa", q, mode="all")
    _poll_ticket_to_complete(client, tid)
    return tid, _download_tarball(client, tid)


@pytest.mark.fc
class TestMonomerWireProtocol:
    """The unpaired submit path (``/ticket/msa`` with multi-record FASTA)."""

    def test_tarball_contains_uniref_a3m(
        self, monomer_tarball: tuple[str, bytes],
    ) -> None:
        """ColabFold + boltz clients look for ``uniref.a3m`` at the top level
        (colabfold.py:247, boltz mmseqs2.py:260).  If we don't ship a file
        with this exact name in the tarball, boltz --use_msa_server pointed
        at us will fail with ``FileNotFoundError: uniref.a3m``."""
        _, tarball = monomer_tarball
        names = _tarball_names(tarball)
        assert "uniref.a3m" in names, (
            f"boltz / ColabFold clients require a top-level 'uniref.a3m' in "
            f"the /result/download tarball; got: {names}"
        )

    def test_uniref_a3m_is_nonempty_text(
        self, monomer_tarball: tuple[str, bytes],
    ) -> None:
        """The file must be a readable a3m (starts with '>' header, has body)."""
        _, tarball = monomer_tarball
        with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as tf:
            member = tf.getmember("uniref.a3m") if "uniref.a3m" in tf.getnames() else None
            if member is None:
                pytest.skip("uniref.a3m not in tarball — covered by test above")
            f = tf.extractfile(member)
            assert f is not None
            body = f.read()
        assert len(body) > 0, "uniref.a3m is empty"
        # A3M lines: `>` header, then AA sequence; ColabFold uses `>{numeric}`
        # for the query seq.  At minimum the query itself should be present.
        text = body.decode("utf-8", errors="replace")
        assert text.startswith(">") or "\x00>" in text, (
            f"uniref.a3m has no FASTA-style '>' header: {text[:200]!r}"
        )


# ---------------------------------------------------------------------------
# Section 2: Paired complex — /ticket/pair with multi-record FASTA
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pair_tarball(client: httpx.Client) -> tuple[str, bytes]:
    """End-to-end run against /ticket/pair.

    Uses ``mode=pairgreedy`` which is UniRef30-only (paired variants that
    also require the env pairing DB are ``pairgreedy-env`` / ``paircomplete-env``,
    intentionally skipped here to keep dependencies minimal).

    Wire format: multi-record FASTA, one record per chain.  This is what
    run_mmseqs2 sends (colabfold.py:80-84 iterates ``seqs`` regardless of
    pairing mode); our server-side must collapse it to a single ``:``-joined
    complex record before handing off to the orchestrator (fixed in v0.0.3;
    see ``app.py:_serialize_fasta`` docstring).
    """
    q = _colabfold_query([SHORT_MONOMER, PAIRED_CHAIN_B])
    tid = _submit_ticket(client, "/ticket/pair", q, mode="pairgreedy")
    _poll_ticket_to_complete(client, tid)
    return tid, _download_tarball(client, tid)


@pytest.mark.fc
class TestPairWireProtocol:
    """The paired submit path (``/ticket/pair`` with multi-record FASTA)."""

    def test_tarball_contains_pair_a3m(
        self, pair_tarball: tuple[str, bytes],
    ) -> None:
        """ColabFold + boltz clients look for ``pair.a3m`` at the top level
        (colabfold.py:245, boltz mmseqs2.py:258).  Without this exact
        filename, ``boltz --use_msa_server`` for multimers fails."""
        _, tarball = pair_tarball
        names = _tarball_names(tarball)
        assert "pair.a3m" in names, (
            f"boltz / ColabFold clients require a top-level 'pair.a3m' in "
            f"the /result/download tarball for paired queries; got: {names}"
        )

    def test_pair_a3m_is_nonempty(
        self, pair_tarball: tuple[str, bytes],
    ) -> None:
        _, tarball = pair_tarball
        with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as tf:
            if "pair.a3m" not in tf.getnames():
                pytest.skip("pair.a3m not in tarball — covered by test above")
            member = tf.getmember("pair.a3m")
            f = tf.extractfile(member)
            assert f is not None
            body = f.read()
        assert len(body) > 0, "pair.a3m is empty"


# ---------------------------------------------------------------------------
# Section 3: Side-by-side comparison with upstream api.colabfold.com
#
# The point: we advertise mmseqs2-server as a drop-in replacement.  This
# section actually proves it by submitting the SAME query to both endpoints
# and verifying tarball layout + query preservation align.
#
# Opt-in (MMSEQS2_COMPARE_UPSTREAM=1) — api.colabfold.com is rate-limited,
# slow (10-30 min per submission at busy hours), and adds an external network
# dependency to CI that we don't want by default.
# ---------------------------------------------------------------------------


def _end_to_end_msa(
    host_url: str, endpoint: str, q: str, mode: str,
    *,
    headers: dict[str, str] | None = None,
    session_header_name: str | None = None,
    timeout_s: int = POLL_TIMEOUT_S,
) -> bytes:
    """Full ColabFold protocol run against ``host_url`` — returns the tar.gz.

    Works for both our FC deployment and the upstream reference server.
    ``session_header_name`` (only used on our server) is the affinity header
    the response's ``bioagent-session-id`` maps to; upstream ignores it.
    """
    base_headers = dict(headers or {})
    with httpx.Client(base_url=host_url, timeout=TIMEOUT) as c:
        r = c.post(endpoint, data={"q": q, "mode": mode}, headers=base_headers)
        assert r.status_code == 200, (
            f"submit to {host_url}{endpoint}: {r.status_code} {r.text!r}"
        )
        body = r.json()
        assert body.get("status") == "PENDING", (
            f"unexpected submit body from {host_url}: {body!r}"
        )
        ticket_id = body["id"]

        # Session affinity so our-side polls hit the same instance.
        poll_headers = dict(base_headers)
        if session_header_name:
            poll_headers[session_header_name] = ticket_id

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            g = c.get(f"/ticket/{ticket_id}", headers=poll_headers)
            if g.status_code == 429:
                time.sleep(POLL_INTERVAL_S)
                continue
            assert g.status_code == 200, g.text
            status = g.json().get("status", "")
            if status == "COMPLETE":
                break
            if status == "ERROR":
                raise AssertionError(f"{host_url} returned ERROR: {g.json()!r}")
            time.sleep(POLL_INTERVAL_S)
        else:
            raise AssertionError(
                f"{host_url}/ticket/{ticket_id} did not COMPLETE within {timeout_s}s"
            )

        d = c.get(f"/result/download/{ticket_id}", headers=poll_headers)
        assert d.status_code == 200, f"download from {host_url}: {d.status_code}"
        return d.content


def _extract_a3m(tarball_bytes: bytes, member_name: str) -> bytes:
    with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode="r:gz") as tf:
        if member_name not in tf.getnames():
            raise AssertionError(
                f"tarball missing '{member_name}'; contents: {tf.getnames()}"
            )
        f = tf.extractfile(member_name)
        assert f is not None
        return f.read()


def _first_query_seq(a3m_bytes: bytes) -> str:
    """Extract the query sequence from an a3m: line 2 (after the ``>`` header).

    A3M format: line 1 = header (``>...``), line 2 = the query sequence, then
    aligned homologues follow.  If multiple queries are concatenated with
    ``\\x00`` separators (multi-query a3m), we still get the FIRST query's
    sequence — that's what both endpoints receive.
    """
    text = a3m_bytes.decode("utf-8", errors="replace").split("\x00", 1)[0]
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert lines and lines[0].startswith(">"), (
        f"a3m does not start with '>': {lines[:3]}"
    )
    assert len(lines) >= 2, f"a3m has no sequence line: {lines}"
    return lines[1].strip()


upstream_only = pytest.mark.skipif(
    not UPSTREAM_ENABLED,
    reason=(
        "upstream compare disabled — set MMSEQS2_COMPARE_UPSTREAM=1 to run "
        "these against api.colabfold.com (slow, rate-limited)"
    ),
)


@upstream_only
@pytest.mark.fc
class TestUpstreamCompareMsa:
    """Same monomer query submitted to both api.colabfold.com and our FC."""

    def _mode(self) -> str:
        # ``mode=all`` = UniRef30 only, filter=1, unpaired.  Chosen because it
        # avoids the envdb (which we haven't staged yet) — upstream also
        # accepts ``all`` and returns just ``uniref.a3m`` in that case.
        return "all"

    def test_both_return_uniref_a3m(self, client: httpx.Client, base_url: str) -> None:
        q = _colabfold_query([SHORT_MONOMER])

        ours = _end_to_end_msa(
            base_url, "/ticket/msa", q, mode=self._mode(),
            session_header_name="bioagent-session-id",
        )
        upstream = _end_to_end_msa(
            UPSTREAM_HOST, "/ticket/msa", q, mode=self._mode(),
            headers={"User-Agent": UPSTREAM_UA},
        )

        our_names = set(tarfile.open(fileobj=io.BytesIO(ours), mode="r:gz").getnames())
        up_names = set(tarfile.open(fileobj=io.BytesIO(upstream), mode="r:gz").getnames())
        assert "uniref.a3m" in our_names, f"our tarball missing uniref.a3m: {our_names}"
        assert "uniref.a3m" in up_names, f"upstream missing uniref.a3m: {up_names}"

    def test_query_seq_matches_upstream(
        self, client: httpx.Client, base_url: str,
    ) -> None:
        """Both tarballs' ``uniref.a3m`` must start with the exact query seq
        we submitted — a3m format contract.  DB contents / MSA depth can
        differ (upstream uses more recent + broader DBs), so we don't compare
        the homologue rows; only the query."""
        q = _colabfold_query([SHORT_MONOMER])

        ours = _end_to_end_msa(
            base_url, "/ticket/msa", q, mode=self._mode(),
            session_header_name="bioagent-session-id",
        )
        upstream = _end_to_end_msa(
            UPSTREAM_HOST, "/ticket/msa", q, mode=self._mode(),
            headers={"User-Agent": UPSTREAM_UA},
        )

        our_q = _first_query_seq(_extract_a3m(ours, "uniref.a3m"))
        up_q = _first_query_seq(_extract_a3m(upstream, "uniref.a3m"))
        # Query sequences must be preserved byte-for-byte (upstream and our
        # servers both echo the submitted seq as row 1 of the a3m).
        assert our_q == SHORT_MONOMER, (
            f"our uniref.a3m query row is {our_q!r}, expected {SHORT_MONOMER!r}"
        )
        assert up_q == SHORT_MONOMER, (
            f"upstream uniref.a3m query row is {up_q!r}, expected {SHORT_MONOMER!r}"
        )

    def test_hit_counts_similar_order_of_magnitude(
        self, client: httpx.Client, base_url: str,
    ) -> None:
        """Sanity: both MSAs should find hits (nonzero + comparable log-scale).

        Not a strict equality — DB versions differ, filters may prune slightly
        differently.  We assert both hit counts are ``>= 5`` (i.e. the query
        actually matched something in UniRef30) and neither is >100x the
        other (catch gross regressions, e.g. our search failing silently and
        emitting only the query row)."""
        q = _colabfold_query([SHORT_MONOMER])
        ours = _end_to_end_msa(
            base_url, "/ticket/msa", q, mode=self._mode(),
            session_header_name="bioagent-session-id",
        )
        upstream = _end_to_end_msa(
            UPSTREAM_HOST, "/ticket/msa", q, mode=self._mode(),
            headers={"User-Agent": UPSTREAM_UA},
        )

        def _count_hits(a3m_bytes: bytes) -> int:
            text = a3m_bytes.decode("utf-8", errors="replace").split("\x00", 1)[0]
            return sum(1 for ln in text.splitlines() if ln.startswith(">"))

        n_ours = _count_hits(_extract_a3m(ours, "uniref.a3m"))
        n_up = _count_hits(_extract_a3m(upstream, "uniref.a3m"))
        assert n_ours >= 5, f"our uniref.a3m only has {n_ours} entries (expected >= 5)"
        assert n_up >= 5, f"upstream uniref.a3m only has {n_up} entries (expected >= 5)"
        # log-scale sanity
        ratio = max(n_ours, n_up) / min(n_ours, n_up)
        assert ratio < 100, (
            f"hit count ratio our={n_ours} up={n_up} differs by {ratio:.1f}x — "
            f"likely a regression in one side's pipeline"
        )


@upstream_only
@pytest.mark.fc
class TestUpstreamComparePair:
    """Same paired complex submitted to both api.colabfold.com and our FC."""

    def _mode(self) -> str:
        return "pairgreedy"  # UniRef30-only paired (no env pairing DB)

    def test_both_return_pair_a3m(self, client: httpx.Client, base_url: str) -> None:
        q = _colabfold_query([SHORT_MONOMER, PAIRED_CHAIN_B])

        ours = _end_to_end_msa(
            base_url, "/ticket/pair", q, mode=self._mode(),
            session_header_name="bioagent-session-id",
        )
        upstream = _end_to_end_msa(
            UPSTREAM_HOST, "/ticket/pair", q, mode=self._mode(),
            headers={"User-Agent": UPSTREAM_UA},
        )

        our_names = set(tarfile.open(fileobj=io.BytesIO(ours), mode="r:gz").getnames())
        up_names = set(tarfile.open(fileobj=io.BytesIO(upstream), mode="r:gz").getnames())
        assert "pair.a3m" in our_names, f"our tarball missing pair.a3m: {our_names}"
        assert "pair.a3m" in up_names, f"upstream missing pair.a3m: {up_names}"
