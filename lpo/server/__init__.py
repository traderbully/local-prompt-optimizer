"""FastAPI backend + WebSocket server for the LPO web UI.

Exposed surface:
  * :mod:`lpo.server.api` — FastAPI app factory :func:`create_app`.
  * :mod:`lpo.server.runs` — :class:`RunManager` singleton that owns active
    optimization runs and brokers WebSocket events + user signals.
  * :mod:`lpo.server.tasks` — pure filesystem readers for task bundles and
    persisted iteration data.

The server is local-only (binds to 127.0.0.1 by default — see
`LPO_SDP.md` §6.3).
"""

from lpo.server.api import create_app  # noqa: F401
