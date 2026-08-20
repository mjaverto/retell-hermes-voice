"""Entrypoint: ``python -m retell_hermes_voice``."""

from __future__ import annotations

import uvicorn

from .config import get_settings
from .logredact import configure_logging
from .server import create_app


def main() -> None:
    settings = get_settings()
    configure_logging(settings)
    uvicorn.run(
        create_app(settings),
        host=settings.listen_host,
        port=settings.listen_port,
        log_config=None,
        # First line of defense: oversized frames are rejected at the transport
        # before buffering; parse_inbound re-checks the same limit in-app.
        ws_max_size=settings.max_ws_message_bytes,
    )


if __name__ == "__main__":
    main()
