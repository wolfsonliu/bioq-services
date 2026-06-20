"""Run ensemble-server with: python -m server."""

import uvicorn

from .app import app


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")


if __name__ == "__main__":
    main()
