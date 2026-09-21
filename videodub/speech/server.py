from __future__ import annotations

import argparse

from .api import create_speech_app
from .constants import DEFAULT_SPEECH_PORT, SPEECH_HOST


def main() -> None:
    parser = argparse.ArgumentParser(description="SCIP local speech service")
    parser.add_argument("--host", default=SPEECH_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_SPEECH_PORT)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("speech service only accepts loopback hosts")
    import uvicorn

    uvicorn.run(create_speech_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
