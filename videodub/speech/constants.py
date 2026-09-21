SPEECH_HOST = "127.0.0.1"
DEFAULT_SPEECH_PORT = 9955
SPEECH_WORKER_BASE_PORT = 12000


def speech_base_url(port: int = DEFAULT_SPEECH_PORT) -> str:
    return f"http://{SPEECH_HOST}:{port}"
