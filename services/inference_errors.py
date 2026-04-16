class InferenceUnavailableError(Exception):
    """RAG 或 Ollama 对话链路失败。"""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
