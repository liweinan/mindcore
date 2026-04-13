class InferenceUnavailableError(Exception):
    """Ollama（或远程推理）不可用且未允许模板回退。"""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
