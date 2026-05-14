class AppError(Exception):
    def __init__(
            self,
            code: str,
            message: str,
            retryable: bool = False,
    ) -> None:
        # super().__init__(message) 里带 message，是为了把错误信息传给 Python 原生的 Exception 父类。
        # 也就是说，AppError 虽然是我们自定义的异常，但它本质上还是一个异常对象。父类 Exception 也需要保存一个“异常说明”。
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable

    def to_dict(self) -> dict:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
            },
        }