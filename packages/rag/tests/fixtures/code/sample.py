import os


class Config:
    def path(self) -> str:
        return os.getcwd()


def load_config() -> Config:
    return Config()
