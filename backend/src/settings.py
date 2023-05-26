from pydantic import BaseSettings


class Settings(BaseSettings):
    APP_HOST: str
    APP_PORT: int
    APP_DEBUG: bool
