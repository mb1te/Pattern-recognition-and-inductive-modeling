import uvicorn
from fastapi import FastAPI

from src.settings import settings

app = FastAPI(
    debug=settings.APP_DEBUG,
)

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.APP_DEBUG,
    )
