import os
import time
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI()


@app.get("/")
def read_root():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    # time.sleep(30)
    # raise SystemError
    uvicorn.run("app:app", host=os.getenv("HOST", "0.0.0.0"), port=os.getenv("PORT", 8000))
