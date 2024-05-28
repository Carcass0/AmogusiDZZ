from random import getrandbits

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")
async def make_model_call(image: UploadFile):
    contents = image.file.read()
    return JSONResponse(content={"answer": bool(getrandbits(1))}, status_code=200)
