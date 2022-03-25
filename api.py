# import
import os
import sys
import time
import uuid
import logging
import uvicorn
from datetime import date
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, Request

# get the project root directory
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__))))
from keyphrasetransformer import KeyPhraseTransformer


# setup logger
logging.basicConfig(
    filename=f"logs/{date.today()}.log",
    filemode="a",
    format="{asctime} : {levelname:7} : {message}",
    style="{",
    level=logging.INFO,
)
logger = logging.getLogger("uvicorn.error")


# fastapi app
app = FastAPI()


# request body
class RequestBodyKPE(BaseModel):
    text: str
    text_block_size: Optional[int] = 64


# prepare KPT model
kp = KeyPhraseTransformer()


# endpoints
# root
@app.get("/")
def read_root():
    return {"Message": "Welcome to KeyPhraseTransformer module"}


# getKPE
@app.post("/getKPE")
def get_keyphrase(request: Request, requestbodykpe: RequestBodyKPE):
    start_time = time.time()
    uid = str(uuid.uuid4())
    outputs = kp.get_key_phrases(
        text=requestbodykpe.text, text_block_size=requestbodykpe.text_block_size
    )
    process_time = round((time.time() - start_time), 4)
    result = {"KeyPhraseTransformer": outputs}
    logger.info(
        f"id={uid} | host={request.client} | process_time={process_time} | response={result} | text={requestbodykpe.text}"
    )

    return result


# main
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0")
