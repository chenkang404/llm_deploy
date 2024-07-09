import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig
)

import re, uvicorn, json
# import tornado.ioloop #核心IO循环模块
# import tornado.httpserver #异步非阻塞HTTP服务器模块
# import tornado.web #Web框架模块
# import tornado.options #解析终端参数模块

import asyncio
from pydantic import BaseModel




class Inputs(BaseModel):
    query:str

class Outputs(BaseModel):
    answer:str

model_id = "/workspace/HOSTDIR/Model_base/llama3-8b-instruct"


print("是否有加速器:",torch.cuda.is_available())
print(f"gpu数量：{torch.cuda.device_count()}")



from fastapi import FastAPI, HTTPException, Query
import re, uvicorn, json
from typing import List
import logging
import pytz


local_tz = pytz.timezone('Asia/Shanghai')

LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

logger.info(f"gpu count :{torch.cuda.device_count()}")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

app = FastAPI()


terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]




@app.post("/create")
async def create_item(request: Inputs):
    # logger.info(f"request:{request}")
    # logger.info(f"request type:{type(request)}")

    query = request.query
    # logger.info(f"query type:{type(query)}")
    logger.info(query)
    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": query},
    ]
    logger.info(f"messages:{messages}")

    input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt").to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    logger.info(f'finish')
    response = outputs[0][input_ids.shape[-1]:]
    answer = tokenizer.decode(response,skip_special_tokens=True)
    logger.info(f"answer:{answer}")
    return {"result":answer}




if __name__ == "__main__":
    # uvicorn.run()
    uvicorn.run(app, host='0.0.0.0', port=8001, workers=1)    