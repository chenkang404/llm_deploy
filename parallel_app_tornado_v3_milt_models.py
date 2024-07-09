import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,5"
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import asyncio
from fastapi import FastAPI
import logging
import pytz
import itertools
import re, uvicorn, json
from pydantic import BaseModel

class Inputs(BaseModel):
    query: str

class Outputs(BaseModel):
    answer: str

model_id = "/workspace/HOSTDIR/Model_base/llama3-8b-instruct"

print("是否有加速器:", torch.cuda.is_available())
print(f"gpu数量：{torch.cuda.device_count()}")

local_tz = pytz.timezone('Asia/Shanghai')

LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

logger.info(f"gpu count :{torch.cuda.device_count()}")

tokenizer = AutoTokenizer.from_pretrained(model_id)

# 加载模型到两个 GPU 上
device_ids = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
models = [AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
).to(f'cuda:{i}') for i in range(torch.cuda.device_count())]

app = FastAPI()

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

num_gpus = torch.cuda.device_count()
gpu_pool = itertools.cycle(range(num_gpus))

def allocate_gpu():
    return next(gpu_pool)

def model_generate(query,gpu_id):    
    model = models[gpu_id]
    device = device_ids[gpu_id]
    logger.info(f"model device:{model.device}")
    logger.info(f"device id :{device}")
    torch.cuda.set_device(device)

    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": query},
    ]
    logger.info(f"messages:{messages}")

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    # 添加 attention_mask
    # attention_mask = input_ids['input_ids'].ne(tokenizer.pad_token_id).to(device)


    outputs = model.generate(
        input_ids,
        # attention_mask=attention_mask,
        max_new_tokens=256,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    logger.info(f'finish')
    response = outputs[0][input_ids.shape[-1]:]
    answer = tokenizer.decode(response, skip_special_tokens=True)
    logger.info(f"answer:{answer}")
    return answer

async def generate_(query: str):
    loop = asyncio.get_event_loop()
    gpu_id = allocate_gpu()
    logger.info(f"gpu id:{gpu_id}")
    result = await loop.run_in_executor(None, model_generate, query,gpu_id)
    return result

@app.post("/create")
async def create_item(request: Inputs):
    query = request.query
    logger.info(query)
    answer = await generate_(query)
    return {"result": answer}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
