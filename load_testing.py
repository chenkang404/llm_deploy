import asyncio
import aiohttp
import time

import argparse


def parser_():
    parser = argparse.ArgumentParser(description="load test")
    parser.add_argument("--label",default='pipe',type=str,help="控制异步,para,pipe")
    parser.add_argument("--load",default=5,type=int, help="并发数")
    
    args = parser.parse_args()
    return args

async def post(session, url, data):
    async with session.post(url, json=data) as response:
        return await response.text()

async def bound_post(sem, session, url, data):
    async with sem:
        await post(session, url, data)

async def run(load, url, data):
    tasks = []
    sem = asyncio.Semaphore(load)  # 控制并发量
    async with aiohttp.ClientSession() as session:
        for _ in range(load):
            task = asyncio.create_task(bound_post(sem, session, url, data))
            tasks.append(task)
        responses = await asyncio.gather(*tasks)
    return responses

if __name__ == "__main__":
    args = parser_()

    load = args.load  # 并发请求数
    url_pipe = "http://0.0.0.0:8001/create"  # pipeline被测试的URL
    url_para = "http://0.0.0.0:8000/create"  # parallel被测试的URL

    data = {"query": "韩国的首都是哪里,请用中文回答"}  # POST请求的数据



    # label = "pipe"
    # label = "para"
    label = args.label


    if label == "pipe":
        url = url_pipe
        print(f"pipeline通道式部署压测,load:{load}")
    elif label == "para":
        url = url_para
        print(f"parallel并发式部署压测,load:{load}")
    

    start_time = time.time()
    asyncio.run(run(load, url, data))
    duration = time.time() - start_time

    print(f"Completed {load} POST requests in {duration} seconds")

