from fastapi import FastAPI
import asyncio

application = FastAPI()
@application.get("/greeting")
async def greeting():
    await asyncio.sleep(20)
    return {"greeting": "Hello, World!"}

@application.get("/greeting2")
async def greeting2():
    await asyncio.sleep(5)
    return {"greeting2": "Hello, World2!"}

@application.post("/submit")
async def submit(data: dict):
    return {"message": f"Form submitted successfully for {data['name']}"}