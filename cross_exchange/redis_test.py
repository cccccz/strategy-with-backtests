import redis.asyncio as aioredis
import json

redis_client = aioredis.from_url("redis://localhost", decode_responses=True)

snapshot = {"foo": "bar",
            "number": 2}

redis_client.set("state",json.dumps(snapshot))

from fastapi import FastAPI

app = FastAPI()

@app.get("/state")
def get_state():
    data = redis_client.get("trading_state")
    if data:
        return json.loads(data)
    return {}