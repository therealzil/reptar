from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn, os, json, uuid, time
app = FastAPI()

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    content = {"compliance": True} # Simplified for brevity
    return {"id": "mock", "choices": [{"message": {"role": "assistant", "content": json.dumps(content)}}]}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
