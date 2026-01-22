#!/usr/bin/env python
# coding: utf-8
import os, sys
import json
import asyncio

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

CURRENT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
EXAMPLES_DIR = os.path.join(REPO_ROOT, "src")

for p in [REPO_ROOT, EXAMPLES_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.super_agent.api.settings import Settings
from src.super_agent.api.schemas import ChatRequest, ResetRequest
from src.super_agent.agent.api_runtime_cdp import init_agents_once, run_turn_stream, reset_session, shutdown_runner
# from examples.super_agent.agent.api_runtime import init_agents_once, run_turn_stream, reset_session, shutdown_runner





app = FastAPI(title="Super Agent API", version="2.2")
settings = Settings.build()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def _startup():
    await init_agents_once()

@app.on_event("shutdown")
async def _shutdown():
    await shutdown_runner()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/v1/reset")
async def reset(req: ResetRequest):
    try:
        await reset_session(req.session_id)
        return {"ok": True, "session_id": req.session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """
    SSE stream: screenshot + user query in -> assistant/tool out (stream)
    """

    if req.message.role != "user":
        raise HTTPException(status_code=400, detail="req.message.role must be 'user'")

    async def event_gen():
        try:
            # Convert history to list of dicts if provided
            history_dicts = None
            if req.history:
                history_dicts = [
                    {"role": msg.role, "content": msg.content}
                    for msg in req.history
                ]

            async for ev in run_turn_stream(
                session_id=req.session_id,
                user_message_content=req.message.content,
                system_prompt=req.system_prompt,
                history=history_dicts,  # Pass frontend history (None = use backend session)
                # language=req.language,
                # request=request,  # Pass request for disconnect detection
            ):
                # SSE frame format
                yield f"event: {ev['type']}\n"
                yield f"data: {json.dumps(ev['data'], ensure_ascii=False)}\n\n"

                if await request.is_disconnected():
                    break

        except asyncio.CancelledError:
            return
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
