from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router
from src.common.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="对抗样本攻击系统 API",
    description="对抗样本攻击系统的后端接口服务",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    logger.info("API服务启动")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API服务关闭")