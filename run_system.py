#!/usr/bin/env python3
"""
对抗样本攻击系统 - 统一启动脚本
支持同时启动API服务和Streamlit前端
"""

import subprocess
import sys
import os
import argparse
import time
import signal

processes = []

def signal_handler(sig, frame):
    print("\n正在关闭所有服务...")
    for p in processes:
        p.terminate()
    sys.exit(0)

def start_api():
    """启动API服务"""
    print("[1/2] 正在启动API服务 (http://localhost:8000)...")
    api_process = subprocess.Popen(
        [sys.executable, "start_api.py"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    processes.append(api_process)
    return api_process

def start_frontend():
    """启动前端服务"""
    print("[2/2] 正在启动前端服务 (http://localhost:8501)...")
    frontend_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run",
         "src/services/frontend/app.py", "--server.port=8501"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    processes.append(frontend_process)
    return frontend_process

def wait_for_services():
    """等待服务启动"""
    print("\n服务已启动!")
    print("=" * 50)
    print("API服务: http://localhost:8000")
    print("API文档: http://localhost:8000/docs")
    print("前端服务: http://localhost:8501")
    print("=" * 50)
    print("\n按 Ctrl+C 停止所有服务")

def main():
    parser = argparse.ArgumentParser(description="对抗样本攻击系统启动器")
    parser.add_argument("--api-only", action="store_true", help="仅启动API服务")
    parser.add_argument("--frontend-only", action="store_true", help="仅启动前端服务")
    parser.add_argument("--no-auto-open", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        if args.api_only:
            start_api()
            wait_for_services()
            while True:
                time.sleep(1)
        elif args.frontend_only:
            start_frontend()
            wait_for_services()
            while True:
                time.sleep(1)
        else:
            start_api()
            time.sleep(2)
            start_frontend()
            wait_for_services()
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in processes:
            p.terminate()

if __name__ == "__main__":
    main()
