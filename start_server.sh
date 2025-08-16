#!/usr/bin/env bash
cd "$(dirname "$0")"
source .venv/bin/activate
PORT="${1:-7891}"
pids=$(lsof -ti :"$PORT"); [ -n "$pids" ] && kill $pids || true
export PORT
nohup python serve.py > "gradio-$PORT.log" 2>&1 & echo $! > "gradio-$PORT.pid"
