#!/usr/bin/env bash
cd "$(dirname "$0")"
PORT="${1:-7891}"
[ -f "gradio-$PORT.pid" ] && kill $(cat "gradio-$PORT.pid") && rm -f "gradio-$PORT.pid" || echo "No PID file for $PORT"
