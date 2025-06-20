#!/bin/bash
cron
uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-config logging.json
