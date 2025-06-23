#!/bin/bash
cron
uvicorn app.main:app --host 0.0.0.0 --port 8085 --log-config logging.json
