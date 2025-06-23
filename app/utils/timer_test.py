from datetime import datetime, timezone


def check_time():
    print(f"UTC time: {datetime.now(timezone.utc)}")
