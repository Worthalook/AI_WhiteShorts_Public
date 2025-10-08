#!/usr/bin/env python3
from datetime import datetime, timedelta
try:
    from zoneinfo import ZoneInfo
except Exception:
    from backports.zoneinfo import ZoneInfo
tz = ZoneInfo("Australia/Melbourne")
now = datetime.now(tz)
if now.hour <= 2:
    target = (now - timedelta(days=1)).strftime("%Y-%m-%d")
else:
    target = now.strftime("%Y-%m-%d")
print(target)
