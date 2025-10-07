#!/usr/bin/env python3
"""
Compute PREDICT_DATE for SportsData projections using Australia/Melbourne local time.

Rule:
- Before midnight (19:00–23:59) → today's Melbourne date
- After midnight (00:00–02:59)  → yesterday's Melbourne date
If run outside that window, we still apply:
- 00:00–02:59 → yesterday; else → today
"""
from datetime import datetime, timedelta
try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    from backports.zoneinfo import ZoneInfo  # fallback if installed

tz = ZoneInfo("Australia/Melbourne")
now = datetime.now(tz)
if now.hour <= 2:
    target = (now - timedelta(days=1)).strftime("%Y-%b-%d")
else:
    target = now.strftime("%Y-%b-%d")
print(target)
