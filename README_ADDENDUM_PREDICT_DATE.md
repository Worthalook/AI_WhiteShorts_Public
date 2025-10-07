## PREDICT_DATE (Australia/Melbourne)

Because NHL games are scheduled in North America, the SportsData.io game date typically lags Melbourne by ~1 day.

**Rule used by `.github/workflows/projections-poll.yml`:**
- **Before midnight (19:00–23:59)** → `PREDICT_DATE = current Melbourne date`
- **After midnight (00:00–02:59)** → `PREDICT_DATE = current Melbourne date - 1 day`

This ensures the projections job always queries the correct North American game date from Melbourne.
