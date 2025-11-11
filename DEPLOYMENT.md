# Concord BI Server - Deployment Guide

Complete guide for deploying the Concord BI Server with automated recommendations and forecasting.

## Quick Start

### 1. One-Command Installation

```bash
# Download and run installation script
chmod +x install.sh
sudo ./install.sh  # Use sudo on Linux, omit on macOS
```

This will:
- ✅ Install all Python dependencies
- ✅ Set up and start Redis
- ✅ Configure environment variables
- ✅ Install systemd/launchd services
- ✅ Set up automated weekly scheduling
- ✅ Generate initial recommendations and forecasts
- ✅ Start the API server

### 2. Configure Database

Edit `.env` file with your database credentials:

```bash
nano .env
```

Required settings:
```env
DB_HOST=your_db_host
DB_PORT=1433
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password
```

### 3. Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# Check Redis
redis-cli ping

# Check scheduled tasks
# Linux:
sudo systemctl status forecast-worker.timer
sudo systemctl status recommendation-worker.timer

# macOS:
launchctl list | grep concord
```

---

## Manual Installation (Alternative)

If you prefer manual installation:

### Step 1: Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Install Redis

**macOS:**
```bash
brew install redis
brew services start redis
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

### Step 3: Configure Environment

Create `.env` file (see configuration section above).

### Step 4: Run Initial Data Generation

```bash
source venv/bin/activate

# Generate recommendations (10-15 minutes)
python3 scripts/weekly_recommendation_worker.py

# Generate forecasts (15-20 minutes)
python3 scripts/forecasting/forecast_worker.py
```

### Step 5: Start API Server

```bash
# Development
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Production
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Automated Scheduling

### What Gets Scheduled

The installation script sets up two weekly jobs:

1. **Recommendations Worker**
   - Runs: Every Sunday at 2:00 AM
   - Duration: ~10-15 minutes
   - Generates: 25 product recommendations for all active customers
   - Cache: Redis with 7-day TTL

2. **Forecast Worker**
   - Runs: Every Sunday at 3:00 AM
   - Duration: ~15-20 minutes
   - Generates: 12-week forecasts for all products
   - Cache: Redis with 7-day TTL

### Linux (systemd)

**View scheduled tasks:**
```bash
sudo systemctl list-timers
```

**Manually trigger:**
```bash
sudo systemctl start recommendation-worker.service
sudo systemctl start forecast-worker.service
```

**View logs:**
```bash
sudo journalctl -u recommendation-worker -f
sudo journalctl -u forecast-worker -f

# Or check log files
tail -f /var/log/recommendation_worker.log
tail -f /var/log/forecast_worker.log
```

**Stop/disable:**
```bash
sudo systemctl stop forecast-worker.timer
sudo systemctl disable forecast-worker.timer
```

### macOS (launchd)

**View scheduled tasks:**
```bash
launchctl list | grep concord
```

**Manually trigger:**
```bash
launchctl start com.concord.bi.recommendations
launchctl start com.concord.bi.forecast
```

**View logs:**
```bash
tail -f ~/Library/Logs/concord-bi/recommendation_worker.log
tail -f ~/Library/Logs/concord-bi/forecast_worker.log
```

**Stop/disable:**
```bash
launchctl unload ~/Library/LaunchAgents/com.concord.bi.forecast.plist
```

---

## Service Management

### Linux (systemd)

**API Server:**
```bash
# Start
sudo systemctl start concord-api

# Stop
sudo systemctl stop concord-api

# Restart
sudo systemctl restart concord-api

# Status
sudo systemctl status concord-api

# Enable auto-start on boot
sudo systemctl enable concord-api

# View logs
sudo journalctl -u concord-api -f
```

**Workers:**
```bash
# Enable/disable weekly schedule
sudo systemctl enable forecast-worker.timer
sudo systemctl disable forecast-worker.timer

# Check next run time
sudo systemctl list-timers | grep forecast
```

### macOS (launchd)

**API Server:**
```bash
# Start
launchctl start com.concord.bi.api

# Stop
launchctl stop com.concord.bi.api

# Reload configuration
launchctl unload ~/Library/LaunchAgents/com.concord.bi.api.plist
launchctl load ~/Library/LaunchAgents/com.concord.bi.api.plist

# View logs
tail -f logs/api.log
```

---

## Cache Management

### View Cache Status

```bash
# Check number of cached items
redis-cli DBSIZE

# View memory usage
redis-cli INFO memory

# List all forecast keys
redis-cli KEYS "forecast:product:*"

# List all recommendation keys
redis-cli KEYS "recommendations:customer:*"
```

### Clear Caches

**Clear all forecasts:**
```bash
redis-cli KEYS "forecast:product:*" | xargs redis-cli DEL
```

**Clear all recommendations:**
```bash
redis-cli KEYS "recommendations:customer:*" | xargs redis-cli DEL
```

**Clear everything:**
```bash
redis-cli FLUSHDB
```

**Via API:**
```bash
# Clear specific customer
curl -X DELETE "http://localhost:8000/cache/410376"

# Clear all (careful!)
curl -X POST "http://localhost:8000/cache/clear-all"
```

---

## Monitoring

### Health Checks

```bash
# API health
curl http://localhost:8000/health | jq

# Metrics
curl http://localhost:8000/metrics | jq

# Redis health
redis-cli ping
redis-cli INFO server
```

### Performance Monitoring

```bash
# Monitor Redis commands in real-time
redis-cli monitor

# Check slow queries
redis-cli slowlog get 10

# Check API response times
curl -w "\nTime: %{time_total}s\n" http://localhost:8000/forecast/25211473
```

### Log Monitoring

**Linux:**
```bash
# All services
sudo journalctl -f

# Specific service
sudo journalctl -u concord-api -f
sudo journalctl -u forecast-worker -f

# Last 100 lines
sudo journalctl -u concord-api -n 100
```

**macOS:**
```bash
# API logs
tail -f logs/api.log

# Worker logs
tail -f logs/forecast_worker.log
tail -f logs/recommendation_worker.log
```

---

## Troubleshooting

### API Not Starting

1. Check port availability:
   ```bash
   lsof -i :8000
   ```

2. Check logs:
   ```bash
   # Linux
   sudo journalctl -u concord-api -n 50

   # macOS
   tail -n 50 logs/api_error.log
   ```

3. Test manually:
   ```bash
   source venv/bin/activate
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

### Redis Connection Issues

1. Check if Redis is running:
   ```bash
   redis-cli ping
   ```

2. Check Redis logs:
   ```bash
   # Linux
   sudo journalctl -u redis -n 50

   # macOS
   tail -n 50 /usr/local/var/log/redis.log
   ```

3. Restart Redis:
   ```bash
   # Linux
   sudo systemctl restart redis-server

   # macOS
   brew services restart redis
   ```

### Database Connection Issues

1. Verify .env configuration
2. Test connection:
   ```python
   from api.db_pool import get_connection
   conn = get_connection()
   cursor = conn.cursor()
   cursor.execute("SELECT @@VERSION")
   print(cursor.fetchone())
   ```

### Workers Not Running

1. Check timer status:
   ```bash
   # Linux
   sudo systemctl status forecast-worker.timer
   sudo systemctl list-timers

   # macOS
   launchctl list | grep concord
   ```

2. Run manually to test:
   ```bash
   source venv/bin/activate
   python3 scripts/forecasting/forecast_worker.py
   ```

3. Check logs for errors

---

## Updating the System

### Pull Latest Code

```bash
cd /path/to/Concord-BI-Server
git pull origin feature/product-forecasting
```

### Update Dependencies

```bash
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

### Restart Services

**Linux:**
```bash
sudo systemctl daemon-reload
sudo systemctl restart concord-api
```

**macOS:**
```bash
launchctl unload ~/Library/LaunchAgents/com.concord.bi.api.plist
launchctl load ~/Library/LaunchAgents/com.concord.bi.api.plist
```

### Trigger Relearn

After major updates, regenerate all data:

```bash
# Clear caches
redis-cli FLUSHDB

# Regenerate
source venv/bin/activate
python3 scripts/weekly_recommendation_worker.py
python3 scripts/forecasting/forecast_worker.py
```

---

## Production Best Practices

### Security

1. **Use environment variables for secrets**
   - Never commit `.env` to git
   - Use secure password storage

2. **Enable HTTPS**
   - Use nginx/Apache as reverse proxy
   - Configure SSL certificates

3. **Restrict API access**
   - Configure CORS properly
   - Use API keys/authentication

### Performance

1. **Monitor Redis memory**
   ```bash
   redis-cli INFO memory | grep used_memory_human
   ```

2. **Set up Redis persistence**
   - Edit `/etc/redis/redis.conf`
   - Enable RDB snapshots or AOF

3. **Use multiple workers**
   - Set `API_WORKERS=4` in .env
   - Adjust based on CPU cores

### Backup

1. **Database backups**
   - Regular SQL Server backups
   - Test restore procedures

2. **Redis backups**
   ```bash
   redis-cli BGSAVE
   cp /var/lib/redis/dump.rdb /backup/
   ```

3. **Code backups**
   - Git repository
   - Tagged releases

---

## API Endpoints

### Forecasting

```bash
# Get product forecast
curl "http://localhost:8000/forecast/25211473?forecast_weeks=12"

# Custom date
curl "http://localhost:8000/forecast/25211473?as_of_date=2025-11-10"
```

### Recommendations

```bash
# Get customer recommendations
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"customer_id": 410376, "top_n": 50}'

# Get weekly recommendations
curl "http://localhost:8000/weekly-recommendations/410376"
```

### Monitoring

```bash
# Health check
curl "http://localhost:8000/health"

# Metrics
curl "http://localhost:8000/metrics"

# Interactive docs
open http://localhost:8000/docs
```

---

## Support

For issues or questions:
1. Check logs first
2. Review troubleshooting section
3. Check GitHub issues
4. Contact development team

---

Generated with [Claude Code](https://claude.com/claude-code)
