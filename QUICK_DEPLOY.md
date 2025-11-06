# Concord BI Server - Quick Deploy Reference

## 🚀 One-Line Deployment

```bash
# First time setup (interactive)
./quickstart.sh

# Quick API deployment
./deploy.sh api-only
```

---

## 📋 Common Commands

| Task | Command |
|------|---------|
| **Deploy API** | `./deploy.sh api-only` |
| **Deploy Full Stack** | `./deploy.sh full` |
| **Stop Everything** | `./deploy.sh stop` |
| **Restart Services** | `./deploy.sh restart` |
| **View Logs** | `./deploy.sh logs` |
| **Check Status** | `./deploy.sh status` |
| **Validate Setup** | `./validate-env.sh` |
| **Clean Everything** | `./deploy.sh clean` |

---

## 🌐 Access URLs

| Service | URL |
|---------|-----|
| **API Docs** | http://localhost:8000/docs |
| **Health Check** | http://localhost:8000/health |
| **Metrics** | http://localhost:8000/metrics |

---

## 🧪 Test API

```bash
# Health check
curl http://localhost:8000/health

# Get recommendations
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"customer_id": 410376, "top_n": 50}'
```

---

## 🔧 Quick Fixes

### Port Already in Use
```bash
# Find and kill process on port 8000
lsof -i :8000
kill -9 <PID>
```

### Docker Not Running
```bash
# Start Docker Desktop (macOS)
open -a Docker

# Or check Docker status
docker info
```

### Reset Everything
```bash
./deploy.sh clean      # Remove all containers/volumes
./deploy.sh api-only   # Fresh deployment
```

---

## 📊 Performance

- **Precision@50:** 75.4%
- **P99 Latency:** <400ms
- **Concurrent Users:** 20+

---

## 📚 More Info

- Full Guide: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- Technical Details: [TECHNICAL_STACK.md](TECHNICAL_STACK.md)
- Project Info: [README.md](README.md)

---

**Quick Start:** `./quickstart.sh` | **Help:** `./deploy.sh --help`
