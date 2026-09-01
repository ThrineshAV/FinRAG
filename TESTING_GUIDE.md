# 🎯 HOW TO TEST THE API

## ✅ Everything is Fixed!

Your API is now fully functional. Here's how to use it:

---

## 1️⃣ START THE SERVER

```bash
uvicorn src.api:app --reload
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

---

## 2️⃣ TEST WITH FRONTEND (Recommended)

Open any of these in your browser:
- `debug_frontend.html` ← **Best for testing** (shows detailed logs)
- `frontend_simple.html` (simple interface)
- `index.html` (full featured)

**Admin Key to use:**
```
test123
```

---

## 3️⃣ TEST WITH CURL (Command Line)

```bash
# List all keys
curl -H "X-API-Key: test123" http://127.0.0.1:8000/admin/keys

# Create a new key
curl -X POST http://127.0.0.1:8000/admin/keys \
  -H "X-API-Key: test123" \
  -H "Content-Type: application/json" \
  -d '{"name": "test-key", "role": "reader"}'

# Delete a key
curl -X DELETE http://127.0.0.1:8000/admin/keys/admin001 \
  -H "X-API-Key: test123"

# Health check (no auth needed)
curl http://127.0.0.1:8000/health

# Ready check (no auth needed)
curl http://127.0.0.1:8000/ready
```

---

## ⚠️ About Swagger UI (/docs)

The Swagger UI at `http://127.0.0.1:8000/docs` requires you to add the header manually:

1. Open `http://127.0.0.1:8000/docs`
2. Click the padlock icon next to an endpoint
3. Enter: `X-API-Key: test123`
4. Click "Authorize"
5. Try the endpoint

**However, it's easier to use the frontend or curl instead.**

---

## 📊 ALL ENDPOINTS

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Server health check |
| `/ready` | GET | No | Readiness probe |
| `/admin/keys` | GET | ✅ Admin | List all API keys |
| `/admin/keys` | POST | ✅ Admin | Create new API key |
| `/admin/keys/{id}` | DELETE | ✅ Admin | Delete API key |
| `/upload` | POST | ✅ Admin | Upload PDF document |
| `/query` | POST | ✅ Reader/Admin | Query documents (stream or direct) |

---

## 🔑 YOUR TEST KEY

| Property | Value |
|----------|-------|
| **Raw Key** | `test123` |
| **Role** | `admin` |
| **Hash** | `ecd71870d1963316a97e3ac3408c9835ad8cf0f3c1bc703527c30265534f75ae` |
| **Status** | ✅ Active |

---

## ✨ WHAT'S WORKING NOW

✅ API key authentication  
✅ Role-based access control  
✅ Admin endpoints (GET, POST, DELETE)  
✅ CORS enabled for browser requests  
✅ Correct SHA-256 hashing  
✅ Request parameter injection  

---

## 🚀 QUICK START

1. **Start server:** `uvicorn src.api:app --reload`
2. **Open frontend:** `debug_frontend.html`
3. **Enter key:** `test123`
4. **Click buttons:** List Keys, Create Key, Delete Key
5. **Everything works!** ✅

