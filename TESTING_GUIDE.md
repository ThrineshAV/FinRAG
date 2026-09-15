# 🎯 HOW TO TEST THE API (Updated for JWT Bearer Auth)

The API uses JWT Bearer authentication only. X-API-Key authentication was removed. All endpoints that previously required X-API-Key now require an `Authorization: Bearer <token>` header.

---

## 1️⃣ START THE SERVER

```bash
# Set JWT_SECRET for development (production pulls from AWS Secrets Manager via ec2_user_data.sh)
export JWT_SECRET="dev-test-secret-key-at-least-32-chars-long!!"
uvicorn src.api:app --reload
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

---

## 2️⃣ TEST WITH JWT BEARER TOKEN (Curl)

First, obtain an access token via `/auth/login`:

```bash
# Login (returns access_token + refresh cookie)
curl -X POST http://127.0.0.1:8000/auth/login \
  -H "Content-Type: application/json" \
  -c cookies.txt \
  -d '{"email":"alice@example.com","password":"SecurePass123"}'
```

Then use the Bearer token for protected endpoints:

```bash
# Health check (no auth needed)
curl http://127.0.0.1:8000/health

# Ready check (no auth needed)
curl http://127.0.0.1:8000/ready

# Query with JWT Bearer header
curl -X POST http://127.0.0.1:8000/query \
  -H "Authorization: Bearer eyJhbGciOiHS256..." \
  -H "Content-Type: application/json" \
  -d '{"question":"What was Apple revenue in 2024?","company":"Apple","fiscal_year":"2024","top_k":5}'

# Upload
curl -X POST http://127.0.0.1:8000/upload \
  -H "Authorization: Bearer eyJhbGciOiHS256..." \
  -F "file=@test.pdf" \
  -F "company=Apple" \
  -F "document_type=10-K" \
  -F "fiscal_year=2024" \
  -F "quarter=Q4"
```

---

## ⚠️ Note: JWT Secret Fallback

If `JWT_SECRET` environment variable is not set (e.g., in some test runners), `src/auth/jwt_utils.py` uses a 32-character fallback (`dev-test-secret-key-at-least-32-chars-long!!`). This allows test collection to succeed without manual env setup. The fallback is suppressed for Bandit (`# nosec B105`). In production, the `ec2_user_data.sh` script pulls the real secret from AWS Secrets Manager (secret name `finsight/prod`).

---

## 📊 CURRENT ENDPOINTS

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/health` | GET | No | Server health check |
| `/ready` | GET | No | Readiness probe |
| `/auth/signup` | POST | No | User signup |
| `/auth/login` | POST | No | User login (JWT + cookie) |
| `/auth/refresh` | POST | Cookie only | Refresh access token |
| `/auth/logout` | POST | Cookie only | Revoke refresh token |
| `/upload` | POST | Bearer (Admin/Uploader) | Upload PDF document |
| `/query` | POST | Bearer (Reader/Admin) | Query documents (direct or stream) |
| `/query/stream` | POST | Bearer (Reader/Admin) | Streaming query response |

---

## ✅ CURRENT CONFIGURATION STATUS

- **Authentication:** JWT Bearer only (`JWT_SECRET` env or AWS Secrets Manager)
- **Role-Based Access:** `READER`, `UPLOADER`, `ADMIN` roles enforced
- **CORS:** Restricted to known origins (not `*`)
- **Rate Limiting:** `/query`: 20/min; `/upload`: 5/min (`RATE_LIMIT_*` env vars)
- **Security Scanning:** Bandit + Trivy passing; B105 suppressed for test fallback
- **Test Environment:** `tests/conftest.py` sets `JWT_SECRET` for test collection

---

## 🚀 QUICK START

1. **Start server:** `export JWT_SECRET=dev-test-secret-key-at-least-32-chars-long!!; uvicorn src.api:app --reload`
2. **Use curl or browser:** Send Bearer token via `Authorization` header
3. **Production deploy:** Use `deploy/ec2_user_data.sh` (pulls JWT_SECRET from AWS Secrets Manager)

---

## 🔧 ENVIRONMENT CONFIGURATION

Ensure these variables are set:

- `JWT_SECRET` (32+ chars) — required for JWT signing
- `AUTH_REQUIRED` — `true` (default) or `false` for local dev
- `JWT_COOKIE_SECURE` — `false` (dev) / `true` (production with HTTPS)
- `OPENAI_API_KEY` — required for grounded answers (optional for evidence retrieval)

---

*Note: The previous `X-API-Key: test123` authentication system has been fully removed. All references to `data/api_keys.json`, `API_KEYS_FILE`, `ADMIN_API_KEY`, and related admin key endpoints (`/admin/keys`) are obsolete.*
