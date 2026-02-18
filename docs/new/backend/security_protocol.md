# Security Protocol & Best Practices

## 1. Authentication Strategy
We move from "No Auth" (Streamlit) to **"State-of-the-Art" JWT Auth**.

### Tokens
- **Access Token**: Short-lived (15 min), signed with `hs256`. Contains `sub` (user_id) and `permissions`.
- **Refresh Token**: Long-lived (7 days), stored in DB (hashed) and HttpOnly cookie.

### Cookie Policy
Cookies are the only safe storage for tokens in browser apps.
```python
response.set_cookie(
    key="access_token",
    value=token,
    httponly=True,   # No JS access (XSS proof)
    secure=True,     # HTTPS only (Simulated locally via bool flag)
    samesite="Lax",  # CSRF protection
    max_age=900
)
```

## 2. Real-time Security (Socket.IO)
- **Handshake Auth**: The initial Socket.IO handshake MUST validate the JWT cookie.
- **Room Authorization**: Users can only join rooms they are authorized for (though currently single-tenant).

## 3. Input Validation
- **Pydantic V2**: All incoming JSON bodies are strictly typed.
- **Query Params**: Validated for SQL Injection risks (handled by SQLAlchemy ORM, but logic checked).

## 4. Operational Security (OpSec)
- **Environment Variables**: secrets (API Keys, JWT Secret) loaded via `.env`, never hardcoded.
- **CORS**: Strict Allow-Origin policy (only allow Frontend URL).
- **Rate Limiting**: `slowapi` to prevent brute-force on `/login`.

## 5. Audit Logging
- **Trade Audit**: Every manual override via "God Mode" is logged to `audit_logs` table with `user_id`, `timestamp`, `action`, `reason`.
- **System Events**: Startup/Shutdown/Error events logged to `system_logs`.
