"""
IARA API — JWT Authentication Logic
Handles token creation, validation, and cookie management.
"""

from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse

from src.api.config import settings


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a signed JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def verify_token(token: str) -> dict:
    """Verify and decode a JWT token. Raises HTTPException on failure."""
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


def get_current_user(request: Request) -> dict:
    """Extract current user from HttpOnly cookie. Used as dependency."""
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    return verify_token(token)


def set_auth_cookies(response: JSONResponse, token: str) -> JSONResponse:
    """Set HttpOnly, Secure, SameSite cookies on the response."""
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=not settings.DEBUG,  # Secure=True in production
        samesite="lax",
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )
    return response


def clear_auth_cookies(response: JSONResponse) -> JSONResponse:
    """Clear auth cookies."""
    response.delete_cookie(key="access_token", path="/")
    return response
