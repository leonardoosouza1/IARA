"""
IARA API — Auth Routes
Login, logout, and user identity endpoints.
"""

from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import JSONResponse

from src.api.config import settings
from src.api.schemas import LoginRequest, TokenResponse, UserResponse
from src.api.auth.jwt import (
    create_access_token,
    get_current_user,
    set_auth_cookies,
    clear_auth_cookies,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    """Authenticate user and set HttpOnly cookie with JWT."""

    # Simple credential check (replace with DB lookup in prod)
    if body.username != settings.ADMIN_USERNAME or body.password != settings.ADMIN_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    # Create JWT token
    token = create_access_token(data={
        "sub": "1",
        "username": body.username,
        "role": "admin",
    })

    # Build response with HttpOnly cookie
    user = UserResponse(id="1", username=body.username, role="admin")
    response = JSONResponse(
        content=TokenResponse(user=user, message="Login successful").model_dump()
    )
    set_auth_cookies(response, token)

    return response


@router.post("/logout")
async def logout():
    """Clear auth cookies."""
    response = JSONResponse(content={"message": "Logged out"})
    clear_auth_cookies(response)
    return response


@router.get("/me", response_model=UserResponse)
async def me(request: Request):
    """Return identity of the current authenticated user."""
    payload = get_current_user(request)
    return UserResponse(
        id=payload["sub"],
        username=payload["username"],
        role=payload["role"],
    )
