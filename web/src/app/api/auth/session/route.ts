import { NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest) {
  // Extract the access_token from the cookies sent by the browser
  const token = request.cookies.get("access_token")?.value;

  if (!token) {
    return NextResponse.json({ user: null, message: "No session found" }, { status: 401 });
  }

  try {
    // Forward the token to the Python backend to validate and decode it
    // The backend uses the same secret key to verify the JWT
    const backendUrl = process.env.BACKEND_URL || "http://localhost:8000";
    const res = await fetch(`${backendUrl}/api/v1/auth/me`, {
      headers: {
        Cookie: `access_token=${token}`,
      },
      cache: "no-store",
    });

    if (!res.ok) {
      return NextResponse.json({ user: null, message: "Session invalid" }, { status: res.status });
    }

    const userData = await res.json();
    
    // Return the user session data to the frontend
    // This allows the frontend to know who is logged in without reading the HttpOnly cookie directly
    return NextResponse.json({ 
      user: userData,
      authenticated: true 
    });
    
  } catch (error) {
    console.error("Session proxy error:", error);
    return NextResponse.json({ user: null, message: "Internal server error" }, { status: 500 });
  }
}
