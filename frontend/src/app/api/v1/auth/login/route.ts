import { NextResponse } from "next/server";
import { cookies } from "next/headers";

// DEV MOCK: Simulates the FastAPI login endpoint
// This will be replaced by the real backend at localhost:8000
export async function POST(request: Request) {
  const body = await request.json();
  const { username, password } = body;

  // Simple dev credentials
  if (username === "admin" && password === "admin") {
    const cookieStore = await cookies();

    // Set a mock JWT token as HttpOnly cookie (same as the real backend will do)
    cookieStore.set("access_token", "dev_mock_token_iara_2026", {
      httpOnly: true,
      secure: false, // dev mode
      sameSite: "lax",
      path: "/",
      maxAge: 60 * 60 * 24, // 24 hours
    });

    return NextResponse.json({
      user: { id: "1", username: "admin", role: "admin" },
      message: "Login successful (dev mode)",
    });
  }

  return NextResponse.json(
    { detail: "Invalid credentials. Use admin/admin in dev mode." },
    { status: 401 }
  );
}
