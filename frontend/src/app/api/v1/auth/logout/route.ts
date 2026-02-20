import { NextResponse } from "next/server";
import { cookies } from "next/headers";

// DEV MOCK: Simulates the FastAPI logout endpoint
export async function POST() {
  const cookieStore = await cookies();

  cookieStore.set("access_token", "", {
    httpOnly: true,
    secure: false,
    sameSite: "lax",
    path: "/",
    maxAge: 0, // Expire immediately
  });

  return NextResponse.json({ message: "Logged out" });
}
