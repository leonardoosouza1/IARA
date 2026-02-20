"use client";

import { Suspense, useState, FormEvent } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { api } from "@/lib/api";
import { Zap } from "lucide-react";

function LoginForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const redirectTo = searchParams.get("redirect") || "/dashboard";

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      await api.post("/api/v1/auth/login", { username, password });
      router.push(redirectTo);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card className="relative w-full max-w-md bg-zinc-900/80 backdrop-blur-xl border-zinc-800 shadow-2xl shadow-indigo-500/5">
      <CardHeader className="text-center space-y-4 pb-2">
        {/* Logo */}
        <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-linear-to-br from-indigo-500 to-purple-600 shadow-lg shadow-indigo-500/25">
          <Zap className="h-8 w-8 text-white" />
        </div>

        <div>
          <CardTitle className="text-2xl font-bold tracking-tight text-zinc-100">
            I.A.R.A.
          </CardTitle>
          <CardDescription className="text-zinc-400 text-sm mt-1">
            Intelligent Autonomous Real-time Agent
          </CardDescription>
        </div>
      </CardHeader>

      <CardContent className="pt-4">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <label
              htmlFor="username"
              className="text-sm font-medium text-zinc-300"
            >
              Username
            </label>
            <Input
              id="username"
              type="text"
              placeholder="operator"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="bg-zinc-800/50 border-zinc-700 text-zinc-100 placeholder:text-zinc-500 focus:border-indigo-500 focus:ring-indigo-500/20"
              required
              autoComplete="username"
            />
          </div>

          <div className="space-y-2">
            <label
              htmlFor="password"
              className="text-sm font-medium text-zinc-300"
            >
              Password
            </label>
            <Input
              id="password"
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="bg-zinc-800/50 border-zinc-700 text-zinc-100 placeholder:text-zinc-500 focus:border-indigo-500 focus:ring-indigo-500/20"
              required
              autoComplete="current-password"
            />
          </div>

          {error && (
            <div className="rounded-lg bg-red-500/10 border border-red-500/20 px-4 py-3 text-sm text-red-400">
              {error}
            </div>
          )}

          <Button
            type="submit"
            disabled={loading}
            className="w-full bg-linear-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white font-semibold shadow-lg shadow-indigo-500/25 transition-all duration-200"
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                Authenticating...
              </span>
            ) : (
              "Access Mission Control"
            )}
          </Button>
        </form>

        <p className="text-xs text-zinc-500 text-center mt-6">
          Secured with encrypted session cookies
        </p>
      </CardContent>
    </Card>
  );
}

export default function LoginPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-zinc-950 relative overflow-hidden">
      {/* Gradient Background */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,var(--tw-gradient-stops))] from-indigo-900/20 via-zinc-950 to-zinc-950" />

      {/* Grid Pattern */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-size-[64px_64px]" />

      <Suspense
        fallback={
          <div className="flex items-center gap-3 text-zinc-500">
            <span className="h-5 w-5 animate-spin rounded-full border-2 border-zinc-700 border-t-indigo-500" />
          </div>
        }
      >
        <LoginForm />
      </Suspense>
    </div>
  );
}
