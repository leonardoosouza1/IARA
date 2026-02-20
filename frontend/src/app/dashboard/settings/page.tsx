"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { Shield, Key, Globe, Server } from "lucide-react";

export default function SettingsPage() {
  return (
    <div className="p-6 space-y-6 max-w-3xl">
      <div>
        <h1 className="text-2xl font-bold text-zinc-100 tracking-tight">Settings</h1>
        <p className="text-sm text-zinc-500 mt-1">System configuration and security settings</p>
      </div>

      {/* API Connection */}
      <Card className="bg-zinc-900/40 border-zinc-800/50">
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2 text-zinc-200">
            <Server className="h-4 w-4 text-indigo-400" />
            API Connection
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm text-zinc-400">Backend URL</label>
            <div className="flex gap-2">
              <Input
                defaultValue="http://localhost:8000"
                className="bg-zinc-800/50 border-zinc-700 text-zinc-200 font-mono text-sm"
              />
              <Button variant="outline" className="border-zinc-700 text-zinc-300 hover:bg-zinc-800">
                Test
              </Button>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20">Connected</Badge>
            <span className="text-xs text-zinc-500">WebSocket: Active (15ms latency)</span>
          </div>
        </CardContent>
      </Card>

      {/* Alpaca Keys */}
      <Card className="bg-zinc-900/40 border-zinc-800/50">
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2 text-zinc-200">
            <Key className="h-4 w-4 text-amber-400" />
            Alpaca API Keys
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm text-zinc-400">API Key</label>
            <Input
              type="password"
              placeholder="PK**********"
              className="bg-zinc-800/50 border-zinc-700 text-zinc-200 font-mono text-sm"
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm text-zinc-400">Secret Key</label>
            <Input
              type="password"
              placeholder="••••••••••••"
              className="bg-zinc-800/50 border-zinc-700 text-zinc-200 font-mono text-sm"
            />
          </div>
          <Button className="bg-indigo-600 hover:bg-indigo-700 text-white">
            Save Keys
          </Button>
        </CardContent>
      </Card>

      {/* Security */}
      <Card className="bg-zinc-900/40 border-zinc-800/50">
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2 text-zinc-200">
            <Shield className="h-4 w-4 text-emerald-400" />
            Security
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          <div className="flex justify-between items-center">
            <span className="text-zinc-400">Session Cookie</span>
            <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20">HttpOnly + Secure</Badge>
          </div>
          <Separator className="bg-zinc-800/40" />
          <div className="flex justify-between items-center">
            <span className="text-zinc-400">CORS Policy</span>
            <span className="text-zinc-300 font-mono text-xs">localhost:3000</span>
          </div>
          <Separator className="bg-zinc-800/40" />
          <div className="flex justify-between items-center">
            <span className="text-zinc-400">Token Expiry</span>
            <span className="text-zinc-300 font-mono text-xs">15 min</span>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
