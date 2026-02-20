"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Brain, FlaskConical, Database } from "lucide-react";

export default function BrainLabPage() {
  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-zinc-100 tracking-tight">Brain Lab</h1>
        <p className="text-sm text-zinc-500 mt-1">
          Train, evaluate, and deploy AI models
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="bg-zinc-900/40 border-zinc-800/50 hover:border-indigo-500/30 transition-colors">
          <CardContent className="p-6 text-center space-y-3">
            <div className="mx-auto h-12 w-12 rounded-xl bg-indigo-500/10 flex items-center justify-center">
              <Database className="h-6 w-6 text-indigo-400" />
            </div>
            <CardTitle className="text-base text-zinc-200">Data Pipeline</CardTitle>
            <p className="text-xs text-zinc-500">Fetch, process, and prepare training data for crypto and forex markets.</p>
          </CardContent>
        </Card>

        <Card className="bg-zinc-900/40 border-zinc-800/50 hover:border-purple-500/30 transition-colors">
          <CardContent className="p-6 text-center space-y-3">
            <div className="mx-auto h-12 w-12 rounded-xl bg-purple-500/10 flex items-center justify-center">
              <Brain className="h-6 w-6 text-purple-400" />
            </div>
            <CardTitle className="text-base text-zinc-200">Training</CardTitle>
            <p className="text-xs text-zinc-500">Launch training sessions with configurable hyperparameters.</p>
          </CardContent>
        </Card>

        <Card className="bg-zinc-900/40 border-zinc-800/50 hover:border-emerald-500/30 transition-colors">
          <CardContent className="p-6 text-center space-y-3">
            <div className="mx-auto h-12 w-12 rounded-xl bg-emerald-500/10 flex items-center justify-center">
              <FlaskConical className="h-6 w-6 text-emerald-400" />
            </div>
            <CardTitle className="text-base text-zinc-200">Model Zoo</CardTitle>
            <p className="text-xs text-zinc-500">Browse, compare, and deploy trained models to the fleet.</p>
          </CardContent>
        </Card>
      </div>

      <Card className="bg-zinc-900/30 border-zinc-800/40">
        <CardContent className="p-8 text-center text-zinc-500">
          <p className="text-sm">Brain Lab features will be available once the backend API is connected.</p>
          <p className="text-xs text-zinc-600 mt-2">Training pipeline, feature factory, and model management are managed via REST API.</p>
        </CardContent>
      </Card>
    </div>
  );
}
