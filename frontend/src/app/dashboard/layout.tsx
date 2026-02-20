"use client";

import React, { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { SocketProvider } from "@/components/providers/socket-provider";
import { useFleetStore } from "@/hooks/use-fleet-store";
import {
  LayoutDashboard,
  Crosshair,
  Brain,
  Settings,
  ChevronLeft,
  ChevronRight,
  Wifi,
  WifiOff,
  Zap,
  LogOut,
} from "lucide-react";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { useRouter } from "next/navigation";

const navigation = [
  {
    label: "Fleet Overview",
    href: "/dashboard",
    icon: LayoutDashboard,
  },
  {
    label: "Agents",
    href: "/dashboard/agents",
    icon: Crosshair,
  },
  {
    label: "Brain Lab",
    href: "/dashboard/brain-lab",
    icon: Brain,
  },
  {
    label: "Settings",
    href: "/dashboard/settings",
    icon: Settings,
  },
];

function SidebarContent({ collapsed }: { collapsed: boolean }) {
  const pathname = usePathname();
  const socketConnected = useFleetStore((s) => s.socketConnected);
  const router = useRouter();

  async function handleLogout() {
    try {
      await api.post("/api/v1/auth/logout");
    } catch {
      // Even if API fails, clear the cookie client-side
    }
    router.push("/login");
  }

  return (
    <div className="flex flex-col h-full">
      {/* Brand */}
      <div
        className={cn(
          "flex items-center gap-3 px-4 py-5",
          collapsed && "justify-center px-2",
        )}
      >
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-linear-to-br from-indigo-500 to-purple-600 shadow-lg shadow-indigo-500/20">
          <Zap className="h-5 w-5 text-white" />
        </div>
        {!collapsed && (
          <div className="overflow-hidden">
            <h1 className="text-base font-bold tracking-tight text-zinc-100 truncate">
              I.A.R.A.
            </h1>
            <p className="text-[10px] text-zinc-500 font-medium uppercase tracking-widest">
              Mission Control
            </p>
          </div>
        )}
      </div>

      <Separator className="bg-zinc-800/60" />

      {/* Navigation */}
      <nav className="flex-1 px-2 py-4 space-y-1">
        {navigation.map((item) => {
          const isActive =
            pathname === item.href ||
            (item.href !== "/dashboard" && pathname.startsWith(item.href));
          const Icon = item.icon;

          const linkContent = (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-150",
                isActive
                  ? "bg-indigo-500/10 text-indigo-400 shadow-sm"
                  : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50",
                collapsed && "justify-center px-2",
              )}
            >
              <Icon
                className={cn(
                  "h-5 w-5 shrink-0",
                  isActive && "text-indigo-400",
                )}
              />
              {!collapsed && <span className="truncate">{item.label}</span>}
            </Link>
          );

          if (collapsed) {
            return (
              <Tooltip key={item.href}>
                <TooltipTrigger asChild>{linkContent}</TooltipTrigger>
                <TooltipContent
                  side="right"
                  className="bg-zinc-800 text-zinc-200 border-zinc-700"
                >
                  {item.label}
                </TooltipContent>
              </Tooltip>
            );
          }

          return linkContent;
        })}
      </nav>

      <Separator className="bg-zinc-800/60" />

      {/* Footer */}
      <div className={cn("px-3 py-4 space-y-3", collapsed && "px-2")}>
        {/* Connection Status */}
        <div
          className={cn(
            "flex items-center gap-2 text-xs",
            collapsed && "justify-center",
          )}
        >
          {socketConnected ? (
            <>
              <Wifi className="h-3.5 w-3.5 text-emerald-400" />
              {!collapsed && (
                <span className="text-emerald-400 font-medium">Connected</span>
              )}
            </>
          ) : (
            <>
              <WifiOff className="h-3.5 w-3.5 text-red-400 animate-pulse" />
              {!collapsed && (
                <span className="text-red-400 font-medium">Disconnected</span>
              )}
            </>
          )}
        </div>

        {/* Logout */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size={collapsed ? "icon" : "sm"}
              onClick={handleLogout}
              className="w-full text-zinc-500 hover:text-red-400 hover:bg-red-500/10"
            >
              <LogOut className="h-4 w-4" />
              {!collapsed && <span className="ml-2">Logout</span>}
            </Button>
          </TooltipTrigger>
          {collapsed && (
            <TooltipContent
              side="right"
              className="bg-zinc-800 text-zinc-200 border-zinc-700"
            >
              Logout
            </TooltipContent>
          )}
        </Tooltip>
      </div>
    </div>
  );
}

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <SocketProvider>
      <div className="flex h-screen overflow-hidden bg-zinc-950">
        {/* Sidebar */}
        <aside
          className={cn(
            "relative flex flex-col border-r border-zinc-800/60 bg-zinc-900/50 backdrop-blur-sm transition-all duration-300",
            collapsed ? "w-16" : "w-60"
          )}
        >
          <SidebarContent collapsed={collapsed} />

          {/* Collapse Button */}
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="absolute -right-3 top-20 flex h-6 w-6 items-center justify-center rounded-full border border-zinc-700 bg-zinc-800 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700 transition-colors z-10"
          >
            {collapsed ? <ChevronRight className="h-3 w-3" /> : <ChevronLeft className="h-3 w-3" />}
          </button>
        </aside>

        {/* Main Content */}
        <main className="flex-1 overflow-auto">
          {children}
        </main>
      </div>
    </SocketProvider>
  );
}
