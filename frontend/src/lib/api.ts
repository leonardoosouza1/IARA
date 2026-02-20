const API_URL = process.env.NEXT_PUBLIC_API_URL || "";

interface RequestOptions extends RequestInit {
  params?: Record<string, string>;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  private buildUrl(path: string, params?: Record<string, string>): string {
    const fullPath = `${this.baseUrl}${path}`;

    if (params && Object.keys(params).length > 0) {
      const searchParams = new URLSearchParams(params);
      return `${fullPath}?${searchParams.toString()}`;
    }

    return fullPath;
  }


  async get<T>(path: string, options?: RequestOptions): Promise<T> {
    const res = await fetch(this.buildUrl(path, options?.params), {
      method: "GET",
      credentials: "include", // Send HttpOnly cookies
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
    });

    if (!res.ok) {
      if (res.status === 401) {
        window.location.href = "/login";
        throw new Error("Unauthorized");
      }
      throw new Error(`API Error: ${res.status} ${res.statusText}`);
    }

    return res.json();
  }

  async post<T>(path: string, body?: unknown, options?: RequestOptions): Promise<T> {
    const res = await fetch(this.buildUrl(path, options?.params), {
      method: "POST",
      credentials: "include",
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!res.ok) {
      if (res.status === 401) {
        window.location.href = "/login";
        throw new Error("Unauthorized");
      }
      const errorBody = await res.json().catch(() => ({}));
      throw new Error(errorBody.detail || `API Error: ${res.status}`);
    }

    return res.json();
  }

  async delete<T>(path: string, options?: RequestOptions): Promise<T> {
    const res = await fetch(this.buildUrl(path, options?.params), {
      method: "DELETE",
      credentials: "include",
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
    });

    if (!res.ok) throw new Error(`API Error: ${res.status}`);
    return res.json();
  }
}

export const api = new ApiClient(API_URL);
