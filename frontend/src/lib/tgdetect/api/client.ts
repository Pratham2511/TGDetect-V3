/**
 * TGDetect centralized HTTP API client.
 *
 * Communicates with the local FastAPI backend on port 8000.
 * Handles timeouts, JSON serialization, HTTP errors, and typed envelopes.
 */

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_TGDETECT_API_URL || '';

export class TgDetectApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = 'TgDetectApiError';
    this.status = status;
    this.code = code;
  }
}

export interface RequestOptions extends RequestInit {
  timeoutMs?: number;
  params?: Record<string, string | number | boolean | null | undefined>;
}

export async function apiClient<T>(
  path: string,
  options: RequestOptions = {}
): Promise<T> {
  const { timeoutMs = 30000, params, ...init } = options;

  if (path.startsWith('/api/') && !path.startsWith('/api/ui/')) path = path.replace('/api/', '/api/ui/');
  const selectedDataset = typeof window !== 'undefined' ? localStorage.getItem('tgdetect_active_dataset') : null;
  const selectedModel = typeof window !== 'undefined' ? localStorage.getItem('tgdetect_active_model') : null;
  let url = path.startsWith('http') ? path : `${API_BASE_URL}${path.startsWith('/') ? path : `/${path}`}`;

  {
    const searchParams = new URLSearchParams();
    for (const [key, value] of Object.entries({dataset_id: selectedDataset, model_id: selectedModel, ...params})) {
      if (value !== null && value !== undefined && value !== '') {
        searchParams.append(key, String(value));
      }
    }
    const qs = searchParams.toString();
    if (qs) {
      url += (url.includes('?') ? '&' : '?') + qs;
    }
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(url, {
      ...init,
      signal: controller.signal,
      headers: {
        Accept: 'application/json',
        ...init.headers,
      },
    });

    if (!res.ok) {
      let code = `HTTP_${res.status}`;
      let message = res.statusText || 'Request failed';
      try {
        const errorBody = await res.json();
        if (errorBody?.error) {
          code = errorBody.error.code || code;
          message = errorBody.error.message || message;
        } else if (errorBody?.detail) {
          message = typeof errorBody.detail === 'string' ? errorBody.detail : JSON.stringify(errorBody.detail);
        }
      } catch {
        // Body was not JSON
      }
      throw new TgDetectApiError(res.status, code, message);
    }

    return (await res.json()) as T;
  } catch (err: unknown) {
    if (err instanceof TgDetectApiError) {
      throw err;
    }
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new TgDetectApiError(
        408,
        'TIMEOUT',
        `Request to ${path} timed out after ${timeoutMs}ms`
      );
    }
    const msg = err instanceof Error ? err.message : String(err);
    throw new TgDetectApiError(
      0,
      'OFFLINE',
      `Could not connect to TGDetect API (${API_BASE_URL}). ${msg}`
    );
  } finally {
    clearTimeout(timeoutId);
  }
}

export const api = {
  get<T>(path: string, options?: RequestOptions): Promise<T> {
    return apiClient<T>(path, { ...options, method: 'GET' });
  },
  post<T>(path: string, body?: unknown, options?: RequestOptions): Promise<T> {
    const isFormData = typeof FormData !== 'undefined' && body instanceof FormData;
    return apiClient<T>(path, {
      ...options,
      method: 'POST',
      body: isFormData ? (body as FormData) : body !== undefined ? JSON.stringify(body) : undefined,
      headers: isFormData
        ? options?.headers
        : {
            'Content-Type': 'application/json',
            ...options?.headers,
          },
    });
  },
  upload<T>(path: string, formData: FormData, options?: RequestOptions): Promise<T> {
    return apiClient<T>(path, {
      ...options,
      method: 'POST',
      body: formData,
    });
  },
  checkHealth(): Promise<{ status: string; service: string; version: string; backend: string }> {
    return apiClient('/api/health', { timeoutMs: 3000 });
  },
};
