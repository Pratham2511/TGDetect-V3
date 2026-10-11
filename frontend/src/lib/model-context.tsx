'use client';

import React, { createContext, useContext, useState, useEffect, useCallback, useMemo } from 'react';
import { api, API_BASE_URL } from '@/lib/tgdetect/api/client';
import { modelService } from '@/lib/tgdetect/services';
import type { ModelMeta } from '@/lib/tgdetect/types';

export type ApiHealthState = 'checking' | 'connected' | 'offline';

export interface ModelContextType {
  models: ModelMeta[];
  activeModelId: string;
  activeModel: ModelMeta | null;
  setActiveModelId: (id: string) => void;
  apiHealth: ApiHealthState;
  healthError: string | null;
  refreshHealth: () => Promise<void>;
  isLoadingModels: boolean;
}

const DEFAULT_MODELS: ModelMeta[] = [];
const ModelContext = createContext<ModelContextType>({
  models: DEFAULT_MODELS,
  activeModelId: '',
  activeModel: null,
  setActiveModelId: () => {},
  apiHealth: 'checking',
  healthError: null,
  refreshHealth: async () => {},
  isLoadingModels: false,
});

export function ModelProvider({ children }: { children: React.ReactNode }) {
  const [models, setModels] = useState<ModelMeta[]>(DEFAULT_MODELS);
  const [activeModelId, setActiveModelIdState] = useState<string>('');
  const [apiHealth, setApiHealth] = useState<ApiHealthState>('checking');
  const [healthError, setHealthError] = useState<string | null>(null);
  const [isLoadingModels, setIsLoadingModels] = useState<boolean>(false);

  const setActiveModelId = useCallback((id: string) => {
    setActiveModelIdState(id);
    try {
      localStorage.setItem('tgdetect_active_model', id);
    } catch {
      // ignore
    }
  }, []);

  const checkHealth = useCallback(() => api.checkHealth().then(async () => {
      setApiHealth('connected');
      setHealthError(null);

      // Fetch live models from authoritative backend
      setIsLoadingModels(true);
      try {
        const liveModels = await modelService.models();
        if (liveModels && liveModels.length > 0) {
          setModels(liveModels);
          setActiveModelIdState(previous => {
            let stored = '';
            try { stored = localStorage.getItem('tgdetect_active_model') || ''; } catch {}
            const urlModel = new URLSearchParams(window.location.search).get('model');
            const preferred = previous || urlModel || stored;
            const id = liveModels.some(m => m.id === preferred) ? preferred! : liveModels[0].id;
            try { localStorage.setItem('tgdetect_active_model', id); } catch {}
            return id;
          });
        }
      } catch (err: any) {
        console.warn('Could not fetch live models from backend:', err);
      } finally {
        setIsLoadingModels(false);
      }
    }).catch((err: any) => {
      setApiHealth('offline');
      setHealthError(err?.message ?? `Cannot reach TGDetect API at ${API_BASE_URL}`);
    }), []);

  const refreshHealth = useCallback(async () => {
    setApiHealth('checking');
    await checkHealth();
  }, [checkHealth]);

  useEffect(() => {
    checkHealth();
    // Poll health periodically every 20 seconds
    const interval = setInterval(checkHealth, 20000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  const activeModel = useMemo(() => {
    return models.find((m) => m.id === activeModelId) ?? models[0] ?? null;
  }, [models, activeModelId]);

  const value = useMemo(
    () => ({
      models,
      activeModelId,
      activeModel,
      setActiveModelId,
      apiHealth,
      healthError,
      refreshHealth,
      isLoadingModels,
    }),
    [models, activeModelId, activeModel, setActiveModelId, apiHealth, healthError, refreshHealth, isLoadingModels]
  );

  return <ModelContext.Provider value={value}>{children}</ModelContext.Provider>;
}

export const useModel = () => useContext(ModelContext);
