'use client';

import React, { createContext, useContext, useState, useEffect, useCallback, useMemo } from 'react';
import { datasetService } from '@/lib/tgdetect/services';
import type { Dataset } from '@/lib/tgdetect/types';

export interface DatasetContextType {
  activeDatasetId: string;
  activeDataset: Dataset | null;
  datasets: Dataset[];
  isLoadingDatasets: boolean;
  isSwitching: boolean;
  switchingMessage: string;
  targetDatasetName: string;
  datasetVersion: number;
  activateDataset: (datasetId: string) => Promise<void>;
  refreshDatasets: () => Promise<void>;
}

const DatasetContext = createContext<DatasetContextType>({
  activeDatasetId: '',
  activeDataset: null,
  datasets: [],
  isLoadingDatasets: false,
  isSwitching: false,
  switchingMessage: '',
  targetDatasetName: '',
  datasetVersion: 1,
  activateDataset: async () => {},
  refreshDatasets: async () => {},
});

export function DatasetProvider({ children }: { children: React.ReactNode }) {
  const [activeDatasetId, setActiveDatasetId] = useState<string>('');
  const [activeDataset, setActiveDataset] = useState<Dataset | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [isLoadingDatasets, setIsLoadingDatasets] = useState<boolean>(true);
  const [isSwitching, setIsSwitching] = useState<boolean>(false);
  const [switchingMessage, setSwitchingMessage] = useState<string>('');
  const [targetDatasetName, setTargetDatasetName] = useState<string>('');
  const [datasetVersion, setDatasetVersion] = useState<number>(1);

  // Initial load: synchronize with URL param, localStorage, or backend active dataset
  const loadDatasets = useCallback(() => Promise.all([
        datasetService.list().catch(() => []),
        datasetService.getActive().catch(() => null),
      ]).then(([list, activeBackend]) => {
      if (list && list.length > 0) {
        setDatasets(list);
      }

      let chosenId = '';
      if (typeof window !== 'undefined') {
        const urlParam = new URLSearchParams(window.location.search).get('dataset');
        let stored = '';
        try { stored = localStorage.getItem('tgdetect_active_dataset') || ''; } catch {}
        if (urlParam && list.some((d) => d.id === urlParam)) {
          chosenId = urlParam;
        } else if (stored && list.some((d) => d.id === stored)) {
          chosenId = stored;
        } else if (activeBackend?.id && list.some((d) => d.id === activeBackend.id)) {
          chosenId = activeBackend.id;
        } else if (list[0]?.id) {
          chosenId = list[0].id;
        }
      } else if (activeBackend?.id) {
        chosenId = activeBackend.id;
      }

      try { localStorage.setItem('tgdetect_active_dataset', chosenId); } catch {}
      setActiveDatasetId(chosenId);
      const matched = list.find((d) => d.id === chosenId) ?? activeBackend ?? null;
      setActiveDataset(matched);
    }).catch(err => {
      console.warn('Could not refresh datasets:', err);
    }).finally(() => {
      setIsLoadingDatasets(false);
    }), []);

  const refreshDatasets = useCallback(async () => {
    setIsLoadingDatasets(true);
    await loadDatasets();
  }, [loadDatasets]);

  useEffect(() => {
    loadDatasets();
  }, [loadDatasets]);

  // Activate a dataset globally
  const activateDataset = useCallback(
    async (newDatasetId: string) => {
      if (!newDatasetId) return;

      const targetMeta = datasets.find((d) => d.id === newDatasetId);
      const targetName = targetMeta?.name ?? newDatasetId.replace('_', ' ').toUpperCase();

      setTargetDatasetName(targetName);
      setIsSwitching(true);
      setSwitchingMessage(`Activating telemetry partition: ${targetName}...`);

      try {
        // 1. Tell backend to switch active dataset
        const res = await datasetService.activate(newDatasetId);

        // 2. Update local state
        setActiveDatasetId(newDatasetId);
        if (res?.dataset) {
          setActiveDataset(res.dataset);
        } else if (targetMeta) {
          setActiveDataset(targetMeta);
        }

        // 3. Update localStorage and URL state cleanly without full reload
        if (typeof window !== 'undefined') {
          try {
            localStorage.setItem('tgdetect_active_dataset', newDatasetId);
            const url = new URL(window.location.href);
            url.searchParams.set('dataset', newDatasetId);
            window.history.replaceState({}, '', url.toString());
          } catch {
            // ignore
          }
        }

        // 4. Increment dataset version so ALL hooks (useGraphStats, useEvents, useGraphNodes, etc.) immediately re-fetch!
        setDatasetVersion((v) => v + 1);

        // 5. Brief smooth transition for visual acknowledgment
        await new Promise((r) => setTimeout(r, 600));
      } catch (err: any) {
        console.error('Failed to activate dataset:', err);
        setSwitchingMessage(`Activation error: ${err?.message ?? String(err)}`);
        await new Promise((r) => setTimeout(r, 1200));
      } finally {
        setIsSwitching(false);
        setSwitchingMessage('');
      }
    },
    [datasets]
  );

  const value = useMemo(
    () => ({
      activeDatasetId,
      activeDataset,
      datasets,
      isLoadingDatasets,
      isSwitching,
      switchingMessage,
      targetDatasetName,
      datasetVersion,
      activateDataset,
      refreshDatasets,
    }),
    [
      activeDatasetId,
      activeDataset,
      datasets,
      isLoadingDatasets,
      isSwitching,
      switchingMessage,
      targetDatasetName,
      datasetVersion,
      activateDataset,
      refreshDatasets,
    ]
  );

  return <DatasetContext.Provider value={value}>{children}</DatasetContext.Provider>;
}

export const useDataset = () => useContext(DatasetContext);
