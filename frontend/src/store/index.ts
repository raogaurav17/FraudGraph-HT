import { create } from 'zustand'

interface LiveScore {
  transaction_id: string
  fraud_probability: number
  risk_level: string
  amount?: number
  created_at: string
}

interface AppStore {
  liveScores: LiveScore[]
  addLiveScore: (score: LiveScore) => void
  clearLiveScores: () => void

  wsConnected: boolean
  setWsConnected: (v: boolean) => void

  selectedDataset: string
  setSelectedDataset: (ds: string) => void

  alertThreshold: number
  setAlertThreshold: (t: number) => void
}

export const useAppStore = create<AppStore>((set) => ({
  liveScores: [],
  addLiveScore: (score) =>
    set((s) => ({
      liveScores: [score, ...s.liveScores].slice(0, 100), // keep last 100
    })),
  clearLiveScores: () => set({ liveScores: [] }),

  wsConnected: false,
  setWsConnected: (v) => set({ wsConnected: v }),

  selectedDataset: 'all',
  setSelectedDataset: (ds) => set({ selectedDataset: ds }),

  alertThreshold: 0.8,
  setAlertThreshold: (t) => set({ alertThreshold: t }),
}))
