import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const api = axios.create({
  baseURL: `${API_BASE}/api`,
  timeout: 10000,
  headers: { 'Content-Type': 'application/json' },
})

// ── Types ──────────────────────────────────────────────────────

export interface TransactionIn {
  transaction_id?: string
  amount: number
  card_id?: string
  merchant_id?: string
  device_id?: string
  channel?: string
  product_category?: string
  hour_of_day?: number
  country_mismatch?: boolean
  velocity_1h?: number
  velocity_24h?: number
  card_avg_amount_30d?: number
  merchant_fraud_rate?: number
  dataset_source?: string
}

export interface PredictionOut {
  transaction_id: string
  fraud_probability: number
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
  model_version: string
  latency_ms: number
  explanation: {
    top_features: { name: string; value: number; contribution: number }[]
    confidence: string
  }
  created_at: string
}

export interface StatsOverview {
  total_transactions: number
  fraud_flagged: number
  fraud_rate: number
  avg_latency_ms: number
  risk_distribution: Record<string, number>
  dataset_distribution: Record<string, number>
}

export interface ModelInfo {
  version: string
  test_metrics: Record<string, number>
  in_channels: Record<string, number>
  is_demo: boolean
}

// ── API calls ──────────────────────────────────────────────────

export const predictTransaction = (txn: TransactionIn) =>
  api.post<PredictionOut>('/predict', txn).then(r => r.data)

export const getTransactions = (page = 1, page_size = 25, risk_level?: string) =>
  api.get('/transactions', { params: { page, page_size, risk_level } }).then(r => r.data)

export const getTransaction = (id: string) =>
  api.get(`/transactions/${id}`).then(r => r.data)

export const getStatsOverview = () =>
  api.get<StatsOverview>('/stats/overview').then(r => r.data)

export const getTimeseries = (hours = 24) =>
  api.get('/stats/timeseries', { params: { hours } }).then(r => r.data)

export const getModelInfo = () =>
  api.get<ModelInfo>('/model/info').then(r => r.data)

// ── WebSocket ──────────────────────────────────────────────────

const WS_BASE = import.meta.env.VITE_WS_URL || 'ws://localhost:8000'

export function createFraudScoreSocket(
  onMessage: (data: any) => void,
  onError?: (e: Event) => void
): WebSocket {
  const ws = new WebSocket(`${WS_BASE}/api/ws/scores`)
  ws.onmessage = (e) => {
    try { onMessage(JSON.parse(e.data)) } catch {}
  }
  ws.onerror = onError || (() => {})
  return ws
}
