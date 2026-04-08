import React, { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Zap, AlertOctagon, CheckCircle } from 'lucide-react'
import { predictTransaction, TransactionIn, PredictionOut } from '../utils/api'
import { RiskBadge, ProbBar, SectionTitle, FeatureBar, Spinner } from '../components/ui'

const CHANNELS = ['online', 'pos', 'atm', 'mobile']
const CATEGORIES = ['electronics', 'travel', 'grocery', 'fuel', 'entertainment', 'other']
const DATASETS = ['live', 'ieee_cis', 'paysim', 'elliptic']

const PRESETS = [
  {
    label: 'Normal purchase',
    icon: '🛒',
    data: { amount: 42.50, channel: 'pos', hour_of_day: 14, velocity_1h: 1, velocity_24h: 3, country_mismatch: false, merchant_fraud_rate: 0.001 }
  },
  {
    label: 'High-value online',
    icon: '💻',
    data: { amount: 3200, channel: 'online', hour_of_day: 3, velocity_1h: 5, velocity_24h: 12, country_mismatch: true, merchant_fraud_rate: 0.04 }
  },
  {
    label: 'Suspicious ATM',
    icon: '🏧',
    data: { amount: 800, channel: 'atm', hour_of_day: 2, velocity_1h: 8, velocity_24h: 20, country_mismatch: true, merchant_fraud_rate: 0.12 }
  },
  {
    label: 'Fraud ring',
    icon: '🔴',
    data: { amount: 15000, channel: 'online', hour_of_day: 23, velocity_1h: 15, velocity_24h: 40, country_mismatch: true, merchant_fraud_rate: 0.35 }
  },
]

function ResultPanel({ result }: { result: PredictionOut }) {
  const isFraud = result.risk_level === 'HIGH' || result.risk_level === 'CRITICAL'
  const borderColor = result.risk_level === 'CRITICAL' ? '#ff4d6d' :
                      result.risk_level === 'HIGH' ? '#ff8c42' :
                      result.risk_level === 'MEDIUM' ? '#ffb700' : '#00ff9d'

  return (
    <div className="card" style={{ borderColor, borderWidth: '1px' }}>
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="text-xs text-[#8892b0] mono mb-1">{result.transaction_id.slice(0, 20)}…</div>
          <div className="flex items-center gap-3">
            {isFraud
              ? <AlertOctagon size={22} className="text-[#ff4d6d]" />
              : <CheckCircle size={22} className="text-[#00ff9d]" />
            }
            <span className="text-xl font-bold text-white">
              {isFraud ? 'Fraud Detected' : 'Legitimate'}
            </span>
            <RiskBadge level={result.risk_level} />
          </div>
        </div>
        <div className="text-right">
          <div className="mono text-3xl font-bold" style={{ color: borderColor }}>
            {(result.fraud_probability * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-[#4a5568]">fraud probability</div>
        </div>
      </div>

      {/* Probability bar */}
      <div className="mb-5">
        <ProbBar value={result.fraud_probability} />
      </div>

      {/* Feature explanations */}
      <SectionTitle>Top contributing features</SectionTitle>
      <div className="space-y-1">
        {result.explanation.top_features.map(f => (
          <FeatureBar key={f.name} name={f.name} value={f.value} contribution={f.contribution} />
        ))}
      </div>

      {/* Meta */}
      <div className="mt-4 pt-4 border-t border-white/5 flex items-center justify-between text-xs text-[#4a5568] mono">
        <span>Model: {result.model_version}</span>
        <span>Latency: {result.latency_ms}ms</span>
        <span>Confidence: {result.explanation.confidence}</span>
      </div>
    </div>
  )
}

export default function Predict() {
  const [form, setForm] = useState<Partial<TransactionIn>>({
    amount: 150,
    channel: 'online',
    hour_of_day: 12,
    velocity_1h: 2,
    velocity_24h: 5,
    country_mismatch: false,
    merchant_fraud_rate: 0.01,
    dataset_source: 'live',
  })

  const mutation = useMutation({
    mutationFn: (data: TransactionIn) => predictTransaction(data),
  })

  const set = (key: string, value: any) => setForm(f => ({ ...f, [key]: value }))

  const applyPreset = (preset: typeof PRESETS[0]) => {
    setForm(f => ({ ...f, ...preset.data }))
    mutation.reset()
  }

  const submit = () => {
    mutation.mutate(form as TransactionIn)
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Predict</h1>
        <p className="text-sm text-[#8892b0] mt-1">Score a transaction in real-time using the HTGNN model</p>
      </div>

      {/* Presets */}
      <div>
        <SectionTitle>Quick presets</SectionTitle>
        <div className="grid grid-cols-4 gap-3">
          {PRESETS.map(p => (
            <button
              key={p.label}
              onClick={() => applyPreset(p)}
              className="card hover:border-[#00d4ff]/30 hover:bg-[#0d0d25] transition-all text-left cursor-pointer"
              style={{ borderColor: 'rgba(255,255,255,0.06)' }}
            >
              <div className="text-xl mb-1">{p.icon}</div>
              <div className="text-xs font-medium text-[#ccd6f6]">{p.label}</div>
              <div className="mono text-xs text-[#4a5568] mt-0.5">${p.data.amount.toLocaleString()}</div>
            </button>
          ))}
        </div>
      </div>

      <div className="two-col items-start">
        {/* Form */}
        <div className="card space-y-4">
          <SectionTitle>Transaction details</SectionTitle>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-[#8892b0] block mb-1">Amount ($)</label>
              <input
                type="number"
                value={form.amount}
                onChange={e => set('amount', parseFloat(e.target.value))}
                min="0.01" step="0.01"
              />
            </div>
            <div>
              <label className="text-xs text-[#8892b0] block mb-1">Channel</label>
              <select value={form.channel} onChange={e => set('channel', e.target.value)}>
                {CHANNELS.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <div>
              <label className="text-xs text-[#8892b0] block mb-1">Hour of day (0–23)</label>
              <input
                type="number"
                value={form.hour_of_day}
                onChange={e => set('hour_of_day', parseInt(e.target.value))}
                min="0" max="23"
              />
            </div>
            <div>
              <label className="text-xs text-[#8892b0] block mb-1">Category</label>
              <select value={form.product_category ?? ''} onChange={e => set('product_category', e.target.value)}>
                <option value="">— select —</option>
                {CATEGORIES.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
          </div>

          <div>
            <label className="text-xs text-[#8892b0] block mb-2">
              Velocity — transactions in last hour: <span className="text-[#00d4ff] mono">{form.velocity_1h}</span>
            </label>
            <input
              type="range" min="0" max="30" step="1"
              value={form.velocity_1h}
              onChange={e => set('velocity_1h', parseInt(e.target.value))}
              className="w-full accent-[#00d4ff]"
            />
          </div>

          <div>
            <label className="text-xs text-[#8892b0] block mb-2">
              Velocity — transactions in last 24h: <span className="text-[#00d4ff] mono">{form.velocity_24h}</span>
            </label>
            <input
              type="range" min="0" max="100" step="1"
              value={form.velocity_24h}
              onChange={e => set('velocity_24h', parseInt(e.target.value))}
              className="w-full accent-[#00d4ff]"
            />
          </div>

          <div>
            <label className="text-xs text-[#8892b0] block mb-2">
              Merchant fraud rate: <span className="text-[#ffb700] mono">{((form.merchant_fraud_rate ?? 0) * 100).toFixed(1)}%</span>
            </label>
            <input
              type="range" min="0" max="0.5" step="0.005"
              value={form.merchant_fraud_rate}
              onChange={e => set('merchant_fraud_rate', parseFloat(e.target.value))}
              className="w-full accent-[#ffb700]"
            />
          </div>

          <div className="flex items-center gap-3">
            <input
              type="checkbox"
              id="cross_border"
              checked={form.country_mismatch ?? false}
              onChange={e => set('country_mismatch', e.target.checked)}
              className="w-4 h-4 accent-[#ff4d6d]"
            />
            <label htmlFor="cross_border" className="text-sm text-[#8892b0] cursor-pointer">
              Country mismatch (billing ≠ transaction country)
            </label>
          </div>

          <div>
            <label className="text-xs text-[#8892b0] block mb-1">Dataset source</label>
            <select value={form.dataset_source} onChange={e => set('dataset_source', e.target.value)}>
              {DATASETS.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>

          <button
            className="btn-primary w-full flex items-center justify-center gap-2"
            onClick={submit}
            disabled={mutation.isPending}
          >
            {mutation.isPending ? (
              <><Spinner size={16} /> Scoring…</>
            ) : (
              <><Zap size={15} /> Score Transaction</>
            )}
          </button>
        </div>

        {/* Result */}
        <div>
          {mutation.isPending && (
            <div className="card flex flex-col items-center justify-center py-16 gap-4">
              <Spinner size={32} />
              <p className="text-sm text-[#8892b0]">Running HTGNN inference…</p>
            </div>
          )}
          {mutation.isError && (
            <div className="card border-[#ff4d6d]/30">
              <p className="text-[#ff4d6d] text-sm">Error: {String(mutation.error)}</p>
            </div>
          )}
          {mutation.isSuccess && mutation.data && (
            <ResultPanel result={mutation.data} />
          )}
          {!mutation.isPending && !mutation.isError && !mutation.isSuccess && (
            <div className="card flex flex-col items-center justify-center py-16 text-[#4a5568]">
              <div className="text-4xl mb-3 opacity-20">⬡</div>
              <p className="text-sm">Fill in the form and hit Score</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
