import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Search, ChevronLeft, ChevronRight, Filter } from 'lucide-react'
import { getTransactions } from '../utils/api'
import { RiskBadge, ProbBar, SectionTitle, EmptyState, Spinner } from '../components/ui'

const RISK_FILTERS = ['', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
const DS_FILTERS = ['', 'ieee_cis', 'paysim', 'elliptic', 'yelp_chi', 'fraud_amazon', 'live']

const DS_LABELS: Record<string, string> = {
  ieee_cis: 'IEEE-CIS',
  paysim: 'PaySim',
  elliptic: 'Elliptic',
  yelp_chi: 'YelpChi',
  fraud_amazon: 'Fr.Amazon',
  live: 'Live',
}

// Mock data for demo when API returns empty
const MOCK_ROWS = Array.from({ length: 12 }, (_, i) => ({
  transaction_id: `TXN_${Math.random().toString(36).slice(2, 14).toUpperCase()}`,
  amount: +(Math.random() * 5000).toFixed(2),
  channel: ['online', 'pos', 'atm', 'mobile'][i % 4],
  card_id: `CARD_${Math.random().toString(36).slice(2, 10)}`,
  merchant_id: `MERCH_${Math.random().toString(36).slice(2, 8)}`,
  dataset_source: ['ieee_cis', 'paysim', 'elliptic', 'live'][i % 4],
  created_at: new Date(Date.now() - i * 3_600_000).toISOString(),
  fraud_probability: Math.random(),
  risk_level: ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL', 'LOW', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL', 'LOW', 'LOW', 'MEDIUM'][i],
  latency_ms: +(5 + Math.random() * 30).toFixed(1),
}))

export default function Explorer() {
  const [page, setPage] = useState(1)
  const [riskFilter, setRiskFilter] = useState('')
  const [search, setSearch] = useState('')
  const PAGE_SIZE = 20

  const { data, isLoading } = useQuery({
    queryKey: ['transactions', page, riskFilter],
    queryFn: () => getTransactions(page, PAGE_SIZE, riskFilter || undefined),
    staleTime: 30_000,
  })

  const items = (data?.items?.length ? data.items : MOCK_ROWS)
    .filter(r => !search || r.transaction_id?.toLowerCase().includes(search.toLowerCase()))

  const total = data?.total ?? MOCK_ROWS.length
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE))

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-bold text-white">Transaction Explorer</h1>
        <p className="text-sm text-[#8892b0] mt-1">Browse and filter scored transactions across all datasets</p>
      </div>

      {/* Filters */}
      <div className="card">
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-2 flex-1 min-w-48">
            <Search size={14} className="text-[#4a5568] flex-shrink-0" />
            <input
              placeholder="Search transaction ID…"
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="bg-transparent border-none text-sm p-0 focus:outline-none"
              style={{ outline: 'none' }}
            />
          </div>

          <div className="flex items-center gap-2">
            <Filter size={12} className="text-[#4a5568]" />
            <div className="flex gap-1">
              {RISK_FILTERS.map(r => (
                <button
                  key={r || 'all'}
                  onClick={() => { setRiskFilter(r); setPage(1) }}
                  className={`text-xs px-2.5 py-1 rounded-full mono transition-all ${
                    riskFilter === r
                      ? 'bg-[#00d4ff]/10 text-[#00d4ff] border border-[#00d4ff]/30'
                      : 'text-[#8892b0] border border-transparent hover:border-white/10'
                  }`}
                >
                  {r || 'ALL'}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="card p-0 overflow-hidden">
        <div className="overflow-x-auto">
          {isLoading ? (
            <div className="flex items-center justify-center py-16"><Spinner size={28} /></div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>TRANSACTION ID</th>
                  <th>AMOUNT</th>
                  <th>CHANNEL</th>
                  <th>DATASET</th>
                  <th>RISK</th>
                  <th>FRAUD PROB</th>
                  <th>LATENCY</th>
                  <th>TIME</th>
                </tr>
              </thead>
              <tbody>
                {items.length === 0 ? (
                  <tr><td colSpan={8}><EmptyState message="No transactions found" /></td></tr>
                ) : items.map((row: any) => (
                  <tr key={row.transaction_id}>
                    <td className="mono text-xs font-medium truncate max-w-[140px]">
                      {row.transaction_id?.slice(0, 16)}…
                    </td>
                    <td className="mono font-medium text-[#ccd6f6]">
                      ${Number(row.amount).toLocaleString('en-US', { minimumFractionDigits: 2 })}
                    </td>
                    <td>
                      <span className="text-xs bg-white/5 border border-white/5 rounded px-2 py-0.5">
                        {row.channel ?? '—'}
                      </span>
                    </td>
                    <td>
                      <span className="text-xs text-[#7b2fff] mono">
                        {DS_LABELS[row.dataset_source] ?? row.dataset_source ?? '—'}
                      </span>
                    </td>
                    <td>
                      {row.risk_level ? <RiskBadge level={row.risk_level} /> : <span className="text-[#4a5568]">—</span>}
                    </td>
                    <td className="min-w-[120px]">
                      {row.fraud_probability != null
                        ? <ProbBar value={row.fraud_probability} size="sm" />
                        : <span className="text-[#4a5568]">—</span>
                      }
                    </td>
                    <td className="mono text-xs text-[#4a5568]">
                      {row.latency_ms != null ? `${row.latency_ms}ms` : '—'}
                    </td>
                    <td className="mono text-xs text-[#4a5568]">
                      {row.created_at ? new Date(row.created_at).toLocaleString() : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Pagination */}
        <div className="px-4 py-3 border-t border-white/5 flex items-center justify-between">
          <span className="text-xs text-[#4a5568] mono">
            {total.toLocaleString()} total · page {page} / {totalPages}
          </span>
          <div className="flex items-center gap-1">
            <button
              className="btn-ghost p-1.5"
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page === 1}
            >
              <ChevronLeft size={14} />
            </button>
            {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
              const p = Math.max(1, Math.min(page - 2 + i, totalPages - 4 + i))
              return (
                <button
                  key={p}
                  onClick={() => setPage(p)}
                  className={`w-7 h-7 text-xs rounded mono ${
                    page === p ? 'bg-[#00d4ff]/10 text-[#00d4ff]' : 'text-[#4a5568] hover:text-[#8892b0]'
                  }`}
                >
                  {p}
                </button>
              )
            })}
            <button
              className="btn-ghost p-1.5"
              onClick={() => setPage(p => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
            >
              <ChevronRight size={14} />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
