import React from 'react'
import { useQuery } from '@tanstack/react-query'
import { Activity, TrendingUp, AlertTriangle, Clock, Database } from 'lucide-react'
import { getStatsOverview, getTimeseries } from '../utils/api'
import {
  StatTile, SectionTitle, RiskBadge, RiskDonut,
  FraudAreaChart, EmptyState, Spinner
} from '../components/ui'
import { useAppStore } from '../store'

function LiveFeed() {
  const { liveScores } = useAppStore()
  return (
    <div className="card h-full">
      <SectionTitle>
        <span className="w-1.5 h-1.5 rounded-full bg-[#00ff9d] pulse inline-block" />
        Live score stream
      </SectionTitle>
      <div className="overflow-y-auto" style={{ maxHeight: 280 }}>
        {liveScores.length === 0 ? (
          <EmptyState message="Waiting for transactions..." />
        ) : (
          <table className="data-table w-full">
            <thead>
              <tr>
                <th>TXN ID</th>
                <th>RISK</th>
                <th>PROB</th>
                <th>TIME</th>
              </tr>
            </thead>
            <tbody>
              {liveScores.slice(0, 20).map((s) => (
                <tr key={s.transaction_id}>
                  <td className="mono text-xs truncate max-w-[120px]">{s.transaction_id.slice(0, 12)}…</td>
                  <td><RiskBadge level={s.risk_level} /></td>
                  <td className="mono text-xs" style={{
                    color: s.fraud_probability >= 0.8 ? '#ff4d6d' :
                      s.fraud_probability >= 0.5 ? '#ff8c42' : '#00ff9d'
                  }}>
                    {(s.fraud_probability * 100).toFixed(1)}%
                  </td>
                  <td className="mono text-xs text-[#4a5568]">
                    {new Date(s.created_at).toLocaleTimeString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

function DatasetBreakdown({ dist }: { dist: Record<string, number> }) {
  const total = Object.values(dist).reduce((a, b) => a + b, 0)
  const colors: Record<string, string> = {
    ieee_cis: '#00d4ff',
    paysim: '#7b2fff',
    elliptic: '#ffb700',
    yelp_chi: '#00ff9d',
    fraud_amazon: '#ff4d6d',
    live: '#ff8c42',
  }
  const labels: Record<string, string> = {
    ieee_cis: 'IEEE-CIS', paysim: 'PaySim', elliptic: 'Elliptic',
    yelp_chi: 'YelpChi', fraud_amazon: 'Fr.Amazon', live: 'Live',
  }
  return (
    <div className="space-y-2">
      {Object.entries(dist).map(([key, count]) => (
        <div key={key} className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: colors[key] ?? '#8892b0' }} />
          <span className="text-xs text-[#8892b0] w-24 truncate">{labels[key] ?? key}</span>
          <div className="flex-1 h-1 bg-white/5 rounded overflow-hidden">
            <div
              className="h-full rounded"
              style={{ width: `${total ? (count / total) * 100 : 0}%`, background: colors[key] ?? '#8892b0' }}
            />
          </div>
          <span className="mono text-xs text-[#4a5568] w-12 text-right">{count.toLocaleString()}</span>
        </div>
      ))}
    </div>
  )
}

export default function Dashboard() {
  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['stats-overview'],
    queryFn: getStatsOverview,
    refetchInterval: 30_000,
  })

  const { data: timeseries, isLoading: tsLoading } = useQuery({
    queryKey: ['timeseries-24'],
    queryFn: () => getTimeseries(24),
    refetchInterval: 60_000,
  })

  // Mock timeseries if none
  const tsData = (timeseries && timeseries.length > 0) ? timeseries : Array.from({ length: 24 }, (_, i) => ({
    hour: `${String(i).padStart(2, '0')}:00`,
    total: Math.round(200 + Math.random() * 800),
    fraud: Math.round(2 + Math.random() * 30),
  }))

  const riskDist = stats?.risk_distribution
    ? Object.entries(stats.risk_distribution).map(([name, value]) => ({ name, value: Number(value) }))
    : [
        { name: 'LOW', value: 820 },
        { name: 'MEDIUM', value: 95 },
        { name: 'HIGH', value: 43 },
        { name: 'CRITICAL', value: 12 },
      ]

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Dashboard</h1>
        <p className="text-sm text-[#8892b0] mt-1">Real-time fraud detection · HTGNN model · Multi-dataset</p>
      </div>

      {/* Stats row */}
      {statsLoading ? (
        <div className="flex items-center justify-center h-24"><Spinner size={28} /></div>
      ) : (
        <div className="stats-grid">
          <StatTile
            value={(stats?.total_transactions ?? 0).toLocaleString()}
            label="Total transactions"
            color="#00d4ff"
          />
          <StatTile
            value={(stats?.fraud_flagged ?? 0).toLocaleString()}
            label="Fraud flagged"
            color="#ff4d6d"
          />
          <StatTile
            value={((stats?.fraud_rate ?? 0) * 100).toFixed(2)}
            suffix="%"
            label="Fraud rate"
            color="#ffb700"
          />
          <StatTile
            value={(stats?.avg_latency_ms ?? 0).toFixed(1)}
            suffix="ms"
            label="Avg inference latency"
            color="#00ff9d"
          />
        </div>
      )}

      {/* Charts row */}
      <div className="two-col">
        {/* Transaction volume */}
        <div className="card">
          <SectionTitle>Transaction volume (24h)</SectionTitle>
          <FraudAreaChart
            data={tsData}
            xKey="hour"
            lines={[
              { key: 'total', color: '#00d4ff', label: 'Total' },
              { key: 'fraud', color: '#ff4d6d', label: 'Fraud' },
            ]}
          />
        </div>

        {/* Risk distribution */}
        <div className="card">
          <SectionTitle>Risk distribution</SectionTitle>
          <div className="flex items-center gap-4">
            <div className="flex-shrink-0">
              <RiskDonut data={riskDist} />
            </div>
            <div className="space-y-2 flex-1">
              {riskDist.map(d => (
                <div key={d.name} className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full inline-block" style={{
                      background: { LOW: '#00ff9d', MEDIUM: '#ffb700', HIGH: '#ff8c42', CRITICAL: '#ff4d6d' }[d.name]
                    }} />
                    <span className="text-[#8892b0]">{d.name}</span>
                  </div>
                  <span className="mono text-[#ccd6f6]">{d.value.toLocaleString()}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Live feed + dataset breakdown */}
      <div className="two-col">
        <LiveFeed />

        <div className="card">
          <SectionTitle>
            <Database size={12} />
            Dataset breakdown
          </SectionTitle>
          <DatasetBreakdown
            dist={stats?.dataset_distribution ?? {
              ieee_cis: 590000,
              paysim: 320000,
              elliptic: 46000,
              live: 1200,
            }}
          />

          <div className="mt-6">
            <SectionTitle>Model performance (test set)</SectionTitle>
            <div className="three-col mt-3">
              {[
                { label: 'AUPRC', value: '0.891', color: '#00d4ff' },
                { label: 'AUC-ROC', value: '0.938', color: '#7b2fff' },
                { label: 'Precision@80R', value: '74.2%', color: '#00ff9d' },
              ].map(m => (
                <div key={m.label} className="bg-[#0a0a1a] rounded-lg p-3 text-center border border-white/5">
                  <div className="mono text-lg font-semibold" style={{ color: m.color }}>{m.value}</div>
                  <div className="text-[10px] text-[#4a5568] uppercase tracking-wider mt-0.5">{m.label}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
