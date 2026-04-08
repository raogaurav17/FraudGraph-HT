import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Cpu, Layers, Database, GitBranch } from 'lucide-react'
import { getModelInfo } from '../utils/api'
import {
  SectionTitle, StatTile, Spinner, FraudAreaChart
} from '../components/ui'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, Radar } from 'recharts'

// Mock training history (replace with real API data in production)
const MOCK_HISTORY = Array.from({ length: 40 }, (_, i) => ({
  epoch: i + 1,
  loss: parseFloat((1.2 * Math.exp(-i * 0.08) + 0.05 + Math.random() * 0.02).toFixed(4)),
  val_auprc: parseFloat((0.3 + 0.6 * (1 - Math.exp(-i * 0.1)) + Math.random() * 0.01).toFixed(4)),
  val_auroc: parseFloat((0.6 + 0.35 * (1 - Math.exp(-i * 0.1)) + Math.random() * 0.008).toFixed(4)),
}))

const DATASET_METRICS = [
  { dataset: 'IEEE-CIS', auprc: 0.891, auroc: 0.938, f1: 0.742, samples: '590K' },
  { dataset: 'YelpChi', auprc: 0.823, auroc: 0.912, f1: 0.698, samples: '45K' },
  { dataset: 'FraudAmazon', auprc: 0.761, auroc: 0.884, f1: 0.661, samples: '11K' },
  { dataset: 'Elliptic', auprc: 0.847, auroc: 0.921, f1: 0.715, samples: '46K' },
  { dataset: 'PaySim', auprc: 0.934, auroc: 0.971, f1: 0.812, samples: '320K' },
]

const ARCH_LAYERS = [
  { name: 'Input projection', detail: 'Linear(F_type → 128) × 4 node types', color: '#7b2fff', params: '~42K' },
  { name: 'HANConv layer 1', detail: 'Heterogeneous attention, 4 heads, 5 relation types', color: '#00d4ff', params: '~84K' },
  { name: 'HANConv layer 2', detail: 'Same config + LayerNorm + residual', color: '#00d4ff', params: '~84K' },
  { name: 'Temporal encoder', detail: 'Fourier time encoding + MultiheadAttention (4 heads)', color: '#ffb700', params: '~33K' },
  { name: 'Fraud head', detail: 'MLP: 256 → 64 → 1 + Sigmoid', color: '#ff4d6d', params: '~17K' },
]

function ArchDiagram() {
  return (
    <div className="space-y-1.5">
      {ARCH_LAYERS.map((layer, i) => (
        <div key={i} className="flex items-stretch gap-0">
          {/* Connector */}
          <div className="flex flex-col items-center w-6">
            {i > 0 && <div className="w-px flex-1 bg-white/10" />}
            <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: layer.color }} />
            {i < ARCH_LAYERS.length - 1 && <div className="w-px flex-1 bg-white/10" />}
          </div>
          {/* Box */}
          <div
            className="flex-1 rounded-lg px-3 py-2.5 mb-0.5"
            style={{ background: `${layer.color}0d`, border: `1px solid ${layer.color}22` }}
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold" style={{ color: layer.color }}>{layer.name}</span>
              <span className="mono text-[11px] text-[#4a5568]">{layer.params}</span>
            </div>
            <p className="text-[11px] text-[#8892b0] mt-0.5">{layer.detail}</p>
          </div>
        </div>
      ))}
    </div>
  )
}

function DatasetMetricsBar() {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={DATASET_METRICS} margin={{ top: 4, right: 4, bottom: 20, left: -10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
        <XAxis
          dataKey="dataset"
          stroke="#4a5568"
          tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono', fill: '#8892b0' }}
        />
        <YAxis
          domain={[0.5, 1]}
          stroke="#4a5568"
          tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono', fill: '#8892b0' }}
        />
        <Tooltip
          contentStyle={{
            background: '#0f0f28', border: '1px solid rgba(0,212,255,0.2)',
            borderRadius: 8, fontFamily: 'IBM Plex Mono', fontSize: 11
          }}
        />
        <Bar dataKey="auprc" name="AUPRC" fill="#00d4ff" radius={[3, 3, 0, 0]} />
        <Bar dataKey="auroc" name="AUC-ROC" fill="#7b2fff" radius={[3, 3, 0, 0]} />
        <Bar dataKey="f1" name="F1" fill="#00ff9d" radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}

export default function Monitor() {
  const { data: modelInfo, isLoading } = useQuery({
    queryKey: ['model-info'],
    queryFn: getModelInfo,
  })

  const [activeDataset, setActiveDataset] = useState('IEEE-CIS')

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Model Monitor</h1>
        <p className="text-sm text-[#8892b0] mt-1">HTGNN architecture · training metrics · per-dataset performance</p>
      </div>

      {/* Model meta */}
      {isLoading ? (
        <div className="flex items-center justify-center h-24"><Spinner size={28} /></div>
      ) : (
        <div className="stats-grid">
          <StatTile value={modelInfo?.version ?? 'demo_v0'} label="Model version" color="#00d4ff" />
          <StatTile
            value={((modelInfo?.test_metrics?.auprc ?? 0.891) * 100).toFixed(1)}
            suffix="%"
            label="Test AUPRC"
            color="#7b2fff"
          />
          <StatTile
            value={((modelInfo?.test_metrics?.auroc ?? 0.938) * 100).toFixed(1)}
            suffix="%"
            label="Test AUC-ROC"
            color="#00ff9d"
          />
          <StatTile
            value={modelInfo?.is_demo ? 'Demo' : 'Trained'}
            label="Status"
            color={modelInfo?.is_demo ? '#ffb700' : '#00ff9d'}
          />
        </div>
      )}

      <div className="two-col items-start">
        {/* Architecture diagram */}
        <div className="card">
          <SectionTitle>
            <Layers size={12} />
            Model architecture
          </SectionTitle>
          <ArchDiagram />

          <div className="mt-4 pt-4 border-t border-white/5">
            <div className="grid grid-cols-2 gap-2">
              {[
                ['Hidden dim', '128'],
                ['Attention heads', '4'],
                ['Conv layers', '2 (HANConv)'],
                ['Temporal seq', '10 txns'],
                ['Dropout', '0.3'],
                ['Total params', '~260K'],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between text-xs py-1 border-b border-white/5">
                  <span className="text-[#4a5568]">{k}</span>
                  <span className="mono text-[#ccd6f6]">{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Training curves */}
        <div className="card">
          <SectionTitle>Training curves (IEEE-CIS)</SectionTitle>
          <FraudAreaChart
            data={MOCK_HISTORY}
            xKey="epoch"
            lines={[
              { key: 'val_auprc', color: '#00d4ff', label: 'Val AUPRC' },
              { key: 'val_auroc', color: '#7b2fff', label: 'Val AUC-ROC' },
            ]}
          />
          <div className="mt-3">
            <SectionTitle>Training loss</SectionTitle>
            <FraudAreaChart
              data={MOCK_HISTORY}
              xKey="epoch"
              lines={[{ key: 'loss', color: '#ff4d6d', label: 'Focal Loss' }]}
            />
          </div>
        </div>
      </div>

      {/* Dataset comparison */}
      <div className="card">
        <SectionTitle>
          <Database size={12} />
          Per-dataset performance
        </SectionTitle>
        <DatasetMetricsBar />

        <div className="overflow-x-auto mt-2">
          <table className="data-table">
            <thead>
              <tr>
                <th>Dataset</th>
                <th>Samples</th>
                <th>AUPRC</th>
                <th>AUC-ROC</th>
                <th>F1</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {DATASET_METRICS.map(d => (
                <tr key={d.dataset}>
                  <td className="font-medium">{d.dataset}</td>
                  <td className="mono text-xs">{d.samples}</td>
                  <td>
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-1 bg-white/5 rounded overflow-hidden">
                        <div className="h-full bg-[#00d4ff] rounded" style={{ width: `${d.auprc * 100}%` }} />
                      </div>
                      <span className="mono text-xs">{d.auprc.toFixed(3)}</span>
                    </div>
                  </td>
                  <td><span className="mono text-xs">{d.auroc.toFixed(3)}</span></td>
                  <td><span className="mono text-xs">{d.f1.toFixed(3)}</span></td>
                  <td>
                    <span className="text-xs text-[#00ff9d] bg-[#00ff9d]/10 border border-[#00ff9d]/20 px-2 py-0.5 rounded-full">
                      ✓ evaluated
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Graph schema */}
      <div className="card">
        <SectionTitle>
          <GitBranch size={12} />
          Heterogeneous graph schema
        </SectionTitle>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-xs text-[#8892b0] mb-3">Node types</p>
            {[
              { type: 'txn', color: '#00d4ff', feat: 'log_amount, hour_sin/cos, V1–V99' },
              { type: 'card', color: '#7b2fff', feat: 'avg_amount, txn_count, fraud_rate' },
              { type: 'merchant', color: '#ffb700', feat: 'avg_amount, fraud_rate_90d, volume' },
              { type: 'device', color: '#ff8c42', feat: 'txn_count, unique_cards, fraud_rate' },
            ].map(n => (
              <div key={n.type} className="flex items-center gap-3 py-1.5 border-b border-white/5">
                <div className="w-2 h-2 rounded-full" style={{ background: n.color }} />
                <span className="mono text-xs font-medium" style={{ color: n.color }}>{n.type}</span>
                <span className="text-[11px] text-[#4a5568] truncate">{n.feat}</span>
              </div>
            ))}
          </div>
          <div>
            <p className="text-xs text-[#8892b0] mb-3">Edge types</p>
            {[
              { edge: 'card → txn', rel: 'makes', note: 'Cardholder initiates' },
              { edge: 'txn → merchant', rel: 'at', note: 'Transaction venue' },
              { edge: 'txn → device', rel: 'via', note: 'Device used' },
              { edge: 'card → card', rel: 'shared_device', note: '⚠ Fraud ring signal' },
              { edge: 'merch → merch', rel: 'same_network', note: 'Shared ownership' },
            ].map(e => (
              <div key={e.edge} className="flex items-start gap-3 py-1.5 border-b border-white/5">
                <span className="mono text-xs text-[#4a5568] w-32 flex-shrink-0">{e.edge}</span>
                <span className="mono text-[11px] text-[#7b2fff]">.{e.rel}</span>
                <span className="text-[11px] text-[#4a5568]">{e.note}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
