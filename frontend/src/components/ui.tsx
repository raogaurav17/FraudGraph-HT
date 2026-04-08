import React from 'react'
import { clsx } from 'clsx'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, Legend, PieChart, Pie, Cell
} from 'recharts'

// ── Risk Badge ────────────────────────────────────────────────────
export function RiskBadge({ level }: { level: string }) {
  const cls = {
    LOW: 'badge-low',
    MEDIUM: 'badge-medium',
    HIGH: 'badge-high',
    CRITICAL: 'badge-critical',
  }[level] ?? 'badge-low'

  return (
    <span className={`${cls} text-[11px] font-mono font-semibold px-2 py-0.5 rounded-full`}>
      {level}
    </span>
  )
}

// ── Probability Bar ───────────────────────────────────────────────
export function ProbBar({ value, size = 'md' }: { value: number; size?: 'sm' | 'md' }) {
  const color =
    value >= 0.8 ? '#ff4d6d' :
    value >= 0.5 ? '#ff8c42' :
    value >= 0.25 ? '#ffb700' : '#00ff9d'

  return (
    <div className="flex items-center gap-2">
      <div className={`prob-bar flex-1 ${size === 'sm' ? 'h-1' : 'h-1.5'}`}>
        <div
          className="prob-bar-fill"
          style={{ width: `${value * 100}%`, background: color }}
        />
      </div>
      <span className="mono text-xs text-[#8892b0] w-10 text-right">
        {(value * 100).toFixed(1)}%
      </span>
    </div>
  )
}

// ── Stat Tile ─────────────────────────────────────────────────────
export function StatTile({
  value, label, color = '#00d4ff', prefix = '', suffix = '', delta
}: {
  value: string | number
  label: string
  color?: string
  prefix?: string
  suffix?: string
  delta?: { value: string; positive: boolean }
}) {
  return (
    <div className="stat-tile">
      <div className="value" style={{ color }}>
        {prefix}{value}{suffix}
      </div>
      <div className="label">{label}</div>
      {delta && (
        <div className={`text-xs mono mt-1 ${delta.positive ? 'text-[#00ff9d]' : 'text-[#ff4d6d]'}`}>
          {delta.positive ? '▲' : '▼'} {delta.value}
        </div>
      )}
    </div>
  )
}

// ── Section title ─────────────────────────────────────────────────
export function SectionTitle({ children }: { children: React.ReactNode }) {
  return <div className="section-title">{children}</div>
}

// ── Empty state ───────────────────────────────────────────────────
export function EmptyState({ message = 'No data yet' }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-[#4a5568]">
      <div className="text-4xl mb-3 opacity-30">⬡</div>
      <p className="text-sm">{message}</p>
    </div>
  )
}

// ── Loading spinner ───────────────────────────────────────────────
export function Spinner({ size = 20 }: { size?: number }) {
  return (
    <div
      className="rounded-full border-2 border-white/10 border-t-[#00d4ff] animate-spin"
      style={{ width: size, height: size }}
    />
  )
}

// ── Area Chart wrapper ────────────────────────────────────────────
export function FraudAreaChart({
  data, xKey = 'hour', lines
}: {
  data: any[]
  xKey?: string
  lines: { key: string; color: string; label?: string }[]
}) {
  return (
    <ResponsiveContainer width="100%" height={200}>
      <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
        <defs>
          {lines.map(l => (
            <linearGradient key={l.key} id={`grad-${l.key}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={l.color} stopOpacity={0.3} />
              <stop offset="95%" stopColor={l.color} stopOpacity={0} />
            </linearGradient>
          ))}
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
        <XAxis dataKey={xKey} stroke="#4a5568" tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }} />
        <YAxis stroke="#4a5568" tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }} />
        <Tooltip
          contentStyle={{
            background: '#0f0f28', border: '1px solid rgba(0,212,255,0.2)',
            borderRadius: 8, fontFamily: 'IBM Plex Mono', fontSize: 12
          }}
          labelStyle={{ color: '#ccd6f6' }}
          itemStyle={{ color: '#8892b0' }}
        />
        {lines.map(l => (
          <Area
            key={l.key}
            type="monotone"
            dataKey={l.key}
            stroke={l.color}
            strokeWidth={1.5}
            fill={`url(#grad-${l.key})`}
            name={l.label || l.key}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  )
}

// ── Donut / Pie chart ─────────────────────────────────────────────
export function RiskDonut({ data }: { data: { name: string; value: number }[] }) {
  const COLORS: Record<string, string> = {
    LOW: '#00ff9d', MEDIUM: '#ffb700', HIGH: '#ff8c42', CRITICAL: '#ff4d6d'
  }
  return (
    <ResponsiveContainer width="100%" height={180}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={50}
          outerRadius={75}
          strokeWidth={0}
          dataKey="value"
        >
          {data.map((entry, i) => (
            <Cell key={i} fill={COLORS[entry.name] ?? '#8892b0'} />
          ))}
        </Pie>
        <Tooltip
          contentStyle={{
            background: '#0f0f28', border: '1px solid rgba(0,212,255,0.2)',
            borderRadius: 8, fontFamily: 'IBM Plex Mono', fontSize: 12
          }}
        />
      </PieChart>
    </ResponsiveContainer>
  )
}

// ── Feature importance bar ────────────────────────────────────────
export function FeatureBar({ name, value, contribution }: {
  name: string; value: number; contribution: number
}) {
  return (
    <div className="flex items-center gap-3 py-1">
      <span className="mono text-xs text-[#8892b0] w-32 truncate">{name}</span>
      <div className="flex-1 h-1 bg-white/5 rounded overflow-hidden">
        <div
          className="h-full rounded"
          style={{
            width: `${Math.min(contribution * 100, 100)}%`,
            background: 'linear-gradient(90deg, #7b2fff, #00d4ff)'
          }}
        />
      </div>
      <span className="mono text-[11px] text-[#4a5568] w-14 text-right">
        {value.toFixed(3)}
      </span>
    </div>
  )
}
