import React, { useEffect } from 'react'
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Activity, BarChart3, Search, Cpu, Zap, Shield } from 'lucide-react'
import { useAppStore } from './store'
import { createFraudScoreSocket } from './utils/api'
import Dashboard from './pages/Dashboard'
import Explorer from './pages/Explorer'
import Predict from './pages/Predict'
import Monitor from './pages/Monitor'
import './index.css'

const qc = new QueryClient({
  defaultOptions: { queries: { retry: 1, staleTime: 30_000 } },
})

function NavItem({ to, icon: Icon, label }: { to: string; icon: any; label: string }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `flex items-center gap-2.5 px-3 py-2 rounded-md text-sm font-medium transition-all
         ${isActive
           ? 'bg-[#1a1a2e] text-[#00d4ff] border border-[#00d4ff]/20'
           : 'text-[#8892b0] hover:text-[#ccd6f6] hover:bg-white/5'}`
      }
    >
      <Icon size={16} strokeWidth={1.5} />
      {label}
    </NavLink>
  )
}

function App() {
  const { addLiveScore, setWsConnected } = useAppStore()

  useEffect(() => {
    let ws: WebSocket | null = null
    let retryTimeout: any

    const connect = () => {
      try {
        ws = createFraudScoreSocket(
          (data) => {
            if (data?.type === 'new_prediction' && data?.data) {
              addLiveScore(data.data)
            }
          },
          () => {
            setWsConnected(false)
            retryTimeout = setTimeout(connect, 3000)
          }
        )
        ws.onopen = () => setWsConnected(true)
        ws.onclose = () => {
          setWsConnected(false)
          retryTimeout = setTimeout(connect, 3000)
        }
      } catch {}
    }

    connect()
    return () => {
      clearTimeout(retryTimeout)
      ws?.close()
    }
  }, [])

  const { wsConnected } = useAppStore()

  return (
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <div className="min-h-screen bg-[#050510] text-[#ccd6f6] flex flex-col">

          {/* Top nav */}
          <header className="border-b border-white/5 bg-[#080820]/80 backdrop-blur-sm sticky top-0 z-50">
            <div className="max-w-screen-xl mx-auto px-6 h-14 flex items-center justify-between">
              {/* Logo */}
              <div className="flex items-center gap-2.5">
                <div className="w-7 h-7 rounded bg-gradient-to-br from-[#00d4ff] to-[#7b2fff] flex items-center justify-center">
                  <Shield size={14} className="text-white" />
                </div>
                <span className="font-bold text-white tracking-tight text-base font-mono">
                  Fraud<span className="text-[#00d4ff]">Graph</span>
                </span>
                <span className="text-[10px] text-[#8892b0] border border-white/10 rounded px-1.5 py-0.5 ml-1">
                  HTGNN v1.0
                </span>
              </div>

              {/* Nav links */}
              <nav className="flex items-center gap-1">
                <NavItem to="/" icon={BarChart3} label="Dashboard" />
                <NavItem to="/predict" icon={Zap} label="Predict" />
                <NavItem to="/explorer" icon={Search} label="Explorer" />
                <NavItem to="/monitor" icon={Cpu} label="Model" />
              </nav>

              {/* WS status */}
              <div className="flex items-center gap-2 text-xs text-[#8892b0]">
                <div className={`w-1.5 h-1.5 rounded-full ${wsConnected ? 'bg-[#00ff9d] shadow-[0_0_6px_#00ff9d]' : 'bg-[#ff4d6d]'}`} />
                {wsConnected ? 'Live' : 'Offline'}
              </div>
            </div>
          </header>

          {/* Main content */}
          <main className="flex-1 max-w-screen-xl mx-auto w-full px-6 py-8">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/predict" element={<Predict />} />
              <Route path="/explorer" element={<Explorer />} />
              <Route path="/monitor" element={<Monitor />} />
            </Routes>
          </main>

          {/* Footer */}
          <footer className="border-t border-white/5 py-4 text-center text-xs text-[#4a5568]">
            FraudGraph · HTGNN Credit Card Fraud Detection Platform · PyTorch Geometric
          </footer>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  )
}

export default App
