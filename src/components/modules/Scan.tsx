import { motion } from 'framer-motion'
import { Camera, ScanLine } from 'lucide-react'

export function Scan() {
  return (
    <motion.section
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.9, ease: 'easeOut' }}
      className="relative overflow-hidden rounded-3xl border border-white/15 bg-white/5 p-8 shadow-[0_0_80px_rgba(140,112,255,0.18)] backdrop-blur-2xl"
    >
      <div className="pointer-events-none absolute inset-0 rounded-3xl bg-gradient-to-b from-transparent via-cyan-300/10 to-transparent mix-blend-screen" />
      <div className="pointer-events-none absolute left-0 right-0 top-1/2 h-px -translate-y-1/2 bg-cyan-200/70 shadow-[0_0_20px_rgba(111,227,255,0.9)]" />
      <div className="mb-6 flex items-center gap-3 text-amber-200">
        <Camera className="h-5 w-5" />
        <p className="text-sm uppercase tracking-[0.28em]">The Scan</p>
      </div>
      <div className="rounded-2xl border border-white/20 bg-black/25 p-6">
        <div className="mb-4 flex items-center gap-2 text-white/70">
          <ScanLine className="h-4 w-4" />
          <span className="text-xs uppercase tracking-[0.2em]">Entry Portal</span>
        </div>
        <p className="text-sm text-white/65">上传面部照片 + 输入当下心情</p>
      </div>
    </motion.section>
  )
}
