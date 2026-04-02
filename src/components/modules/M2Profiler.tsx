import { motion } from 'framer-motion'

export function M2Profiler() {
  return (
    <motion.section
      initial={{ opacity: 0, y: 18 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-10%' }}
      transition={{ duration: 0.8, ease: 'easeOut' }}
      className="relative overflow-hidden rounded-3xl border border-sky-100/20 bg-sky-100/5 p-8 backdrop-blur-2xl"
    >
      <div className="pointer-events-none absolute -left-12 top-8 h-28 w-28 rounded-full bg-fuchsia-400/25 blur-3xl mix-blend-screen" />
      <div className="pointer-events-none absolute bottom-6 right-0 h-36 w-36 rounded-full bg-cyan-300/30 blur-3xl mix-blend-screen" />
      <p className="mb-3 text-xs uppercase tracking-[0.28em] text-sky-50/70">M2</p>
      <h2 className="text-2xl font-light text-sky-50">The Profiler</h2>
      <p className="mt-3 text-sm text-sky-50/60">情绪极性提取与语义碰撞占位</p>
    </motion.section>
  )
}
