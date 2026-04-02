import { motion } from 'framer-motion'

export function M1Geometry() {
  return (
    <motion.section
      initial={{ opacity: 0, y: 18 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-10%' }}
      transition={{ duration: 0.8, ease: 'easeOut' }}
      className="rounded-3xl border border-amber-100/20 bg-amber-50/5 p-8 backdrop-blur-2xl"
    >
      <p className="mb-3 text-xs uppercase tracking-[0.28em] text-amber-100/70">M1</p>
      <h2 className="text-2xl font-light text-amber-50">The Geometry</h2>
      <p className="mt-3 text-sm text-amber-50/60">68 点拓扑星盘 + 三庭五眼 + SVM 分类占位</p>
    </motion.section>
  )
}
