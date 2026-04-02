import { motion } from 'framer-motion'

export function M4Destiny() {
  return (
    <motion.section
      initial={{ opacity: 0, y: 18 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-10%' }}
      transition={{ duration: 0.8, ease: 'easeOut' }}
      className="rounded-3xl border border-orange-100/20 bg-orange-100/5 p-8 backdrop-blur-2xl"
    >
      <p className="mb-3 text-xs uppercase tracking-[0.28em] text-orange-50/70">M4</p>
      <h2 className="text-2xl font-light text-orange-50">The Destiny</h2>
      <p className="mt-3 text-sm text-orange-50/60">等离子日食 + 分享卡片生成占位</p>
    </motion.section>
  )
}
