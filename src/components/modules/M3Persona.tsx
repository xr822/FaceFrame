import { motion } from 'framer-motion'

export function M3Persona() {
  return (
    <motion.section
      initial={{ opacity: 0, y: 18 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-10%' }}
      transition={{ duration: 0.8, ease: 'easeOut' }}
      className="rounded-3xl border border-violet-100/20 bg-violet-100/5 p-8 backdrop-blur-2xl"
    >
      <p className="mb-3 text-xs uppercase tracking-[0.28em] text-violet-50/70">M3</p>
      <h2 className="text-2xl font-light text-violet-50">The Persona</h2>
      <p className="mt-3 text-sm text-violet-50/60">琉璃矩阵 + 夺权滑动条占位</p>
    </motion.section>
  )
}
