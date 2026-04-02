import { Sigma } from 'lucide-react'
import { AmbientPlayer } from './AmbientPlayer'

export function TopRightNav() {
  return (
    <div className="fixed right-6 top-6 z-50 flex items-center gap-3">
      <AmbientPlayer />
      <button
        type="button"
        className="flex h-10 w-10 items-center justify-center rounded-full border border-white/20 bg-black/35 text-white backdrop-blur-xl transition hover:bg-white/10"
        aria-label="打开算法决策路径"
      >
        <Sigma className="h-4 w-4" />
      </button>
      <button
        type="button"
        className="rounded-full border border-white/20 bg-black/35 px-3 py-2 text-xs tracking-widest text-white backdrop-blur-xl transition hover:bg-white/10"
        aria-label="切换语言"
      >
        中 / EN
      </button>
    </div>
  )
}
