import { Pause, Play, Volume2 } from 'lucide-react'
import { useRef, useState } from 'react'

const spectrumBars = [10, 16, 13, 18, 9]

export function AmbientPlayer() {
  const [isPlaying, setIsPlaying] = useState(false)
  const audioRef = useRef<HTMLAudioElement>(null)

  const handleToggle = async () => {
    const audio = audioRef.current
    if (!audio) {
      return
    }

    if (isPlaying) {
      audio.pause()
      setIsPlaying(false)
      return
    }

    try {
      await audio.play()
      setIsPlaying(true)
    } catch {
      setIsPlaying(false)
    }
  }

  return (
    <div className="flex items-center gap-3 rounded-full border border-white/20 bg-black/35 px-4 py-2 backdrop-blur-xl">
      <audio ref={audioRef} loop preload="none" src="/audio/ambient-vocal-free-instrumental.mp3" />
      <Volume2 className="h-4 w-4 text-cyan-100/80" />
      <button
        type="button"
        onClick={handleToggle}
        className="flex h-8 w-8 items-center justify-center rounded-full border border-white/20 bg-white/10 text-white transition hover:bg-white/20"
        aria-label={isPlaying ? '暂停无人声纯器乐' : '播放无人声纯器乐'}
      >
        {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
      </button>
      <div className="flex items-end gap-1">
        {spectrumBars.map((height, index) => (
          <span
            key={index}
            style={{ height }}
            className={`w-1 rounded-full bg-cyan-200/80 transition ${
              isPlaying ? 'animate-pulse' : 'opacity-50'
            }`}
          />
        ))}
      </div>
    </div>
  )
}
