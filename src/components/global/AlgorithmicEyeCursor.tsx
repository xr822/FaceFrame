import { useEffect, useRef, useState } from 'react'

type GazeMode = 'idle' | 'analyzing'

type Props = {
  mode: GazeMode
}

export function AlgorithmicEyeCursor({ mode }: Props) {
  const rafRef = useRef<number | null>(null)
  const targetRef = useRef({ x: 0, y: 0 })
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const [isBlinking, setIsBlinking] = useState(false)

  useEffect(() => {
    const handleMove = (event: MouseEvent) => {
      targetRef.current = { x: event.clientX, y: event.clientY }
    }

    window.addEventListener('mousemove', handleMove)

    const smoothFollow = () => {
      setPosition((current) => {
        const lerp = 0.16
        const dx = targetRef.current.x - current.x
        const dy = targetRef.current.y - current.y
        return { x: current.x + dx * lerp, y: current.y + dy * lerp }
      })
      rafRef.current = window.requestAnimationFrame(smoothFollow)
    }

    rafRef.current = window.requestAnimationFrame(smoothFollow)

    return () => {
      window.removeEventListener('mousemove', handleMove)
      if (rafRef.current !== null) {
        window.cancelAnimationFrame(rafRef.current)
      }
    }
  }, [])

  useEffect(() => {
    let timeoutId: number | null = null

    const scheduleBlink = () => {
      const delay = 2800 + Math.random() * 2600
      timeoutId = window.setTimeout(() => {
        setIsBlinking(true)
        window.setTimeout(() => {
          setIsBlinking(false)
          scheduleBlink()
        }, 140)
      }, delay)
    }

    scheduleBlink()

    return () => {
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId)
      }
    }
  }, [])

  const isHot = mode === 'analyzing'

  const eyeSize = isHot ? 40 : 32
  const offset = eyeSize / 2

  return (
    <div
      className="pointer-events-none fixed left-0 top-0 z-[120]"
      style={{
        transform: `translate3d(${position.x - offset}px, ${position.y - offset}px, 0)`,
      }}
    >
      <div
        className={`relative flex items-center justify-center transition-all duration-300 ${
          isHot ? 'scale-110' : 'scale-100'
        }`}
        style={{ width: eyeSize, height: eyeSize }}
      >
        <div
          className={`absolute inset-0 rounded-full transition-all duration-300 ${
            isHot ? 'bg-[#f6c868]/25 shadow-[0_0_24px_rgba(245,201,120,0.6)]' : 'bg-[#11a8c4]/18 shadow-[0_0_18px_rgba(48,193,220,0.4)]'
          }`}
        />
        <div
          className={`absolute inset-[18%] rounded-full border-[1px] transition-all duration-300 ${
            isHot ? 'border-[#f6c868]' : 'border-[#7ee0ff]'
          }`}
        />
        <svg
          viewBox="0 0 40 24"
          className="relative h-4 w-7"
          style={{
            opacity: isBlinking ? 0.06 : 1,
            transform: isBlinking ? 'scaleY(0.12)' : 'scaleY(1)',
            transformOrigin: '50% 50%',
            transition: 'transform 120ms ease-out, opacity 120ms ease-out',
          }}
        >
          <path
            d="M2 12C6 4.5 12 2 20 2s14 2.5 18 10c-4 7.5-10 10-18 10S6 19.5 2 12Z"
            fill="none"
            stroke={isHot ? '#f6c868' : '#7ee0ff'}
            strokeWidth={0.7}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <circle
            cx="20"
            cy="12"
            r={isHot ? 3.4 : 2.6}
            fill={isHot ? '#f6c868' : '#7ee0ff'}
            opacity={0.95}
          />
          <circle cx="19" cy="11" r={1.1} fill="#111827" opacity={0.95} />
        </svg>
      </div>
    </div>
  )
}

