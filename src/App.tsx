import { AnimatePresence, motion } from 'framer-motion'
import { useRef, useState } from 'react'
import FaceFrameExhibition from './pages/FaceFrameExhibition'
import FaceFrameHome from './pages/FaceFrameHome'

function App() {
  const [currentRoute, setCurrentRoute] = useState<'home' | 'exhibition'>('home')
  const [eventHorizonPhase, setEventHorizonPhase] = useState<'idle' | 'collapse' | 'sweep'>('idle')
  const timeoutsRef = useRef<number[]>([])
  const ambienceAudioRef = useRef<HTMLAudioElement | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const sourceNodeRef = useRef<MediaElementAudioSourceNode | null>(null)
  const lowPassNodeRef = useRef<BiquadFilterNode | null>(null)
  const audioReadyRef = useRef(false)

  const ensureAmbientAudioReady = async () => {
    const audio = ambienceAudioRef.current
    if (!audio) {
      return
    }

    if (!audioContextRef.current) {
      const context = new window.AudioContext()
      const source = context.createMediaElementSource(audio)
      const filter = context.createBiquadFilter()
      filter.type = 'lowpass'
      filter.frequency.value = 9800
      source.connect(filter)
      filter.connect(context.destination)
      audioContextRef.current = context
      sourceNodeRef.current = source
      lowPassNodeRef.current = filter
    }

    if (audioContextRef.current?.state === 'suspended') {
      await audioContextRef.current.resume()
    }

    if (!audioReadyRef.current) {
      audio.volume = 0.55
      try {
        await audio.play()
        audioReadyRef.current = true
      } catch {
        audioReadyRef.current = false
      }
    }
  }

  const setAudioLowPass = (frequency: number, duration: number) => {
    const context = audioContextRef.current
    const filter = lowPassNodeRef.current
    if (!context || !filter) {
      return
    }
    const now = context.currentTime
    filter.frequency.cancelScheduledValues(now)
    filter.frequency.setValueAtTime(filter.frequency.value, now)
    filter.frequency.exponentialRampToValueAtTime(Math.max(60, frequency), now + duration)
  }

  const startEventHorizonTransition = () => {
    if (eventHorizonPhase !== 'idle') {
      return
    }

    timeoutsRef.current.forEach((timer) => window.clearTimeout(timer))
    timeoutsRef.current = []

    void ensureAmbientAudioReady().then(() => {
      setAudioLowPass(240, 0.18)
    })
    setEventHorizonPhase('sweep')

    timeoutsRef.current.push(
      window.setTimeout(() => {
        setEventHorizonPhase('collapse')
      }, 520),
    )

    timeoutsRef.current.push(
      window.setTimeout(() => {
        setCurrentRoute('exhibition')
      }, 1080),
    )

    timeoutsRef.current.push(
      window.setTimeout(() => {
        setEventHorizonPhase('idle')
        setAudioLowPass(9000, 0.9)
      }, 1780),
    )
  }

  return (
    <div className="min-h-screen w-full overflow-hidden bg-[#050505] text-[#EAEAEA] selection:bg-[#d4af37] selection:text-black">
      <AnimatePresence mode="wait">
        {currentRoute === 'home' ? (
          <motion.div
            key="home"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0, filter: 'blur(10px)' }}
            transition={{ duration: 1.5, ease: 'easeInOut' }}
            className={`h-full w-full ${eventHorizonPhase === 'collapse' ? 'event-horizon-decay' : ''}`}
          >
            <FaceFrameHome onEnterExhibition={startEventHorizonTransition} />
          </motion.div>
        ) : (
          <motion.div
            key="exhibition"
            initial={{ opacity: 0, filter: 'blur(10px)' }}
            animate={{ opacity: 1, filter: 'blur(0px)' }}
            transition={{ duration: 2, ease: 'easeOut', delay: 0.2 }}
            className="h-full w-full"
          >
            <FaceFrameExhibition onBackHome={() => setCurrentRoute('home')} />
          </motion.div>
        )}
      </AnimatePresence>
      {eventHorizonPhase !== 'idle' && (
        <div className="pointer-events-none fixed inset-0 z-[180] overflow-hidden">
          <motion.div
            className="absolute inset-0 bg-[#050505]"
            initial={{ opacity: 0 }}
            animate={{ opacity: eventHorizonPhase === 'sweep' ? 0.9 : 0.58 }}
            transition={{ duration: 0.35, ease: 'easeInOut' }}
          />
          {eventHorizonPhase === 'sweep' && (
            <motion.div
              className="absolute left-1/2 top-0 h-[2px] w-[170vw] -translate-x-1/2 bg-cyan-200 shadow-[0_0_35px_rgba(112,245,245,0.95)]"
              initial={{ y: '-12vh', opacity: 0.55, scaleX: 0.02 }}
              animate={{ y: '112vh', opacity: [1, 1, 0.9], scaleX: [1.35, 1, 0.95] }}
              transition={{ duration: 0.72, ease: [0.25, 0.7, 0.2, 1] }}
            />
          )}
          {eventHorizonPhase === 'collapse' && (
            <div className="absolute inset-0">
              {[
                { id: 'g1', left: '18%', top: '26%' },
                { id: 'g2', left: '75%', top: '30%' },
                { id: 'g3', left: '30%', top: '68%' },
                { id: 'g4', left: '72%', top: '72%' },
              ].map((glyph) => (
                <motion.div
                  key={glyph.id}
                  className="absolute"
                  style={{ left: glyph.left, top: glyph.top }}
                  initial={{ x: 0, y: 0, scale: 1, opacity: 0.96 }}
                  animate={{ left: '50%', top: '50%', x: '-50%', y: '-50%', scale: 0.22, opacity: 0 }}
                  transition={{ duration: 0.56, ease: [0.2, 0.82, 0.15, 1] }}
                >
                  <span className="block h-8 w-8 rounded-full border border-[#6ad5d2]/65 shadow-[0_0_28px_rgba(109,224,214,0.55)]" />
                </motion.div>
              ))}
              <motion.div
                className="absolute left-1/2 top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-[#b7f4ee] shadow-[0_0_55px_rgba(136,247,238,1),0_0_110px_rgba(212,175,55,0.78)]"
                initial={{ scale: 0 }}
                animate={{ scale: [0, 1.7, 0.72, 1.35, 0] }}
                transition={{ duration: 0.62, times: [0, 0.25, 0.5, 0.72, 1], ease: 'easeOut' }}
              />
              <motion.div
                className="absolute left-1/2 top-1/2 h-[1px] w-[170vw] -translate-x-1/2 -translate-y-1/2 bg-cyan-100/95 shadow-[0_0_25px_rgba(129,240,236,0.92)]"
                initial={{ scaleX: 0, opacity: 0 }}
                animate={{ scaleX: [0, 0.16, 1.4], opacity: [0, 1, 0] }}
                transition={{ duration: 0.48, times: [0, 0.35, 1], ease: 'easeInOut' }}
              />
            </div>
          )}
        </div>
      )}
      <audio ref={ambienceAudioRef} loop preload="auto" src="/audio/ambient-vocal-free-instrumental.mp3" className="hidden" />
    </div>
  )
}

export default App
