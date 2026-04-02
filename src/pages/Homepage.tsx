import { useEffect, useRef, useState } from 'react'
import { Globe, X } from 'lucide-react'
import * as THREE from 'three'
import { AlgorithmicEyeCursor } from '../components/global/AlgorithmicEyeCursor'
import { useLanguage } from '../i18n-context'

type ModuleCard = {
  id: 'M1' | 'M2' | 'M3' | 'M4'
  title: string
  sub: string
  desc: string
}

type HomepageProps = {
  onEnterExhibition?: () => void
}

function ModuleGlyph({ moduleId }: { moduleId: ModuleCard['id'] }) {
  if (moduleId === 'M1') {
    return (
      <div className="relative h-28 w-28 transition-transform duration-700 md:h-32 md:w-32 group-hover:scale-[1.06]">
        <div className="absolute inset-0 rounded-full border border-white/20 transition-all duration-700 group-hover:border-[#87CEFA]/70 group-hover:shadow-[0_0_24px_rgba(135,206,250,0.35)]" />
        <div className="absolute inset-3 rounded-full border border-[#87CEFA]/25 opacity-0 transition-opacity duration-700 group-hover:opacity-100" />
        <div className="absolute inset-7 rounded-full border border-dashed border-[#d4af37]/40 transition-all duration-700 group-hover:animate-[spin_4s_linear_infinite] group-hover:border-[#87CEFA]/60" />
        <div className="absolute left-1/2 top-1/2 h-px w-32 -translate-x-1/2 -translate-y-1/2 rotate-45 bg-white/20 transition-transform duration-700 md:w-36 group-hover:rotate-[135deg]" />
        <div className="absolute left-1/2 top-1/2 h-32 w-px -translate-x-1/2 -translate-y-1/2 rotate-45 bg-white/20 transition-transform duration-700 md:h-36 group-hover:rotate-[135deg]" />
        <div className="absolute left-1/2 top-1/2 h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white shadow-[0_0_10px_#fff] transition-all duration-700 group-hover:scale-150 group-hover:bg-[#87CEFA] group-hover:shadow-[0_0_20px_rgba(135,206,250,0.9)]" />
      </div>
    )
  }

  if (moduleId === 'M2') {
    return (
      <div className="relative h-32 w-32 rounded-full bg-[radial-gradient(circle_at_30%_30%,rgba(135,206,235,0.8),rgba(65,105,225,0.2))] blur-[15px] transition-all duration-700 group-hover:scale-[1.2] group-hover:bg-[radial-gradient(circle_at_70%_70%,rgba(255,215,0,0.6),rgba(135,206,235,0.2))] group-hover:blur-[20px]" />
    )
  }

  if (moduleId === 'M3') {
    return (
      <div className="group/icon relative h-24 w-24 [transform-style:preserve-3d] transition-all duration-700 [transform:rotateX(60deg)_rotateZ(45deg)] group-hover:[transform:rotateX(60deg)_rotateZ(90deg)]">
        <div className="absolute inset-0 border border-[#ffd70080] bg-[#ffd7000d] backdrop-blur-sm transition-all duration-700 [transform:translateZ(0px)] group-hover:[transform:translateZ(20px)] group-hover:bg-[#ffd70026]" />
        <div className="absolute inset-0 scale-80 border border-white/20 bg-white/5 transition-all duration-700 [transform:translateZ(-20px)]" />
      </div>
    )
  }

  return (
    <div className="relative h-[90px] w-[90px]">
      <div className="absolute inset-0 rounded-full bg-black shadow-[0_0_30px_10px_rgba(0,255,200,0.3),inset_0_0_10px_rgba(0,0,0,1)] transition-all duration-700 group-hover:scale-105 group-hover:shadow-[0_0_50px_15px_rgba(0,255,200,0.5),inset_0_0_10px_rgba(0,0,0,1)]" />
      <div className="absolute -inset-2 rounded-full bg-[#00ffc8]/10 blur-xl transition-all duration-700 group-hover:scale-110 group-hover:bg-[#00ffc8]/18" />
    </div>
  )
}

function useHomepageParticles(containerRef: React.RefObject<HTMLDivElement | null>) {
  useEffect(() => {
    const container = containerRef.current
    if (!container) {
      return
    }

    const scene = new THREE.Scene()
    scene.fog = new THREE.FogExp2(0x030406, 0.002)
    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 1, 1000)
    camera.position.z = 200

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setSize(window.innerWidth, window.innerHeight)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.domElement.style.pointerEvents = 'none'
    container.appendChild(renderer.domElement)

    const particleCount = 1500
    const geometry = new THREE.BufferGeometry()
    const positions = new Float32Array(particleCount * 3)
    for (let i = 0; i < particleCount; i += 1) {
      positions[i * 3] = (Math.random() - 0.5) * 800
      positions[i * 3 + 1] = (Math.random() - 0.5) * 800
      positions[i * 3 + 2] = (Math.random() - 0.5) * 600
    }
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    const material = new THREE.PointsMaterial({
      color: 0x87cefa,
      size: 1.5,
      transparent: true,
      opacity: 0.6,
      blending: THREE.AdditiveBlending,
    })
    const particles = new THREE.Points(geometry, material)
    scene.add(particles)

    let mouseX = 0
    let mouseY = 0
    const handleMouseMove = (event: MouseEvent) => {
      mouseX = event.clientX - window.innerWidth / 2
      mouseY = event.clientY - window.innerHeight / 2
    }
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight
      camera.updateProjectionMatrix()
      renderer.setSize(window.innerWidth, window.innerHeight)
    }

    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('resize', handleResize)

    const clock = new THREE.Clock()
    let raf = 0
    const animate = () => {
      const time = clock.getElapsedTime()
      particles.rotation.y = time * 0.02
      particles.rotation.z = time * 0.01

      const targetX = mouseX * 0.05
      const targetY = mouseY * 0.05
      camera.position.x += (targetX - camera.position.x) * 0.02
      camera.position.y += (-targetY - camera.position.y) * 0.02
      camera.lookAt(scene.position)
      renderer.render(scene, camera)
      raf = window.requestAnimationFrame(animate)
    }

    raf = window.requestAnimationFrame(animate)

    return () => {
      window.cancelAnimationFrame(raf)
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('resize', handleResize)
      geometry.dispose()
      material.dispose()
      renderer.dispose()
      if (renderer.domElement.parentElement === container) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [containerRef])
}

export function Homepage({ onEnterExhibition }: HomepageProps) {
  const { lang, toggle } = useLanguage()
  const [engineOpen, setEngineOpen] = useState(false)
  const [gazeMode, setGazeMode] = useState<'idle' | 'analyzing'>('idle')
  const bgCanvasRef = useRef<HTMLDivElement | null>(null)
  useHomepageParticles(bgCanvasRef)

  const t = {
    scan: lang === 'cn' ? '开始凝视 (Begin Scan)' : 'Begin The Scan',
    heroEyebrow: 'A Digital Art Experiment',
    heroIntro:
      lang === 'cn'
        ? '欢迎来到赛博面相与自由意志的交互场域。\n该装置内容涉及生物特征降维与算法审视，建议在绝对专注的环境中开启。'
        : 'Welcome to the interactive field of cyber physiognomy and free will.\nThis installation involves biometric abstraction and algorithmic inspection. Enter in full focus.',
    modulesTitle: lang === 'cn' ? '模块矩阵' : 'The Modules.',
    modulesTagline: lang === 'cn' ? '一面之缘 / 一键共鸣 / 一生宿命' : 'One Face / One Click / One Destiny',
    footerPowered: lang === 'cn' ? '由 CV & NLP & WebGL 驱动' : 'Powered by CV & NLP & WebGL',
    modules: [
      {
        id: 'M1',
        title: lang === 'cn' ? '几何拓扑' : 'The Geometry',
        sub: lang === 'cn' ? '解剖蓝图' : 'Anatomical Blueprint',
        desc: lang === 'cn' ? '交出面容。算法将暴力拆解五官降维为冰冷的物理坐标，这是先天宿命的数字化宣判。' : 'Surrender your facial structure to topological extraction.',
      },
      {
        id: 'M2',
        title: lang === 'cn' ? '心境流墨' : 'The Profiler',
        sub: lang === 'cn' ? '语义渗流' : 'Semantic Bleed',
        desc: lang === 'cn' ? '语义暴露潜意识。算法捕捉混沌心境化为流体，与静态骨相形成形与神的博弈。' : 'Semantic flow reveals subconscious emotional turbulence.',
      },
      {
        id: 'M3',
        title: lang === 'cn' ? '琉璃重构' : 'The Persona',
        sub: lang === 'cn' ? '琉璃变形' : 'Glass Metamorphosis',
        desc: lang === 'cn' ? '夺回数字身份权。擦除机器偏见滤镜，用人类的自由意志重新定义灵魂图腾。' : 'Reconstruct identity with glass-like algorithmic agency.',
      },
      {
        id: 'M4',
        title: lang === 'cn' ? '宿命日食' : 'The Destiny',
        sub: lang === 'cn' ? '关系日食' : 'Relational Eclipse',
        desc: lang === 'cn' ? '主宰坍缩。骨相、心境与意志在此融合成专属数字星球，实现赛博时代的永生。' : 'Collapse physiognomy and will into one digital destiny.',
      },
    ] satisfies ModuleCard[],
  }

  return (
    <div className="min-h-screen bg-[#111112] font-sans text-[#eaeaea] selection:bg-[#d4af37] selection:text-black">
      <AlgorithmicEyeCursor mode={gazeMode} />
      <div ref={bgCanvasRef} className="pointer-events-none fixed inset-0 z-0" />
      <div className="pointer-events-none fixed inset-0 z-0 flex items-end justify-center">
        <div className="ambient-amber-breath h-[58vh] w-[72vw] rounded-full bg-[#9b6f2a]/16 blur-[140px]" />
      </div>
      <div
        className="pointer-events-none fixed inset-0 z-0 opacity-[0.03]"
        style={{ backgroundImage: 'url("https://www.transparenttextures.com/patterns/stardust.png")' }}
      />
      <header className="fixed top-0 z-50 flex w-full items-center justify-end p-4 mix-blend-difference md:p-6 lg:px-12">
        <div className="flex items-center gap-6">
          <button
            type="button"
            onClick={toggle}
            className="flex items-center gap-1 px-1 text-xs tracking-widest text-[#888] uppercase transition-all duration-500 hover:text-[#d4af37] hover:drop-shadow-[0_0_10px_rgba(212,175,55,0.36)]"
          >
            <Globe size={14} />
            {lang === 'cn' ? 'EN' : '中'}
          </button>
          <button
            type="button"
            onClick={() => setEngineOpen(true)}
            className="text-[#888] transition-colors hover:text-[#d4af37]"
            title={lang === 'cn' ? '引擎：算法透明' : 'The Engine: Algorithm Transparency'}
          >
            <span className="pr-1 text-lg italic">∑</span>
          </button>
        </div>
      </header>

      <main className="relative z-10 mx-auto max-w-[1600px] px-4 pb-8 pt-0 sm:px-6 lg:px-12">
        <section className="relative flex min-h-screen w-full flex-col items-center justify-center pt-16 md:pt-20">
          <div className="w-full overflow-hidden px-4 text-center">
            <p className="mb-6 text-xs tracking-[0.4em] text-[#87CEFA] uppercase md:text-sm">{t.heroEyebrow}</p>
            <h1 className="font-serif-title mb-4 text-6xl leading-[0.9] font-bold tracking-tighter sm:text-7xl md:text-[188px]">
              FACE
              <br />
              FRAME
            </h1>
            <p className="font-serif-title mx-auto mb-12 mt-8 max-w-xl whitespace-pre-line text-sm leading-relaxed tracking-widest text-gray-400 md:text-base">{t.heroIntro}</p>
            <button
              type="button"
              onClick={onEnterExhibition}
              className="inline-flex items-center justify-center rounded-full border border-white/20 bg-white/5 px-10 py-4 text-sm tracking-[0.08em] text-white uppercase backdrop-blur-md transition-all duration-500 hover:scale-105 hover:bg-white hover:text-black"
              onMouseEnter={() => setGazeMode('analyzing')}
              onMouseLeave={() => setGazeMode('idle')}
            >
              {t.scan} <span className="ml-2">↓</span>
            </button>
          </div>
        </section>

        <section className="relative z-10 min-h-screen w-full px-0 py-14 md:px-6 md:py-24">
          <div className="mx-auto max-w-7xl">
            <div className="mb-10 flex items-end justify-between border-b border-gray-800 pb-6 md:mb-20 md:pb-8">
              <h2 className="font-serif-title text-3xl tracking-[0.1em] md:text-5xl">{t.modulesTitle}</h2>
              <p className="text-[10px] tracking-[0.2em] text-[#87CEFA] uppercase md:text-xs md:tracking-[0.3em]">{t.modulesTagline}</p>
            </div>
            <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
              {t.modules.map((mod) => (
                <article
                  key={mod.id}
                  className="group relative flex min-h-[390px] flex-col justify-between overflow-hidden rounded-3xl border border-white/5 bg-white/[0.02] p-6 transition-all duration-500 hover:-translate-y-2.5 hover:border-white/20 hover:bg-white/[0.04] hover:shadow-[0_20px_40px_rgba(0,0,0,0.5)] md:min-h-[460px] md:p-8 lg:min-h-[480px] lg:p-10"
                  onMouseEnter={() => setGazeMode('analyzing')}
                  onMouseLeave={() => setGazeMode('idle')}
                >
                  <span className="absolute left-8 top-8 font-mono text-[10px] tracking-[0.2em] text-gray-500">[ {mod.id} ]</span>
                  <div className="mb-4 flex h-[140px] items-center justify-center md:mb-5 md:h-[180px]">
                    <ModuleGlyph moduleId={mod.id} />
                  </div>
                  <div>
                    <h3 className="font-serif-title mb-1 text-xl text-white transition-colors group-hover:text-[#87CEFA] md:text-2xl">{mod.title}</h3>
                    <p className="mb-4 font-mono text-[10px] tracking-widest text-gray-500 uppercase">{mod.sub}</p>
                    <p className="font-serif-title text-xs leading-relaxed text-gray-400">{mod.desc}</p>
                  </div>
                </article>
              ))}
            </div>
          </div>
        </section>

        <footer className="relative z-10 flex w-full flex-col items-center border-t border-gray-900 py-12 text-center">
          <h2 className="font-serif-title mb-4 text-3xl">FaceFrame.</h2>
          <p className="text-[10px] tracking-[0.2em] text-gray-600 uppercase">{t.footerPowered}</p>
        </footer>
      </main>

      {engineOpen && (
        <div className="fixed inset-0 z-[100] flex flex-col items-center justify-center bg-[#0d0d0e]/95 p-6 backdrop-blur-md animate-in fade-in duration-500">
          <button
            type="button"
            onClick={() => setEngineOpen(false)}
            className="absolute top-8 right-8 text-[#888] transition-colors hover:text-white lg:right-12"
          >
            <X size={32} strokeWidth={1} />
          </button>
          <div className="max-w-2xl text-center">
            <span className="mb-6 block text-4xl text-[#d4af37] italic">∑</span>
            <h2 className="mb-6 text-3xl lg:text-5xl">{lang === 'cn' ? '引擎' : 'The Engine'}</h2>
            <p className="mb-12 leading-relaxed font-light tracking-wide text-[#888]">
              {lang === 'cn'
                ? '在这里，宿命被剥离为 68 个坐标与决策树的节点。我们公开算法路径，只为将最终的定义权交还于你。'
                : 'Here, fate is stripped down to 68 coordinates and decision tree nodes. We expose the algorithmic path, returning the power of definition back to you.'}
            </p>
            <div className="border border-[#222] bg-[#0a0a0a] p-6 text-left font-mono text-xs text-[#555]">
              <div className="animate-pulse">{lang === 'cn' ? '正在加载节点阈值矩阵...' : 'Loading node threshold matrix...'}</div>
              <div className="mt-2 text-[#d4af37]">{'>>'} jaw_angle &gt; 115.0 : True</div>
              <div className="mt-1">{'>>'} eye_distance_ratio &lt; 0.42 : False</div>
              <div className="mt-1 text-[#d4af37]">{'>>'} class_assigned : "FIRE_TYPE"</div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
