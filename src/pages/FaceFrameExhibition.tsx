import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent } from 'react'
import { FaceLandmarker, FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision'
import { ArrowRight, Crosshair, Fingerprint, Globe, Maximize, X } from 'lucide-react'
import * as THREE from 'three'
import { AlgorithmicEyeCursor } from '../components/global/AlgorithmicEyeCursor'
import { useLanguage } from '../i18n-context'
import { EMOTION_DICTIONARY } from '../../lexicon.js'

type ExhibitionPage = 'SCAN' | 'M1' | 'M2' | 'M3' | 'M4'
type FaceFrameExhibitionProps = {
  onBackHome?: () => void
}

type ScanUploadResponse = {
  uploadId?: string
  imageUrl?: string
  status?: string
  message?: string
}

type PersonaResponse = {
  personaId?: string
  imageUrl?: string
  status?: string
  message?: string
}

type M2SemanticState = 'ANGER' | 'SADNESS' | 'ANXIETY' | 'EXCITEMENT' | 'JOY' | 'CALM'
type M2State = 'NULL' | 'CALCULATING' | M2SemanticState | 'NEUTRAL'

type M2AnalysisResult = {
  state: M2SemanticState | 'NEUTRAL'
  confidence01: number
  confidencePercent: number
  sentiment: number
  polarity: { pos: number; neg: number; neu: number }
  keywords: string[]
  flow: { speed: number; density: number }
  accepted: boolean
  tokens: string[]
}

const MathUtils = {
  distance: (a: { x: number; y: number; z: number }, b: { x: number; y: number; z: number }) =>
    Math.hypot(a.x - b.x, a.y - b.y, (a.z ?? 0) - (b.z ?? 0)),
  angleBetween3Points: (
    a: { x: number; y: number; z: number },
    b: { x: number; y: number; z: number },
    c: { x: number; y: number; z: number },
  ) => {
    const v1 = { x: a.x - b.x, y: a.y - b.y, z: (a.z ?? 0) - (b.z ?? 0) }
    const v2 = { x: c.x - b.x, y: c.y - b.y, z: (c.z ?? 0) - (b.z ?? 0) }
    const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z
    const m1 = Math.hypot(v1.x, v1.y, v1.z)
    const m2 = Math.hypot(v2.x, v2.y, v2.z)
    if (m1 <= 0.00001 || m2 <= 0.00001) {
      return 120
    }
    const cos = Math.max(-1, Math.min(1, dot / (m1 * m2)))
    return (Math.acos(cos) * 180) / Math.PI
  },
}

const M1_MEDIAPIPE_68_INDICES = [
  234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 454,
  70, 63, 105, 66, 107,
  336, 296, 334, 293, 300,
  168, 6, 197, 195,
  98, 97, 2, 326, 327,
  33, 160, 158, 133, 153, 144,
  362, 385, 387, 263, 373, 380,
  61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91,
  78, 191, 80, 81, 82, 13, 312, 311,
] as const

const reduceToM168 = (landmarks: Array<{ x: number; y: number; z: number }>) => {
  const reduced = M1_MEDIAPIPE_68_INDICES.map((index) => landmarks[index]).filter(Boolean) as Array<{ x: number; y: number; z: number }>
  return reduced.length === 68 ? reduced : null
}

const extract12DimensionalFeatures = (landmarks68: Array<{ x: number; y: number; z: number }>) => {
  const required = [0, 1, 2, 4, 8, 10, 12, 14, 16, 19, 27, 30, 33, 36, 39, 42, 45, 48, 54, 57]
  if (required.some((index) => !landmarks68[index])) {
    return null
  }
  const p = (index: number) => landmarks68[index]
  const totalLength = MathUtils.distance(p(19), p(8))
  const upperThird = MathUtils.distance(p(19), p(27)) / Math.max(totalLength, 0.00001)
  const middleThird = MathUtils.distance(p(27), p(30)) / Math.max(totalLength, 0.00001)
  const lowerThird = MathUtils.distance(p(30), p(8)) / Math.max(totalLength, 0.00001)
  const faceWidth = MathUtils.distance(p(0), p(16))
  const leftEyeWidth = MathUtils.distance(p(36), p(39))
  const rightEyeWidth = MathUtils.distance(p(42), p(45))
  const avgEyeWidth = (leftEyeWidth + rightEyeWidth) / 2
  const fiveEyeRatio = faceWidth / Math.max(avgEyeWidth, 0.00001)
  const eyeToFaceRatio = MathUtils.distance(p(39), p(42)) / Math.max(faceWidth, 0.00001)
  const leftMandible = MathUtils.angleBetween3Points(p(0), p(4), p(8))
  const rightMandible = MathUtils.angleBetween3Points(p(16), p(12), p(8))
  const avgMandibular = (leftMandible + rightMandible) / 2
  const zygomaticAngle = MathUtils.angleBetween3Points(p(36), p(2), p(4))
  const browProminence = (p(19).z - p(27).z) * 100
  const aspectRatio = totalLength / Math.max(faceWidth, 0.00001)
  const angularity = MathUtils.distance(p(4), p(12)) / Math.max(faceWidth, 0.00001)
  const zygomaticProminence = MathUtils.distance(p(2), p(14)) / Math.max(faceWidth, 0.00001)
  const chinSharpness = MathUtils.distance(p(48), p(54)) / Math.max(MathUtils.distance(p(57), p(8)), 0.00001)

  return [
    upperThird,
    middleThird,
    lowerThird,
    fiveEyeRatio,
    eyeToFaceRatio,
    avgMandibular,
    zygomaticAngle,
    browProminence,
    aspectRatio,
    angularity,
    zygomaticProminence,
    chinSharpness,
  ]
}

const predictFacialDestiny = (landmarks68: Array<{ x: number; y: number; z: number }>) => {
  const dist = (p1: { x: number; y: number; z: number }, p2: { x: number; y: number; z: number }) =>
    Math.hypot(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z)

  const eyeWidth = dist(landmarks68[36], landmarks68[45])
  const innerEye = dist(landmarks68[39], landmarks68[42])
  const noseLen = dist(landmarks68[27], landmarks68[30])

  const ratioEye = innerEye / eyeWidth
  const ratioNose = noseLen / eyeWidth

  const lock1 = Math.round(ratioEye * 100)
  const lock2 = Math.round(ratioNose * 100)

  const bioHash = lock1 * 73 + lock2 * 37
  const destinyIndex = bioHash % 5

  let scores = [0, 0, 0, 0, 0]
  scores[destinyIndex] = 99.99

  const noise = (Date.now() % 100) / 10000
  scores = scores.map((score, index) => (index === destinyIndex ? score - noise * 4 : noise))

  const classes = ['METAL', 'WOOD', 'WATER', 'FIRE', 'EARTH']

  console.log('=== 绝对颅骨基因锁 ===')
  console.log(`硬骨比例: R1=${ratioEye.toFixed(4)}, R2=${ratioNose.toFixed(4)}`)
  console.log(`量化基因: [${lock1}, ${lock2}] -> Hash: ${bioHash}`)
  console.log(`宿命宣判: ${classes[destinyIndex]}`)
  console.log('======================')

  return {
    type: classes[destinyIndex],
    confidence: scores[destinyIndex].toFixed(2),
    allProbabilities: scores,
    normalizedVector: [ratioEye, ratioNose, lock1, lock2],
  }
}

type M3Element = 'metal' | 'water' | 'wood' | 'fire' | 'earth'
type M3Emotion = 'anxiety' | 'joy' | 'calm'
type M4EnergyElement = M3Element
type M4EnergyMood = 'anxiety' | 'joy' | 'calm' | 'neutral'

const m3Elements: M3Element[] = ['metal', 'water', 'wood', 'fire', 'earth']
const m3Emotions: M3Emotion[] = ['anxiety', 'joy', 'calm']
const m3TotemSpeciesMap: Record<M3Element, { cn: string; en: string }> = {
  metal: { cn: '狐狸', en: 'FOX' },
  wood: { cn: '鹿', en: 'DEER' },
  water: { cn: '锦鲤', en: 'KOI' },
  fire: { cn: '凤凰', en: 'PHOENIX' },
  earth: { cn: '熊', en: 'BEAR' },
}
const m4AmbientAudioSrc: string | null = null
const buildM3MaskCandidates = (element: M3Element, emotion: M3Emotion) => [`/m3-personas/${element}-${emotion}.png`]
const m3MaskPreloadPaths = Array.from(
  new Set(m3Elements.flatMap((element) => m3Emotions.flatMap((emotion) => buildM3MaskCandidates(element, emotion)))),
)

const resolveM3Element = (m1ResultType: string): M3Element => {
  const normalized = m1ResultType.toUpperCase()
  if (normalized.includes('METAL')) {
    return 'metal'
  }
  if (normalized.includes('WATER')) {
    return 'water'
  }
  if (normalized.includes('WOOD')) {
    return 'wood'
  }
  if (normalized.includes('FIRE')) {
    return 'fire'
  }
  return 'earth'
}

const mapEmotionToMorph = (fineGrainedEmotion: M2State): M3Emotion => {
  switch (fineGrainedEmotion) {
    case 'JOY':
    case 'EXCITEMENT':
      return 'joy'
    case 'ANXIETY':
    case 'ANGER':
    case 'SADNESS':
      return 'anxiety'
    case 'CALM':
    case 'NEUTRAL':
    case 'CALCULATING':
    case 'NULL':
    default:
      return 'calm'
  }
}

const resolveM4Mood = (m2State: M2State): M4EnergyMood => {
  if (m2State === 'ANXIETY' || m2State === 'ANGER' || m2State === 'SADNESS') {
    return 'anxiety'
  }
  if (m2State === 'JOY' || m2State === 'EXCITEMENT') {
    return 'joy'
  }
  if (m2State === 'CALM') {
    return 'calm'
  }
  return 'neutral'
}

const m4ElementColorMap: Record<M4EnergyElement, string> = {
  metal: '#D4AF37',
  water: '#3BA7FF',
  wood: '#4FBF5A',
  fire: '#FF5A36',
  earth: '#B8894F',
}

const m4MoodColorMap: Record<M4EnergyMood, string> = {
  anxiety: '#4B0082',
  joy: '#FFD700',
  calm: '#87CEEB',
  neutral: '#8B8B8B',
}

const m4CoreParticleColorMap: Record<M4EnergyElement, string> = {
  metal: '#DCDCDC',
  wood: '#228B22',
  water: '#0047AB',
  fire: '#DC143C',
  earth: '#DAA520',
}

const m4MoodWeights: Record<M4EnergyMood, Partial<Record<M4EnergyElement, number>>> = {
  anxiety: { water: 0.14, metal: 0.1, fire: 0.06 },
  joy: { fire: 0.16, wood: 0.09, metal: 0.05 },
  calm: { earth: 0.14, water: 0.1, wood: 0.06 },
  neutral: { earth: 0.18, metal: 0.07, water: 0.05 },
}

const m4Generating: Record<M4EnergyElement, M4EnergyElement> = {
  metal: 'water',
  water: 'wood',
  wood: 'fire',
  fire: 'earth',
  earth: 'metal',
}

const m4Overcoming: Record<M4EnergyElement, M4EnergyElement> = {
  metal: 'wood',
  wood: 'earth',
  earth: 'water',
  water: 'fire',
  fire: 'metal',
}

const mixHex = (from: string, to: string, ratio: number) => {
  const left = new THREE.Color(from)
  const right = new THREE.Color(to)
  return left.lerp(right, Math.max(0, Math.min(1, ratio))).getStyle()
}

const resolveM4RingColorByEmotion = (emotionState: M2State) => {
  if (emotionState === 'JOY' || emotionState === 'EXCITEMENT') {
    return '#FFD700'
  }
  if (emotionState === 'ANXIETY' || emotionState === 'ANGER' || emotionState === 'SADNESS') {
    return '#8A2BE2'
  }
  return '#00CED1'
}

const calcM4Energy = (element: M4EnergyElement, mood: M4EnergyMood, agencyValue: number) => {
  const scores: Record<M4EnergyElement, number> = { metal: 0, water: 0, wood: 0, fire: 0, earth: 0 }
  const moodVector = m4MoodWeights[mood]
  const moodAdded = Object.values(moodVector).reduce((sum, value) => sum + (value ?? 0), 0)
  scores[element] += 0.6
  Object.entries(moodVector).forEach(([key, value]) => {
    scores[key as M4EnergyElement] += value ?? 0
  })

  const baseAfterMood = { ...scores }
  let generateAdded = 0
  let overcomeReduced = 0
  ;(['metal', 'water', 'wood', 'fire', 'earth'] as M4EnergyElement[]).forEach((key) => {
    const source = baseAfterMood[key]
    const generateDelta = source * 0.12
    scores[m4Generating[key]] += generateDelta
    generateAdded += generateDelta
    const target = m4Overcoming[key]
    const before = scores[target]
    scores[target] = Math.max(0, before - source * 0.1)
    overcomeReduced += Math.max(0, before - scores[target])
  })

  const volitionRatio = Math.max(0, Math.min(1, agencyValue / 100))
  const provisionalDominant = (Object.entries(scores) as Array<[M4EnergyElement, number]>).sort((a, b) => b[1] - a[1])[0][0]
  const algorithmBiasDelta = (1 - volitionRatio) * 0.1
  const freedomBiasDelta = volitionRatio * 0.1
  scores[provisionalDominant] += algorithmBiasDelta
  scores[m4Generating[provisionalDominant]] += freedomBiasDelta

  const ranking = (Object.entries(scores) as Array<[M4EnergyElement, number]>).sort((a, b) => b[1] - a[1])
  const dominant = ranking[0][0]
  const total = ranking.reduce((sum, [, value]) => sum + value, 0.0001)
  const confidence = Number(((ranking[0][1] / total) * 100).toFixed(1))
  const moodLabelMap: Record<M4EnergyMood, string> = { anxiety: '焦虑', joy: '喜悦', calm: '平静', neutral: '中性' }
  const elementCnMap: Record<M4EnergyElement, string> = { metal: '金', water: '水', wood: '木', fire: '火', earth: '土' }
  const title = `${elementCnMap[dominant]}曜${moodLabelMap[mood]}态`
  const primaryColor = m4ElementColorMap[dominant]
  const overlayColor = m4MoodColorMap[mood]
  const ringBreakRatio = volitionRatio

  return {
    scores,
    dominant,
    confidence,
    title,
    primaryColor,
    overlayColor,
    ringBreakRatio,
    volitionRatio,
    algorithmWeight: 0.6,
    moodWeight: 0.3,
    willWeight: 0.1,
    explain: {
      boneAdded: 0.6,
      moodAdded: Number(moodAdded.toFixed(3)),
      generateAdded: Number(generateAdded.toFixed(3)),
      overcomeReduced: Number(overcomeReduced.toFixed(3)),
      algorithmBiasDelta: Number(algorithmBiasDelta.toFixed(3)),
      freedomBiasDelta: Number(freedomBiasDelta.toFixed(3)),
      provisionalDominant,
      generatedTarget: m4Generating[provisionalDominant],
    },
  }
}

const logicData = {
  M1: {
    title: 'M1 / Facial Topology',
    logic: 'MediaPipe FaceLandmarker + handcrafted geometry features.',
    data: 'Face mesh landmarks are sampled into 12 deterministic ratios/angles.',
    rule: 'predictFacialDestiny + thresholded zygomatic prominence => WuXing + bone label.',
  },
  M2: {
    title: 'M2 / Subconscious Profiling',
    logic: 'Lexicon-driven semantic parser with negation/booster handling.',
    data: 'Token segmentation + category buckets + punctuation boosts + confidence heuristic.',
    rule: 'State = bucket/sentiment decision; Confidence = dominance+margin+|sentiment|.',
  },
  M3: {
    title: 'M3 / Human Agency Override',
    logic: 'Single intervention variable: agencyValue (0-100).',
    data: 'Slider controls persona filter params and downstream M4 volition ratio.',
    rule: 'grayscale/saturate/contrast/brightness are mapped from agencyValue in realtime.',
  },
  M4: {
    title: 'M4 / Destiny Collapse',
    logic: 'Deterministic weighted synthesis with generating/overcoming cycles.',
    data: 'bone(0.6)+mood vector+sheng/ke transform+algorithm/will bias => dominant element.',
    rule: 'html2canvas exports the current DOM card with fixed render params.',
  },
} as const

export default function FaceFrameExhibition({ onBackHome }: FaceFrameExhibitionProps) {
  const { lang, toggle } = useLanguage()
  const [engineOpen, setEngineOpen] = useState(false)
  const [logicPanelContent, setLogicPanelContent] = useState<{ moduleId: 'M1' | 'M2' | 'M3' | 'M4'; lines: string[] }>({
    moduleId: 'M1',
    lines: [],
  })
  const [activePage, setActivePage] = useState<ExhibitionPage>('M1')
  const [mousePos, setMousePos] = useState({ x: -100, y: -100 })
  const [smoothPos, setSmoothPos] = useState({ x: -100, y: -100 })
  const [agencyValue, setAgencyValue] = useState(0)
  const [scanState, setScanState] = useState<'idle' | 'scanning' | 'complete'>('idle')
  const [gazeMode, setGazeMode] = useState<'idle' | 'analyzing'>('idle')
  const [moodText, setMoodText] = useState('')
  const [m2Input, setM2Input] = useState('')
  const [m2Logs, setM2Logs] = useState<string[]>([])
  const [m2State, setM2State] = useState<M2State>('NULL')
  const [m2Confidence, setM2Confidence] = useState(0)
  const [m2Keywords, setM2Keywords] = useState<string[]>([])
  const [m2FlowProfile, setM2FlowProfile] = useState({ speed: 0.6, density: 50, sentiment: 0, confidence: 0 })
  const [scanTotemTilt, setScanTotemTilt] = useState({ x: 0, y: 0 })
  const [m2ReadyToAdvance, setM2ReadyToAdvance] = useState(false)
  const [m2Collapsing, setM2Collapsing] = useState(false)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [selectedImageFile, setSelectedImageFile] = useState<File | null>(null)
  const [scanError, setScanError] = useState<string | null>(null)
  const [uploadResult, setUploadResult] = useState<ScanUploadResponse | null>(null)
  const [cameraOpen, setCameraOpen] = useState(false)
  const [m4GestureState, setM4GestureState] = useState<'open' | 'fist'>('open')
  const [m4TrackingActive, setM4TrackingActive] = useState(false)
  const [m4Activated, setM4Activated] = useState(false)
  const [m4GestureLabel, setM4GestureLabel] = useState('IDLE')
  const [m4CamStatus, setM4CamStatus] = useState('AWAITING CAMERA...')
  const [m4CamStatusColor, setM4CamStatusColor] = useState('#FFD700')
  const [m4Exporting, setM4Exporting] = useState(false)
  const [m4ExportHint, setM4ExportHint] = useState('')
  const [m4ExportError, setM4ExportError] = useState('')
  const [m4ExportEntryOpen, setM4ExportEntryOpen] = useState(false)
  const [m4ObserverInput, setM4ObserverInput] = useState('')
  const [m4ObserverName, setM4ObserverName] = useState('ANONYMOUS OBSERVER')
  const m4ExportFallbackStar = useMemo(() => {
    const fracturedRing = m2State === 'ANXIETY' || m2State === 'ANGER' || m2State === 'SADNESS'
    const seedSource = `m4-${m2State}`
    let seed = seedSource.split('').reduce((acc, char) => ((acc << 5) - acc + char.charCodeAt(0)) | 0, 0)
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0
      return seed / 4294967296
    }
    const coreCount = 1680
    const ringCount = 1360
    const corePoints = Array.from({ length: coreCount }, () => {
      const maxRadius = 78
      const radius = Math.cbrt(rand()) * maxRadius
      const theta = rand() * 2 * Math.PI
      const phi = Math.acos(2 * rand() - 1)
      const rawX = Math.sin(phi) * Math.cos(theta) * radius
      const rawY = Math.sin(phi) * Math.sin(theta) * radius
      const rawZ = Math.cos(phi) * radius
      const perspective = 0.78 + ((rawZ / maxRadius + 1) * 0.22)
      const x = rawX * perspective
      const y = rawY * perspective
      const projectedRadius = Math.sqrt(x * x + y * y)
      const centerBias = Math.max(0, 1 - projectedRadius / maxRadius)
      const rimBias = Math.max(0, 1 - Math.abs(projectedRadius - maxRadius * 0.78) / (maxRadius * 0.28))
      const nx = rawX / maxRadius
      const ny = rawY / maxRadius
      const nz = rawZ / maxRadius
      const lightBias = Math.max(0, nx * -0.26 + ny * -0.18 + nz * 0.74)
      return {
        x,
        y,
        r: 0.24 + centerBias * 0.94 + rimBias * 0.4 + lightBias * 0.58 + rand() * 0.14,
        a: 0.12 + centerBias * 0.46 + rimBias * 0.26 + lightBias * 0.4 + rand() * 0.08,
      }
    })
    const segmentCount = fracturedRing ? 4 : 1
    const segmentArc = (Math.PI * 2) / segmentCount
    const visibleArc = fracturedRing ? segmentArc * 0.62 : segmentArc * 0.96
    const ringTilt = Math.PI / 3
    const ringBackPoints: Array<{ x: number; y: number; r: number; a: number }> = []
    const ringFrontPoints: Array<{ x: number; y: number; r: number; a: number }> = []
    for (let segment = 0; segment < segmentCount; segment += 1) {
      const segmentStart = segment * segmentArc
      const perSegment = Math.floor(ringCount / segmentCount)
      for (let i = 0; i < perSegment; i += 1) {
        const progress = perSegment <= 1 ? 0 : i / (perSegment - 1)
        const angle = segmentStart + progress * visibleArc
        const radius = 106 + (rand() - 0.5) * 2.8
        const localY = (rand() - 0.5) * 1.1
        const x = Math.cos(angle) * radius
        const z = Math.sin(angle) * radius
        const y = localY * Math.cos(ringTilt) - z * Math.sin(ringTilt) * 0.4
        const zAfterTilt = localY * Math.sin(ringTilt) + z * Math.cos(ringTilt)
        const point = {
          x,
          y,
          r: 0.2 + rand() * 0.46,
          a: fracturedRing ? 0.14 + rand() * 0.24 : 0.2 + rand() * 0.24,
        }
        if (zAfterTilt >= 0) {
          ringFrontPoints.push({ ...point, a: point.a * 1.02 })
        } else {
          ringBackPoints.push({ ...point, a: point.a * 0.58 })
        }
      }
    }
    return { corePoints, ringBackPoints, ringFrontPoints }
  }, [m2State])
  const [m1SystemLog, setM1SystemLog] = useState(lang === 'cn' ? '等待数据源...' : 'AWAITING DATA SOURCE...')
  const [m1DeconstructVisible, setM1DeconstructVisible] = useState(false)
  const [m1ResultVisible, setM1ResultVisible] = useState(false)
  const [m1ResultType, setM1ResultType] = useState('--')
  const [m1ResultAngle, setM1ResultAngle] = useState('--°')
  const [m1ResultBone, setM1ResultBone] = useState('--')
  const [m1CaptureMode, setM1CaptureMode] = useState<'camera' | 'static' | null>(null)
  const [m1ShowEnterM2, setM1ShowEnterM2] = useState(false)
  const [m3Touched, setM3Touched] = useState(false)
  const [m3MaskSrc, setM3MaskSrc] = useState<string | null>(null)
  const [m3MaskLoading, setM3MaskLoading] = useState(false)
  const [m3MaskProgress, setM3MaskProgress] = useState(0)
  const [returnSigilActive, setReturnSigilActive] = useState(false)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const returnTimeoutRef = useRef<number | null>(null)
  const handLandmarkerRef = useRef<HandLandmarker | null>(null)
  const gestureRafRef = useRef<number | null>(null)
  const gestureStreakRef = useRef({ open: 0, fist: 0 })
  const m4AudioRef = useRef<HTMLAudioElement | null>(null)
  const m4CardRef = useRef<HTMLDivElement | null>(null)
  const m4ExportCardRef = useRef<HTMLDivElement | null>(null)
  const m4ContainerRef = useRef<HTMLDivElement | null>(null)
  const m4AnimationRef = useRef<number | null>(null)
  const m4ControlRef = useRef({
    targetZ: 180,
    currentZ: 180,
    targetRotX: 0,
    currentRotX: 0,
    targetRotY: 0,
    currentRotY: 0,
    targetRotZ: 0,
    currentRotZ: 0,
    targetScale: 1,
    currentScale: 1,
    targetBrightness: 1,
    currentBrightness: 1,
  })
  const cameraOpenRef = useRef(false)
  const m2ContainerRef = useRef<HTMLDivElement | null>(null)
  const m2RendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const m2SceneRef = useRef<THREE.Scene | null>(null)
  const m2CameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const m2SphereRef = useRef<THREE.Mesh | null>(null)
  const m2GroupRef = useRef<THREE.Group | null>(null)
  const m2UniformsRef = useRef<{
    u_time: { value: number }
    u_noise_freq: { value: number }
    u_noise_amp: { value: number }
    u_color_main: { value: THREE.Color }
    u_color_accent: { value: THREE.Color }
    u_mouse_hit: { value: THREE.Vector3 }
    u_mouse_active: { value: number }
  } | null>(null)
  const m2RaycasterRef = useRef<THREE.Raycaster | null>(null)
  const m2MouseRef = useRef<THREE.Vector2 | null>(null)
  const m2TargetRef = useRef<{
    freq: number
    amp: number
    rotSpeed: number
    colorMain: THREE.Color
    colorAccent: THREE.Color
  } | null>(null)
  const m2AnimationRef = useRef<number | null>(null)
  const m1ContainerRef = useRef<HTMLDivElement | null>(null)
  const m1VideoRef = useRef<HTMLVideoElement | null>(null)
  const m1UploadImageRef = useRef<HTMLImageElement | null>(null)
  const m1FileInputRef = useRef<HTMLInputElement | null>(null)
  const m1FaceLandmarkerRef = useRef<FaceLandmarker | null>(null)
  const m1ImageLandmarkerRef = useRef<FaceLandmarker | null>(null)
  const m1DetectRafRef = useRef<number | null>(null)
  const m1CameraStreamRef = useRef<MediaStream | null>(null)
  const m1CameraActiveRef = useRef(false)
  const m1CurrentLandmarksRef = useRef<Array<{ x: number; y: number; z: number }> | null>(null)
  const m1CurrentReduced68Ref = useRef<Array<{ x: number; y: number; z: number }> | null>(null)
  const m1PreviewUpdateAtRef = useRef(0)
  const m1IsDeconstructingRef = useRef(false)
  const m1DeconstructProgressRef = useRef(0)
  const m3MaskLoadVersionRef = useRef(0)
  const m3MaskLoadedOnceRef = useRef(false)
  const m3MaskBlobUrlRef = useRef<string | null>(null)
  const m3MaskSizeCacheRef = useRef<Record<string, number>>({})
  const engineButtonRef = useRef<HTMLButtonElement | null>(null)
  const logicOverlayRef = useRef<HTMLDivElement | null>(null)
  const scanApiUrl = import.meta.env.VITE_SCAN_API_URL ?? '/api/scan'
  const personaApiUrl = import.meta.env.VITE_PERSONA_API_URL ?? '/api/persona'

  useEffect(() => {
    const handleMouseMove = (event: MouseEvent) => setMousePos({ x: event.clientX, y: event.clientY })
    window.addEventListener('mousemove', handleMouseMove)
    return () => window.removeEventListener('mousemove', handleMouseMove)
  }, [])

  useEffect(() => {
    cameraOpenRef.current = cameraOpen
  }, [cameraOpen])


  useEffect(() => {
    let animationFrameId = 0
    const renderLoop = () => {
      setSmoothPos((previous) => ({
        x: previous.x + (mousePos.x - previous.x) * 0.15,
        y: previous.y + (mousePos.y - previous.y) * 0.15,
      }))
      animationFrameId = requestAnimationFrame(renderLoop)
    }
    animationFrameId = requestAnimationFrame(renderLoop)
    return () => cancelAnimationFrame(animationFrameId)
  }, [mousePos])

  useEffect(() => {
    return () => {
      if (gestureRafRef.current !== null) {
        window.cancelAnimationFrame(gestureRafRef.current)
      }
      if (handLandmarkerRef.current) {
        handLandmarkerRef.current.close()
      }
      if (returnTimeoutRef.current !== null) {
        window.clearTimeout(returnTimeoutRef.current)
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop())
      }
      if (m1DetectRafRef.current !== null) {
        window.cancelAnimationFrame(m1DetectRafRef.current)
      }
      if (m1FaceLandmarkerRef.current) {
        m1FaceLandmarkerRef.current.close()
      }
      if (m1ImageLandmarkerRef.current) {
        m1ImageLandmarkerRef.current.close()
      }
    }
  }, [])

  const handleReturnToOrigin = () => {
    setReturnSigilActive(true)
    if (returnTimeoutRef.current !== null) {
      window.clearTimeout(returnTimeoutRef.current)
    }
    returnTimeoutRef.current = window.setTimeout(() => {
      onBackHome?.()
    }, 220)
  }

  const ensureM1FaceLandmarker = async (mode: 'VIDEO' | 'IMAGE') => {
    if (mode === 'VIDEO' && m1FaceLandmarkerRef.current) {
      return m1FaceLandmarkerRef.current
    }
    if (mode === 'IMAGE' && m1ImageLandmarkerRef.current) {
      return m1ImageLandmarkerRef.current
    }
    const vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm')
    const landmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      },
      runningMode: mode,
      numFaces: 1,
    })
    if (mode === 'VIDEO') {
      m1FaceLandmarkerRef.current = landmarker
    } else {
      m1ImageLandmarkerRef.current = landmarker
    }
    return landmarker
  }

  const stopM1Camera = () => {
    m1CameraActiveRef.current = false
    if (m1DetectRafRef.current !== null) {
      window.cancelAnimationFrame(m1DetectRafRef.current)
      m1DetectRafRef.current = null
    }
    if (m1VideoRef.current) {
      m1VideoRef.current.srcObject = null
    }
  }

  const handleM1Camera = async () => {
    setM1CaptureMode('camera')
    setM1SystemLog(lang === 'cn' ? '初始化摄像模块...' : 'INITIALIZING CAMERA MODULE...')
    setM1ResultVisible(false)
    setM1ShowEnterM2(false)
    try {
      const detector = await ensureM1FaceLandmarker('VIDEO')
      const stream =
        streamRef.current ?? (await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false }))
      streamRef.current = stream
      m1CameraStreamRef.current = stream
      if (!m1VideoRef.current) {
        return
      }
      m1VideoRef.current.srcObject = stream
      await m1VideoRef.current.play()
      m1CameraActiveRef.current = true
      setM1SystemLog(lang === 'cn' ? '摄像头已激活，等待人脸...' : 'CAMERA ACTIVE. AWAITING FACE...')

      const loop = () => {
        if (!m1CameraActiveRef.current || !m1VideoRef.current) {
          return
        }
        if (m1VideoRef.current.readyState >= 2) {
          const result = detector.detectForVideo(m1VideoRef.current, performance.now())
          const landmarks = result.faceLandmarks?.[0]
          if (landmarks?.length) {
            const mapped = landmarks.map((point) => ({ x: point.x, y: point.y, z: point.z }))
            const reduced68 = reduceToM168(mapped)
            m1CurrentLandmarksRef.current = mapped
            m1CurrentReduced68Ref.current = reduced68
            const now = performance.now()
            if (reduced68 && now - m1PreviewUpdateAtRef.current > 160) {
              analyzeM1Topology(reduced68)
              m1PreviewUpdateAtRef.current = now
            }
            setM1SystemLog(reduced68 ? (lang === 'cn' ? '拓扑提取完成，准备解构。' : 'TOPOLOGY EXTRACTED. READY.') : (lang === 'cn' ? '68骨架提取失败...' : '68-SKELETON EXTRACTION FAILED...'))
            setM1DeconstructVisible(Boolean(reduced68))
          } else {
            setM1SystemLog(lang === 'cn' ? '正在搜索骨相结构...' : 'SEARCHING FOR STRUCTURE...')
            setM1DeconstructVisible(false)
          }
        }
        m1DetectRafRef.current = window.requestAnimationFrame(loop)
      }
      m1DetectRafRef.current = window.requestAnimationFrame(loop)
    } catch (error) {
      setM1SystemLog(`${lang === 'cn' ? '摄像错误' : 'CAMERA ERROR'}: ${error instanceof Error ? error.message : lang === 'cn' ? '未知错误' : 'UNKNOWN ERROR'}`)
    }
  }

  const handleM1UploadClick = () => {
    setM1CaptureMode('static')
    m1FileInputRef.current?.click()
  }

  const handleM1ImageFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) {
      return
    }
    stopM1Camera()
    setM1SystemLog(lang === 'cn' ? '处理静态图像...' : 'PROCESSING STATIC IMAGE...')
    setM1ResultVisible(false)
    setM1ShowEnterM2(false)
    try {
      const detector = await ensureM1FaceLandmarker('IMAGE')
      if (!m1UploadImageRef.current) {
        return
      }
      const url = URL.createObjectURL(file)
      m1UploadImageRef.current.onload = () => {
        const imageElement = m1UploadImageRef.current
        if (!imageElement) {
          return
        }
        const result = detector.detect(imageElement)
        const landmarks = result.faceLandmarks?.[0]
        if (landmarks?.length) {
          const mapped = landmarks.map((point) => ({ x: point.x, y: point.y, z: point.z }))
          const reduced68 = reduceToM168(mapped)
          m1CurrentLandmarksRef.current = mapped
          m1CurrentReduced68Ref.current = reduced68
          if (reduced68) {
            analyzeM1Topology(reduced68)
          }
          setM1SystemLog(reduced68 ? (lang === 'cn' ? '拓扑提取完成，准备解构。' : 'TOPOLOGY EXTRACTED. READY.') : (lang === 'cn' ? '68骨架提取失败...' : '68-SKELETON EXTRACTION FAILED...'))
          setM1DeconstructVisible(Boolean(reduced68))
        } else {
          setM1SystemLog(lang === 'cn' ? '正在搜索骨相结构...' : 'SEARCHING FOR STRUCTURE...')
          setM1DeconstructVisible(false)
        }
        URL.revokeObjectURL(url)
      }
      m1UploadImageRef.current.src = url
    } catch (error) {
      setM1SystemLog(`${lang === 'cn' ? '图像错误' : 'IMAGE ERROR'}: ${error instanceof Error ? error.message : lang === 'cn' ? '未知错误' : 'UNKNOWN ERROR'}`)
    }
  }

  const analyzeM1Topology = (landmarks68: Array<{ x: number; y: number; z: number }>) => {
    const features = extract12DimensionalFeatures(landmarks68)
    if (!features) {
      return {
        typeLabel: 'EARTH (土)',
        angleLabel: '115.0°',
        boneLabel: 'BALANCED',
      }
    }
    const result = predictFacialDestiny(landmarks68)
    const zygomaticProminence = features[10]
    const boneLabel = zygomaticProminence > 0.94 ? 'PROMINENT' : zygomaticProminence > 0.87 ? 'BALANCED' : 'SOFT'
    return {
      typeLabel: result.type,
      angleLabel: `${features[5].toFixed(1)}°`,
      boneLabel,
    }
  }

  const handleM1Deconstruct = () => {
    const landmarks68 = m1CurrentReduced68Ref.current
    if (!landmarks68) {
      return
    }
    setM1DeconstructVisible(false)
    setM1SystemLog(lang === 'cn' ? '执行拓扑解构中...' : 'EXECUTING TOPOLOGICAL DECONSTRUCTION...')
    m1IsDeconstructingRef.current = true
    m1DeconstructProgressRef.current = 0
    const interval = window.setInterval(() => {
      m1DeconstructProgressRef.current += 0.1
    }, 100)
    window.setTimeout(() => {
      window.clearInterval(interval)
      m1IsDeconstructingRef.current = false
      stopM1Camera()
      const analyzed = analyzeM1Topology(landmarks68)
      setM1ResultType(analyzed.typeLabel)
      setM1ResultAngle(analyzed.angleLabel)
      setM1ResultBone(analyzed.boneLabel)
      setM1ResultVisible(true)
      setM1SystemLog(lang === 'cn' ? '宿命锁定，拓扑已固化。' : 'FATE SECURED. TOPOLOGY LOCKED.')
      window.setTimeout(() => {
        setM1ShowEnterM2(true)
      }, 1000)
    }, 2500)
  }

  const handleM4Start = () => {
    setM4Activated(true)
    setM4GestureLabel('SEARCHING ENTITY...')
    setM4CamStatus('AWAITING CAMERA...')
    setM4CamStatusColor('#FFD700')
    setM4ExportHint('')
    if (m4AudioRef.current) {
      m4AudioRef.current.volume = 0.6
      void m4AudioRef.current.play().catch(() => undefined)
    }
    void openCamera()
  }

  const openCamera = async () => {
    if (streamRef.current) {
      if (videoRef.current) {
        videoRef.current.srcObject = streamRef.current
      }
      setCameraOpen(true)
      setM4TrackingActive(true)
      setM4CamStatus('')
      return
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
      setCameraOpen(true)
      setM4TrackingActive(true)
      setM4CamStatus('')
      setScanError(null)
    } catch {
      setCameraOpen(false)
      setM4TrackingActive(false)
      setM4GestureLabel('MOUSE OVERRIDE')
      setM4CamStatus('CAMERA BLOCKED / USING MOUSE')
      setM4CamStatusColor('#ff4444')
    }
  }

  const closeCamera = () => {
    if (gestureRafRef.current !== null) {
      window.cancelAnimationFrame(gestureRafRef.current)
      gestureRafRef.current = null
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    setCameraOpen(false)
    setM4TrackingActive(false)
    setM4GestureState('open')
    gestureStreakRef.current = { open: 0, fist: 0 }
  }

  const captureFromCamera = () => {
    if (!videoRef.current) {
      return
    }
    const video = videoRef.current
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth || 1080
    canvas.height = video.videoHeight || 1440
    const context = canvas.getContext('2d')
    if (!context) {
      return
    }
    context.drawImage(video, 0, 0, canvas.width, canvas.height)
    canvas.toBlob((blob) => {
      if (!blob) {
        return
      }
      const file = new File([blob], `faceframe-capture-${Date.now()}.jpg`, { type: 'image/jpeg' })
      setSelectedImageFile(file)
      setImagePreview(URL.createObjectURL(file))
      setScanError(null)
    }, 'image/jpeg', 0.95)
  }

  const handleImageFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) {
      return
    }
    setSelectedImageFile(file)
    setImagePreview(URL.createObjectURL(file))
    setScanError(null)
  }

  const uploadScanData = async (): Promise<ScanUploadResponse> => {
    if (!selectedImageFile) {
      throw new Error(lang === 'cn' ? '缺少面容图像，请先拍照或上传照片。' : 'Missing face image. Please capture or upload a photo first.')
    }
    if (!moodText.trim()) {
      throw new Error(lang === 'cn' ? '请输入当前心境文本后再开始凝视。' : 'Please enter your current mood text before starting.')
    }

    const formData = new FormData()
    formData.append('image', selectedImageFile)
    formData.append('mood', moodText.trim())
    formData.append('timestamp', new Date().toISOString())

    const response = await fetch(scanApiUrl, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      throw new Error(lang === 'cn' ? `上传失败（${response.status}）` : `Upload failed (${response.status})`)
    }

    const data = (await response.json()) as ScanUploadResponse
    return data
  }

  const requestPersona = async (uploadId?: string): Promise<PersonaResponse> => {
    const response = await fetch(personaApiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ uploadId }),
    })

    if (!response.ok) {
      throw new Error(lang === 'cn' ? `Persona 生成失败（${response.status}）` : `Persona generation failed (${response.status})`)
    }

    const data = (await response.json()) as PersonaResponse
    return data
  }

  const handleScan = async () => {
    if (scanState !== 'idle') {
      return
    }
    setScanError(null)
    setUploadResult(null)
    setScanState('scanning')
    try {
      const [result] = await Promise.all([
        uploadScanData(),
        new Promise((resolve) => setTimeout(resolve, 4000)),
      ])
      setUploadResult(result)
      void requestPersona(result.uploadId).catch(() => undefined)
      setScanState('complete')
    } catch (error) {
      setScanState('idle')
      setScanError(error instanceof Error ? error.message : lang === 'cn' ? '扫描失败，请稍后重试。' : 'Scan failed, please try again.')
    }
  }

  const m2StopWords = new Set([
    '的',
    '了',
    '吗',
    '呢',
    '啊',
    '哦',
    '呀',
    '吧',
    '我',
    '你',
    '他',
    '她',
    '它',
    '我们',
    '你们',
    '他们',
    '是',
    '在',
    '有',
    '和',
    '也',
    '都',
    '就',
    '还',
    '与',
    '及',
    '着',
  ])

  const m2SemanticCategories: M2SemanticState[] = ['ANGER', 'SADNESS', 'ANXIETY', 'EXCITEMENT', 'JOY', 'CALM']
  const m2EmotionLexicon = Object.fromEntries(
    m2SemanticCategories.map((category) => [category, new Set(EMOTION_DICTIONARY[category] ?? [])]),
  ) as Record<M2SemanticState, Set<string>>
  const m2ValenceByCategory: Record<M2SemanticState, number> = {
    ANGER: -1.1,
    SADNESS: -1.05,
    ANXIETY: -1,
    EXCITEMENT: 1,
    JOY: 1.05,
    CALM: 0.45,
  }

  const m2Negations = ['不', '没', '無', '无', '非', '别', '未']
  const m2Boosters: Record<string, number> = { 非常: 0.32, 很: 0.2, 特别: 0.28, 极其: 0.4, 太: 0.22 }

  const cleanM2Text = (text: string) => text.replace(/[^\p{L}\p{N}\s]/gu, ' ').replace(/\s+/g, ' ').trim()

  const m2SegmentDictionary = Array.from(
    new Set([
      ...m2SemanticCategories.flatMap((category) => [...m2EmotionLexicon[category]]),
      ...Object.keys(m2Boosters),
      ...m2Negations,
    ]),
  ).sort((a, b) => b.length - a.length)

  const segmentM2Text = (text: string) => {
    const compact = text.replace(/\s+/g, '')
    const tokens: string[] = []
    let i = 0
    while (i < compact.length) {
      let matched = ''
      for (const term of m2SegmentDictionary) {
        if (compact.startsWith(term, i)) {
          matched = term
          break
        }
      }
      if (matched) {
        tokens.push(matched)
        i += matched.length
      } else {
        const single = compact[i]
        if (single.trim()) {
          tokens.push(single)
        }
        i += 1
      }
    }
    return tokens.filter((token) => token && !m2StopWords.has(token))
  }

  const analyzeSemantics = (inputText: string): M2AnalysisResult => {
    const exclamationCount = (inputText.match(/[!！]/g) ?? []).length
    const ellipsisCount = (inputText.match(/(\.\.\.|…|……)/g) ?? []).length
    const cleaned = cleanM2Text(inputText)
    const tokens = segmentM2Text(cleaned)
    const freqMap = new Map<string, number>()
    tokens.forEach((token) => freqMap.set(token, (freqMap.get(token) ?? 0) + 1))
    const matchedTokens = tokens.filter((token) => m2SemanticCategories.some((category) => m2EmotionLexicon[category].has(token)))
    const effectiveStates: M2SemanticState[] = []

    let sentimentAccumulator = 0
    const bucketScores: Record<M2SemanticState, number> = {
      ANGER: 0,
      SADNESS: 0,
      ANXIETY: 0,
      EXCITEMENT: 0,
      JOY: 0,
      CALM: 0,
    }

    tokens.forEach((token, index) => {
      const prev = tokens[Math.max(0, index - 1)] ?? ''
      const prev2 = tokens[Math.max(0, index - 2)] ?? ''
      const negated = m2Negations.includes(prev) || m2Negations.includes(prev2)
      const booster = (m2Boosters[prev] ?? 0) + (m2Boosters[prev2] ?? 0)
      const weight = 1 + booster
      m2SemanticCategories.forEach((category) => {
        if (!m2EmotionLexicon[category].has(token)) {
          return
        }
        if (negated) {
          if (category === 'JOY' || category === 'EXCITEMENT') {
            bucketScores.SADNESS += weight * 0.7
            bucketScores.ANXIETY += weight * 0.45
            sentimentAccumulator -= weight * 0.9
            effectiveStates.push('SADNESS')
            return
          }
          if (category === 'CALM') {
            bucketScores.ANXIETY += weight * 0.7
            sentimentAccumulator -= weight * 0.55
            effectiveStates.push('ANXIETY')
            return
          }
          bucketScores.CALM += weight * 0.45
          sentimentAccumulator += weight * 0.15
          effectiveStates.push('CALM')
          return
        }
        bucketScores[category] += weight
        sentimentAccumulator += m2ValenceByCategory[category] * weight
        effectiveStates.push(category)
      })
    })

    if (exclamationCount > 0) {
      const energeticBoost = 1 + Math.min(0.42, exclamationCount * 0.08)
      bucketScores.ANGER *= energeticBoost
      bucketScores.EXCITEMENT *= energeticBoost
      bucketScores.ANXIETY *= 1 + Math.min(0.24, exclamationCount * 0.05)
      bucketScores.JOY *= 1 + Math.min(0.15, exclamationCount * 0.03)
    }
    if (ellipsisCount > 0) {
      const lowEnergyBoost = 1 + Math.min(0.5, ellipsisCount * 0.1)
      bucketScores.SADNESS *= lowEnergyBoost
      bucketScores.CALM *= 1 + Math.min(0.35, ellipsisCount * 0.08)
    }

    const denom = Math.sqrt(sentimentAccumulator * sentimentAccumulator + 15)
    const sentiment = denom === 0 ? 0 : Number((sentimentAccumulator / denom).toFixed(4))
    const posScore = Math.max(0, sentiment)
    const negScore = Math.max(0, -sentiment)
    const neuScore = Number((1 - Math.min(1, Math.abs(sentiment))).toFixed(4))

    const rankedBuckets = (Object.entries(bucketScores) as Array<[keyof typeof bucketScores, number]>).sort(
      (a, b) => b[1] - a[1],
    )
    const [topBucket, topBucketScore] = rankedBuckets[0]
    const secondBucketScore = rankedBuckets[1][1]

    const hasLexiconHit = matchedTokens.length > 0
    let state: M2SemanticState | 'NEUTRAL' = 'NEUTRAL'
    if (hasLexiconHit) {
      state = topBucket
    } else if (topBucketScore >= 1.1 && topBucketScore - secondBucketScore >= 0.18) {
      state = topBucket
    } else if (sentiment <= -0.58) {
      state = 'SADNESS'
    } else if (sentiment < -0.3) {
      state = 'ANXIETY'
    } else if (sentiment >= 0.58) {
      state = 'EXCITEMENT'
    } else if (sentiment >= 0.28) {
      state = 'JOY'
    } else if (sentiment >= -0.22 && sentiment <= 0.22) {
      state = 'CALM'
    }
    const distinctEffectiveStates = Array.from(new Set(effectiveStates))
    if (distinctEffectiveStates.length >= 2 && effectiveStates.length > 0) {
      state = effectiveStates[effectiveStates.length - 1]
    }

    const keywordScores = [...freqMap.entries()].map(([term, tf]) => {
      const matchedCategory = m2SemanticCategories.find((category) => m2EmotionLexicon[category].has(term))
      return { term, score: tf * ((matchedCategory ? Math.abs(m2ValenceByCategory[matchedCategory]) : 0.5) + 1) }
    })
    const topKeywords = keywordScores
      .sort((a, b) => b.score - a.score)
      .slice(0, 3)
      .map((item) => item.term)

    const totalBucketScore = Object.values(bucketScores).reduce((sum, value) => sum + value, 0.00001)
    const dominance = topBucketScore / totalBucketScore
    const margin = (topBucketScore - secondBucketScore) / Math.max(topBucketScore, 0.00001)
    const confidence01 = Number(Math.min(1, dominance * 0.62 + Math.max(0, margin) * 0.28 + Math.abs(sentiment) * 0.18).toFixed(3))
    const accepted = (confidence01 >= 0.4 || hasLexiconHit) && state !== 'NEUTRAL'
    const finalState = accepted ? state : 'NEUTRAL'
    const finalConfidence = accepted ? confidence01 : Number((confidence01 * 0.72).toFixed(3))

    const fallbackKeywords: Record<typeof finalState, string[]> = {
      ANGER: ['VOLATILE', 'ERUPTION', 'SPIKE'],
      SADNESS: ['GRAVITY', 'SINKING', 'DEEP_SEA'],
      ANXIETY: ['ENTROPY_UP', 'CHAOS', 'TREMOR'],
      EXCITEMENT: ['ADRENALINE', 'PULSE', 'FLASH'],
      JOY: ['DOPAMINE', 'EXPANSION', 'RADIANT'],
      CALM: ['ZEN', 'FLOW', 'SLOW_WAVE'],
      NEUTRAL: ['未锁定', '低置信', '再描述'],
    }

    const flowMap: Record<typeof finalState, { speed: number; density: number }> = {
      ANGER: { speed: 1.35, density: 88 },
      SADNESS: { speed: 0.35, density: 22 },
      ANXIETY: { speed: 1.1, density: 72 },
      EXCITEMENT: { speed: 1.15, density: 84 },
      JOY: { speed: 0.85, density: 58 },
      CALM: { speed: 0.3, density: 18 },
      NEUTRAL: { speed: 0.6, density: 35 },
    }

    return {
      state: finalState,
      confidence01: finalConfidence,
      confidencePercent: Number((finalConfidence * 100).toFixed(1)),
      sentiment,
      polarity: { pos: Number(posScore.toFixed(4)), neg: Number(negScore.toFixed(4)), neu: neuScore },
      keywords: topKeywords.length === 3 ? topKeywords : [...fallbackKeywords[finalState]],
      flow: flowMap[finalState],
      accepted,
      tokens,
    }
  }

  const handleM2Submit = () => {
    const text = m2Input.trim()
    if (!text) {
      return
    }
    setM2Logs((previous) => [...previous, text].slice(-5))
    setM2Input('')
    setM2State('CALCULATING')
    setM2Confidence(0)
    setM2Keywords([])
    setM2ReadyToAdvance(false)
    window.setTimeout(() => {
      const result = analyzeSemantics(text)
      setM2State(result.state)
      setM2Confidence(result.confidencePercent)
      setM2ReadyToAdvance(result.accepted)
      setM2Keywords(result.keywords)
      setM2FlowProfile({
        speed: result.flow.speed,
        density: result.flow.density,
        sentiment: result.sentiment,
        confidence: result.confidence01,
      })
      triggerStateTransition(result)
    }, 700)
  }

  const handleM2Advance = () => {
    setM2Collapsing(true)
    const target = m2TargetRef.current
    if (target) {
      target.amp = 0
      target.freq = 0.25
      target.colorMain = new THREE.Color('#ffffff')
      target.colorAccent = new THREE.Color('#ffffff')
    }
    window.setTimeout(() => {
      setM2Collapsing(false)
      setM2ReadyToAdvance(false)
      setActivePage('M3')
    }, 850)
  }

  const handleM3Manifest = () => {
    setM4Activated(false)
    setM4TrackingActive(false)
    setM4GestureLabel('IDLE')
    setM4CamStatus('AWAITING CAMERA...')
    setM4CamStatusColor('#FFD700')
    setActivePage('M4')
  }

  const m3Element = useMemo(() => resolveM3Element(m1ResultType), [m1ResultType])
  const m3Morph = useMemo(() => mapEmotionToMorph(m2State), [m2State])
  const m3MaskCandidates = useMemo(() => buildM3MaskCandidates(m3Element, m3Morph), [m3Element, m3Morph])
  const m4BoneElement = useMemo(() => resolveM3Element(m1ResultType), [m1ResultType])
  const m4Mood = useMemo(() => resolveM4Mood(m2State), [m2State])
  const m4Energy = useMemo(() => calcM4Energy(m4BoneElement, m4Mood, agencyValue), [m4BoneElement, m4Mood, agencyValue])
  const shouldRenderM4ExportCard = m4ExportEntryOpen || m4Exporting
  const loadM3Mask = useCallback((paths: string[]) => {
    const currentVersion = m3MaskLoadVersionRef.current + 1
    m3MaskLoadVersionRef.current = currentVersion
    const shouldShowLoadingOverlay = !m3MaskLoadedOnceRef.current
    if (shouldShowLoadingOverlay) {
      setM3MaskLoading(true)
      setM3MaskProgress(0)
    }

    const tryLoad = async (index: number): Promise<void> => {
      const path = paths[index]
      if (!path) {
        if (m3MaskLoadVersionRef.current !== currentVersion) {
          return
        }
        if (shouldShowLoadingOverlay) {
          setM3MaskLoading(false)
          setM3MaskProgress(0)
        }
        return
      }

      try {
        const nextSrc = await new Promise<string>((resolve, reject) => {
          const xhr = new XMLHttpRequest()
          const requestPath = `${path}${path.includes('?') ? '&' : '?'}m3t=${Date.now()}`
          xhr.open('GET', requestPath, true)
          xhr.responseType = 'blob'
          xhr.onprogress = (event) => {
            if (!shouldShowLoadingOverlay || m3MaskLoadVersionRef.current !== currentVersion) {
              return
            }
            if (event.lengthComputable && event.total > 0) {
              setM3MaskProgress(Math.max(1, Math.min(99, Math.round((event.loaded / event.total) * 100))))
              return
            }
            const knownSize = m3MaskSizeCacheRef.current[path]
            if (knownSize > 0) {
              setM3MaskProgress(Math.max(1, Math.min(99, Math.round((event.loaded / knownSize) * 100))))
            }
          }
          xhr.onload = () => {
            if (xhr.status >= 200 && xhr.status < 300 && xhr.response) {
              m3MaskSizeCacheRef.current[path] = xhr.response.size
              resolve(URL.createObjectURL(xhr.response))
              return
            }
            reject(new Error(`HTTP ${xhr.status}`))
          }
          xhr.onerror = () => reject(new Error('network error'))
          xhr.send()
        })

        if (m3MaskLoadVersionRef.current !== currentVersion) {
          if (nextSrc.startsWith('blob:')) {
            URL.revokeObjectURL(nextSrc)
          }
          return
        }

        if (m3MaskBlobUrlRef.current) {
          URL.revokeObjectURL(m3MaskBlobUrlRef.current)
          m3MaskBlobUrlRef.current = null
        }
        m3MaskBlobUrlRef.current = nextSrc

        setM3MaskSrc(nextSrc)
        m3MaskLoadedOnceRef.current = true
        if (shouldShowLoadingOverlay) {
          setM3MaskProgress(100)
          setM3MaskLoading(false)
        }
      } catch {
        if (m3MaskLoadVersionRef.current !== currentVersion) {
          return
        }
        await tryLoad(index + 1)
      }
    }

    void tryLoad(0)
  }, [])

  const handleM4Export = () => {
    if (m4Exporting) {
      return
    }
    setM4ExportError('')
    setM4ObserverInput('')
    setM4ExportEntryOpen(true)
  }

  const handleM4ExportConfirm = async () => {
    if (!m4ExportCardRef.current || m4Exporting) {
      setM4ExportHint('EXPORT FAILED')
      setM4ExportError('CARD_NOT_READY')
      return
    }
    const finalName = m4ObserverInput.trim() || 'ANONYMOUS OBSERVER'
    setM4ObserverName(finalName)
    setM4ExportEntryOpen(false)
    setM4Exporting(true)
    setM4ExportHint('EXPORTING...')
    setM4ExportError('')
    try {
      if (typeof document !== 'undefined' && 'fonts' in document) {
        await (document as Document & { fonts?: FontFaceSet }).fonts?.ready
      }
      await new Promise((resolve) => window.requestAnimationFrame(() => window.requestAnimationFrame(resolve)))
      const { default: html2canvas } = await import('html2canvas')
      const exportScale = Math.max(3, (window.devicePixelRatio || 1) * 2)
      const canvas = await html2canvas(m4ExportCardRef.current, {
        backgroundColor: null,
        useCORS: true,
        scale: exportScale,
        removeContainer: true,
        onclone: (clonedDocument) => {
          clonedDocument.querySelectorAll('style,link[rel="stylesheet"]').forEach((node) => node.remove())
        },
      })
      if (!canvas.width || !canvas.height) {
        throw new Error('EMPTY_CANVAS')
      }
      const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'))
      if (!blob) {
        throw new Error('BLOB_CREATION_FAILED')
      }
      const href = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = href
      anchor.download = `faceframe-destiny-${Date.now()}.png`
      anchor.click()
      window.setTimeout(() => URL.revokeObjectURL(href), 1000)
      setM4ExportHint('EXPORTED')
      window.setTimeout(() => setM4ExportHint(''), 1600)
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'UNKNOWN_ERROR'
      setM4ExportError(detail)
      setM4ExportHint('EXPORT FAILED')
      console.error('M4 export failed:', error)
      window.setTimeout(() => setM4ExportHint(''), 2000)
    } finally {
      setM4Exporting(false)
    }
  }

  useEffect(() => {
    if (cameraOpen && streamRef.current && videoRef.current) {
      videoRef.current.srcObject = streamRef.current
      void videoRef.current.play().catch(() => undefined)
    }
  }, [cameraOpen, activePage, m4TrackingActive])

  useEffect(() => {
    const preloaders = m3MaskPreloadPaths.map((src) => {
      const image = new Image()
      image.decoding = 'async'
      image.src = src
      return image
    })
    return () => {
      preloaders.forEach((image) => {
        image.src = ''
      })
    }
  }, [])

  useEffect(() => {
    return () => {
      if (m3MaskBlobUrlRef.current) {
        URL.revokeObjectURL(m3MaskBlobUrlRef.current)
      }
    }
  }, [])

  useEffect(() => {
    if (activePage !== 'M3') {
      return
    }
    loadM3Mask(m3MaskCandidates)
  }, [activePage, m3MaskCandidates, loadM3Mask])

  useEffect(() => {
    if (!m4TrackingActive || !cameraOpen) {
      if (gestureRafRef.current !== null) {
        window.cancelAnimationFrame(gestureRafRef.current)
        gestureRafRef.current = null
      }
      return
    }

    let cancelled = false

    const inferGesture = (landmarks: Array<{ x: number; y: number; z?: number }>) => {
      const wrist = landmarks[0]
      const middleTip = landmarks[12]
      const indexBase = landmarks[5]
      const middleMcp = landmarks[9]
      if (!wrist || !middleTip || !indexBase || !middleMcp) {
        return
      }

      const dWristToTip = Math.hypot(middleTip.x - wrist.x, middleTip.y - wrist.y)
      const dWristToBase = Math.hypot(indexBase.x - wrist.x, indexBase.y - wrist.y)
      if (dWristToBase <= 0.0001) {
        return
      }
      const ratio = dWristToTip / dWristToBase
      const control = m4ControlRef.current
      const yawAngle = Math.atan2(middleMcp.x - wrist.x, middleMcp.y - wrist.y)
      control.targetRotY = yawAngle * 1.2
      control.targetRotZ = -yawAngle * 0.6

      if (ratio < 1.4) {
        setM4GestureState('fist')
        setM4GestureLabel('FIST // REPULSION')
        control.targetZ = 280
        control.targetScale = 0.9
        control.targetBrightness = 0.78
      } else {
        setM4GestureState('open')
        setM4GestureLabel('OPEN // ATTRACTION')
        control.targetZ = 90
        control.targetScale = 1.08
        control.targetBrightness = 1.14
      }
    }

    const bootstrap = async () => {
      if (!handLandmarkerRef.current) {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm',
        )
        if (cancelled) {
          return
        }
        handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
          },
          runningMode: 'VIDEO',
          numHands: 1,
        })
      }

      const loop = () => {
        if (cancelled) {
          return
        }
        const video = videoRef.current
        const detector = handLandmarkerRef.current
        if (!video || !detector) {
          gestureRafRef.current = window.requestAnimationFrame(loop)
          return
        }

        if (video.readyState >= 2) {
          const result = detector.detectForVideo(video, performance.now())
          const landmarks = result.landmarks?.[0]
          if (landmarks?.length) {
            inferGesture(landmarks)
          } else {
            setM4GestureLabel('SEARCHING ENTITY...')
            const control = m4ControlRef.current
            control.targetZ = 180
            control.targetRotZ = 0
            control.targetScale = 1
            control.targetBrightness = 1
          }
        }

        gestureRafRef.current = window.requestAnimationFrame(loop)
      }

      gestureRafRef.current = window.requestAnimationFrame(loop)
    }

    void bootstrap()

    return () => {
      cancelled = true
      if (gestureRafRef.current !== null) {
        window.cancelAnimationFrame(gestureRafRef.current)
        gestureRafRef.current = null
      }
    }
  }, [m4TrackingActive, cameraOpen])

  const viewportWidth = typeof window === 'undefined' ? 1440 : window.innerWidth
  const viewportHeight = typeof window === 'undefined' ? 900 : window.innerHeight
  const pointerXRatio = Math.min(1, Math.max(0, smoothPos.x / viewportWidth))
  const pointerYRatio = Math.min(1, Math.max(0, smoothPos.y / viewportHeight))
  const prismRotateY = (pointerXRatio - 0.5) * 16
  const prismRotateX = (0.5 - pointerYRatio) * 12
  const agencyRatio = agencyValue / 100
  const personaGrayscale = Math.max(0, Math.min(1, 1 - agencyRatio))
  const personaFilter = `drop-shadow(0 0 20px rgba(180, 238, 255, 0.2)) grayscale(${personaGrayscale}) saturate(${0.92 + agencyRatio * 0.35}) contrast(${1 + agencyRatio * 0.12}) brightness(${0.98 + agencyRatio * 0.06})`
  const m2Accent = useMemo(() => {
    if (m2State === 'ANXIETY') {
      return '#4B0082'
    }
    if (m2State === 'SADNESS') {
      return '#4A5A70'
    }
    if (m2State === 'ANGER') {
      return '#FF4500'
    }
    if (m2State === 'JOY') {
      return '#FFD700'
    }
    if (m2State === 'EXCITEMENT') {
      return '#00FFFF'
    }
    if (m2State === 'CALM') {
      return '#87CEEB'
    }
    if (m2State === 'CALCULATING') {
      return '#E0E0E0'
    }
    return '#BDC3C7'
  }, [m2State])
  const m2StateCopy = useMemo(() => {
    if (m2State === 'ANGER') {
      return lang === 'cn' ? { label: '易变 / 喷发', sub: '尖峰前沿' } : { label: 'VOLATILE / ERUPTION', sub: 'SPIKE FRONT' }
    }
    if (m2State === 'SADNESS') {
      return lang === 'cn' ? { label: '重力 / 下沉', sub: '深海漂移' } : { label: 'GRAVITY / SINKING', sub: 'DEEP SEA DRIFT' }
    }
    if (m2State === 'ANXIETY') {
      return lang === 'cn' ? { label: '熵增 / 混沌', sub: '高速微脉冲' } : { label: 'ENTROPY_UP / CHAOS', sub: 'RAPID MICRO-PULSE' }
    }
    if (m2State === 'EXCITEMENT') {
      return lang === 'cn' ? { label: '肾上腺素 / 脉冲', sub: '青色闪爆' } : { label: 'ADRENALINE / PULSE', sub: 'CYAN FLASH BURST' }
    }
    if (m2State === 'JOY') {
      return lang === 'cn' ? { label: '多巴胺 / 扩张', sub: '金色盛放' } : { label: 'DOPAMINE / EXPANSION', sub: 'GOLDEN BLOOM' }
    }
    if (m2State === 'CALM') {
      return lang === 'cn' ? { label: '禅 / 流动', sub: '低频静稳' } : { label: 'ZEN / FLOW', sub: 'LOW-FREQ STILLNESS' }
    }
    if (m2State === 'CALCULATING') {
      return lang === 'cn' ? { label: '解析中 / 语义', sub: '墨场校准中' } : { label: 'PARSING / SEMANTICS', sub: 'INK FIELD CALIBRATING' }
    }
    return lang === 'cn' ? { label: '等待中 / 低置信', sub: '基线微波' } : { label: 'AWAITING / LOW_CONFIDENCE', sub: 'BASELINE MICRO-WAVE' }
  }, [m2State, lang])

  useEffect(() => {
    if (activePage !== 'M2') {
      return
    }
    const container = m2ContainerRef.current
    if (!container) {
      return
    }

    const vertexShader = `
      #define PI 3.14159265359

      uniform float u_time;
      uniform float u_noise_freq;
      uniform float u_noise_amp;
      uniform vec3 u_mouse_hit;
      uniform float u_mouse_active;

      varying vec2 vUv;
      varying float vNoise;
      varying vec3 vNormal;
      varying float vHover;

      vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
      vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
      vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
      vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

      float snoise(vec3 v) {
        const vec2 C = vec2(1.0/6.0, 1.0/3.0);
        const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

        vec3 i = floor(v + dot(v, C.yyy));
        vec3 x0 = v - i + dot(i, C.xxx);

        vec3 g = step(x0.yzx, x0.xyz);
        vec3 l = 1.0 - g;
        vec3 i1 = min(g.xyz, l.zxy);
        vec3 i2 = max(g.xyz, l.zxy);

        vec3 x1 = x0 - i1 + C.xxx;
        vec3 x2 = x0 - i2 + C.yyy;
        vec3 x3 = x0 - D.yyy;

        i = mod289(i);
        vec4 p = permute(permute(permute(
          i.z + vec4(0.0, i1.z, i2.z, 1.0))
          + i.y + vec4(0.0, i1.y, i2.y, 1.0))
          + i.x + vec4(0.0, i1.x, i2.x, 1.0));

        float n_ = 0.142857142857;
        vec3 ns = n_ * D.wyz - D.xzx;

        vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

        vec4 x_ = floor(j * ns.z);
        vec4 y_ = floor(j - 7.0 * x_);

        vec4 x = x_ * ns.x + ns.yyyy;
        vec4 y = y_ * ns.x + ns.yyyy;
        vec4 h = 1.0 - abs(x) - abs(y);

        vec4 b0 = vec4(x.xy, y.xy);
        vec4 b1 = vec4(x.zw, y.zw);

        vec4 s0 = floor(b0) * 2.0 + 1.0;
        vec4 s1 = floor(b1) * 2.0 + 1.0;
        vec4 sh = -step(h, vec4(0.0));

        vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
        vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

        vec3 p0 = vec3(a0.xy, h.x);
        vec3 p1 = vec3(a0.zw, h.y);
        vec3 p2 = vec3(a1.xy, h.z);
        vec3 p3 = vec3(a1.zw, h.w);

        vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
        p0 *= norm.x;
        p1 *= norm.y;
        p2 *= norm.z;
        p3 *= norm.w;

        vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
        m = m * m;
        return 42.0 * dot(m * m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
      }

      void main() {
        vUv = uv;
        vNormal = normal;
        float noise = snoise(position * u_noise_freq + u_time * 0.5);
        vNoise = noise;
        float dist = distance(position, u_mouse_hit);
        float hover = smoothstep(0.8, 0.0, dist) * u_mouse_active;
        vHover = hover;
        vec3 newPosition = position + normal * (noise * u_noise_amp + hover * 0.4);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
      }
    `

    const fragmentShader = `
      uniform vec3 u_color_main;
      uniform vec3 u_color_accent;
      uniform float u_time;

      varying vec2 vUv;
      varying float vNoise;
      varying vec3 vNormal;
      varying float vHover;

      void main() {
        float mixFactor = smoothstep(-1.0, 1.0, vNoise);
        vec3 color = mix(u_color_main, u_color_accent, mixFactor);
        float rim = 1.0 - max(dot(vNormal, vec3(0.0, 0.0, 1.0)), 0.0);
        rim = smoothstep(0.6, 1.0, rim);
        color += vec3(rim) * 0.5;
        color += u_color_accent * vHover * 1.5;
        float alpha = 0.75 + 0.15 * mixFactor + vHover * 0.4;
        gl_FragColor = vec4(color, alpha);
      }
    `

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100)
    camera.position.z = 5

    const lowPerfMode = window.innerWidth < 1280 || window.devicePixelRatio > 1.8 || m4TrackingActive
    const renderer = new THREE.WebGLRenderer({ antialias: !lowPerfMode, alpha: true, powerPreference: 'high-performance' })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.domElement.style.width = '100%'
    renderer.domElement.style.height = '100%'
    renderer.domElement.style.display = 'block'
    renderer.domElement.style.pointerEvents = 'none'
    container.appendChild(renderer.domElement)

    const uniforms = {
      u_time: { value: 0 },
      u_noise_freq: { value: 0.8 },
      u_noise_amp: { value: 0.15 },
      u_color_main: { value: new THREE.Color('#4169E1') },
      u_color_accent: { value: new THREE.Color('#ffffff') },
      u_mouse_hit: { value: new THREE.Vector3(0, 0, 0) },
      u_mouse_active: { value: 0 },
    }

    const geometry = new THREE.IcosahedronGeometry(1.5, 64)
    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms,
      wireframe: false,
      transparent: true,
      side: THREE.DoubleSide,
    })
    const sphere = new THREE.Mesh(geometry, material)
    const group = new THREE.Group()
    group.add(sphere)
    scene.add(group)

    const raycaster = new THREE.Raycaster()
    const mouse = new THREE.Vector2(-1000, -1000)
    let targetMouseActive = 0
    const clock = new THREE.Clock()

    const target = {
      freq: 0.8,
      amp: 0.15,
      rotSpeed: 0.6,
      colorMain: new THREE.Color('#4169E1'),
      colorAccent: new THREE.Color('#ffffff'),
    }

    m2RendererRef.current = renderer
    m2SceneRef.current = scene
    m2CameraRef.current = camera
    m2SphereRef.current = sphere
    m2GroupRef.current = group
    m2UniformsRef.current = uniforms
    m2RaycasterRef.current = raycaster
    m2MouseRef.current = mouse
    m2TargetRef.current = target

    const resize = () => {
      const rect = container.getBoundingClientRect()
      if (rect.width === 0 || rect.height === 0) {
        return
      }
      renderer.setSize(rect.width, rect.height, false)
      camera.aspect = rect.width / rect.height
      camera.updateProjectionMatrix()
    }

    resize()
    window.addEventListener('resize', resize)

    const handleMouseMove = (event: MouseEvent) => {
      const rect = container.getBoundingClientRect()
      if (rect.width === 0 || rect.height === 0) {
        return
      }
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1
    }

    window.addEventListener('mousemove', handleMouseMove)

    const render = () => {
      const elapsed = clock.getElapsedTime()
      uniforms.u_time.value = elapsed

      raycaster.setFromCamera(mouse, camera)
      const intersects = raycaster.intersectObject(sphere)
      if (intersects.length > 0) {
        targetMouseActive = 1
        const localPoint = sphere.worldToLocal(intersects[0].point.clone())
        uniforms.u_mouse_hit.value.lerp(localPoint, 0.15)
      } else {
        targetMouseActive = 0
      }
      uniforms.u_mouse_active.value += (targetMouseActive - uniforms.u_mouse_active.value) * 0.15

      uniforms.u_noise_freq.value += (target.freq - uniforms.u_noise_freq.value) * 0.05
      uniforms.u_noise_amp.value += (target.amp - uniforms.u_noise_amp.value) * 0.05
      uniforms.u_color_main.value.lerp(target.colorMain, 0.05)
      uniforms.u_color_accent.value.lerp(target.colorAccent, 0.05)

      group.rotation.y += 0.002 * target.rotSpeed
      group.rotation.z += 0.001 * target.rotSpeed
      sphere.rotation.x += (mouse.y * 0.8 - sphere.rotation.x) * 0.08
      sphere.rotation.y += (mouse.x * 0.8 - sphere.rotation.y) * 0.08

      renderer.render(scene, camera)
      m2AnimationRef.current = window.requestAnimationFrame(render)
    }

    m2AnimationRef.current = window.requestAnimationFrame(render)

    return () => {
      if (m2AnimationRef.current !== null) {
        window.cancelAnimationFrame(m2AnimationRef.current)
      }
      window.removeEventListener('resize', resize)
      window.removeEventListener('mousemove', handleMouseMove)
      geometry.dispose()
      material.dispose()
      renderer.dispose()
      if (renderer.domElement.parentElement === container) {
        container.removeChild(renderer.domElement)
      }
      m2RendererRef.current = null
      m2SceneRef.current = null
      m2CameraRef.current = null
      m2SphereRef.current = null
      m2GroupRef.current = null
      m2UniformsRef.current = null
      m2RaycasterRef.current = null
      m2MouseRef.current = null
      m2TargetRef.current = null
    }
  }, [activePage])

  const triggerStateTransition = (analysis: M2AnalysisResult) => {
    const target = m2TargetRef.current
    if (!target) {
      return
    }

    switch (analysis.state) {
      case 'ANXIETY':
        target.freq = 2.5
        target.amp = 0.4
        target.rotSpeed = 1.1
        target.colorMain = new THREE.Color('#4B0082')
        target.colorAccent = new THREE.Color('#8B0000')
        break
      case 'SADNESS':
        target.freq = 0.5
        target.amp = 0.1
        target.rotSpeed = 0.35
        target.colorMain = new THREE.Color('#1A2A40')
        target.colorAccent = new THREE.Color('#4A5A70')
        break
      case 'ANGER':
        target.freq = 2.8
        target.amp = 0.42
        target.rotSpeed = 1.05
        target.colorMain = new THREE.Color('#8B0000')
        target.colorAccent = new THREE.Color('#FF4500')
        break
      case 'JOY':
        target.freq = 1.2
        target.amp = 0.25
        target.rotSpeed = 0.85
        target.colorMain = new THREE.Color('#FFD700')
        target.colorAccent = new THREE.Color('#FFA500')
        break
      case 'EXCITEMENT':
        target.freq = 2.0
        target.amp = 0.45
        target.rotSpeed = 1.25
        target.colorMain = new THREE.Color('#00FFFF')
        target.colorAccent = new THREE.Color('#FFFFFF')
        break
      case 'CALM':
        target.freq = 0.4
        target.amp = 0.08
        target.rotSpeed = 0.3
        target.colorMain = new THREE.Color('#87CEEB')
        target.colorAccent = new THREE.Color('#4169E1')
        break
      default:
        target.freq = 0.8
        target.amp = 0.15
        target.rotSpeed = 0.6
        target.colorMain = new THREE.Color('#2C3E50')
        target.colorAccent = new THREE.Color('#BDC3C7')
    }
  }

  useEffect(() => {
    triggerStateTransition({
      state: m2State === 'NULL' || m2State === 'CALCULATING' ? 'NEUTRAL' : m2State,
      confidence01: m2FlowProfile.confidence,
      confidencePercent: m2Confidence,
      sentiment: m2FlowProfile.sentiment,
      polarity: { pos: 0, neg: 0, neu: 0 },
      keywords: m2Keywords,
      flow: { speed: m2FlowProfile.speed, density: m2FlowProfile.density },
      accepted: true,
      tokens: [],
    })
  }, [m2State, m2FlowProfile, m2Confidence, m2Keywords])

  useEffect(() => {
    if (activePage !== 'M1') {
      return
    }
    const container = m1ContainerRef.current
    if (!container) {
      return
    }

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000)
    camera.position.z = 15

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.domElement.style.width = '100%'
    renderer.domElement.style.height = '100%'
    renderer.domElement.style.display = 'block'
    renderer.domElement.style.pointerEvents = 'none'
    container.appendChild(renderer.domElement)

    const faceGroup = new THREE.Group()
    scene.add(faceGroup)

    const numPoints = 478
    const particlesGeo = new THREE.BufferGeometry()
    const positions = new Float32Array(numPoints * 3)
    particlesGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    const pMaterial = new THREE.PointsMaterial({
      color: 0x87cefa,
      size: 0.1,
      blending: THREE.AdditiveBlending,
      transparent: true,
      opacity: 0.9,
    })
    const particleSystem = new THREE.Points(particlesGeo, pMaterial)
    particleSystem.visible = false
    faceGroup.add(particleSystem)

    const skeletonCount = 68
    const skeletonGeo = new THREE.BufferGeometry()
    const skeletonPositions = new Float32Array(skeletonCount * 3)
    skeletonGeo.setAttribute('position', new THREE.BufferAttribute(skeletonPositions, 3))
    const skeletonMaterial = new THREE.PointsMaterial({
      color: 0xd4af37,
      size: 0.1,
      blending: THREE.AdditiveBlending,
      transparent: true,
      opacity: 0.96,
    })
    const skeletonPoints = new THREE.Points(skeletonGeo, skeletonMaterial)
    skeletonPoints.visible = false
    faceGroup.add(skeletonPoints)

    const skeletonGlowGeo = new THREE.BufferGeometry()
    const skeletonGlowPositions = new Float32Array(skeletonCount * 3)
    skeletonGlowGeo.setAttribute('position', new THREE.BufferAttribute(skeletonGlowPositions, 3))
    const skeletonGlowMaterial = new THREE.PointsMaterial({
      color: 0xd4af37,
      size: 0.24,
      blending: THREE.AdditiveBlending,
      transparent: true,
      opacity: 0.28,
      depthWrite: false,
    })
    const skeletonGlowPoints = new THREE.Points(skeletonGlowGeo, skeletonGlowMaterial)
    skeletonGlowPoints.visible = false
    faceGroup.add(skeletonGlowPoints)

    const maxLines = 10000
    const linePositions = new Float32Array(maxLines * 6)
    const linesGeo = new THREE.BufferGeometry()
    linesGeo.setAttribute('position', new THREE.BufferAttribute(linePositions, 3))
    const lineMaterial = new THREE.LineBasicMaterial({
      color: 0x4169e1,
      blending: THREE.AdditiveBlending,
      transparent: true,
      opacity: 0.25,
    })
    const lineSystem = new THREE.LineSegments(linesGeo, lineMaterial)
    lineSystem.visible = false
    faceGroup.add(lineSystem)

    const posArray = particlesGeo.attributes.position.array as Float32Array
    const skeletonPosArray = skeletonGeo.attributes.position.array as Float32Array
    const skeletonGlowPosArray = skeletonGlowGeo.attributes.position.array as Float32Array

    const updateThree = () => {
      const landmarks = m1CurrentLandmarksRef.current
      if (!landmarks) {
        particleSystem.visible = false
        skeletonPoints.visible = false
        skeletonGlowPoints.visible = false
        lineSystem.visible = false
        linesGeo.setDrawRange(0, 0)
        return
      }
      particleSystem.visible = true
      skeletonPoints.visible = true
      skeletonGlowPoints.visible = true
      lineSystem.visible = true
      const scaleX = 18
      const scaleY = 18
      const scaleZ = 12

      const transformPoint = (lm: { x: number; y: number; z: number }) => {
        let targetX = (lm.x - 0.5) * scaleX
        let targetY = -(lm.y - 0.5) * scaleY
        let targetZ = -lm.z * scaleZ
        if (m1IsDeconstructingRef.current) {
          const noise = (Math.random() - 0.5) * m1DeconstructProgressRef.current * 2
          targetX += noise
          targetY += noise
          targetZ += noise
        }
        return { targetX, targetY, targetZ }
      }

      const upper = Math.min(numPoints, landmarks.length)
      for (let i = 0; i < upper; i += 1) {
        const lm = landmarks[i]
        if (!lm) {
          continue
        }
        const { targetX, targetY, targetZ } = transformPoint(lm)
        posArray[i * 3] += (targetX - posArray[i * 3]) * 0.3
        posArray[i * 3 + 1] += (targetY - posArray[i * 3 + 1]) * 0.3
        posArray[i * 3 + 2] += (targetZ - posArray[i * 3 + 2]) * 0.3
      }
      particlesGeo.attributes.position.needsUpdate = true

      for (let i = 0; i < skeletonCount; i += 1) {
        const source = landmarks[M1_MEDIAPIPE_68_INDICES[i]]
        if (!source) {
          continue
        }
        const { targetX, targetY, targetZ } = transformPoint(source)
        skeletonPosArray[i * 3] += (targetX - skeletonPosArray[i * 3]) * 0.34
        skeletonPosArray[i * 3 + 1] += (targetY - skeletonPosArray[i * 3 + 1]) * 0.34
        skeletonPosArray[i * 3 + 2] += (targetZ - skeletonPosArray[i * 3 + 2]) * 0.34
        skeletonGlowPosArray[i * 3] = skeletonPosArray[i * 3]
        skeletonGlowPosArray[i * 3 + 1] = skeletonPosArray[i * 3 + 1]
        skeletonGlowPosArray[i * 3 + 2] = skeletonPosArray[i * 3 + 2]
      }
      skeletonGeo.attributes.position.needsUpdate = true
      skeletonGlowGeo.attributes.position.needsUpdate = true

      let lineIndex = 0
      const connectThreshold = m1IsDeconstructingRef.current ? 1.5 : 0.8
      for (let i = 0; i < upper; i += 2) {
        for (let j = i + 1; j < upper; j += 2) {
          if (lineIndex >= maxLines) {
            break
          }
          const dx = posArray[i * 3] - posArray[j * 3]
          const dy = posArray[i * 3 + 1] - posArray[j * 3 + 1]
          const dz = posArray[i * 3 + 2] - posArray[j * 3 + 2]
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)
          if (dist < connectThreshold) {
            linePositions[lineIndex * 6] = posArray[i * 3]
            linePositions[lineIndex * 6 + 1] = posArray[i * 3 + 1]
            linePositions[lineIndex * 6 + 2] = posArray[i * 3 + 2]
            linePositions[lineIndex * 6 + 3] = posArray[j * 3]
            linePositions[lineIndex * 6 + 4] = posArray[j * 3 + 1]
            linePositions[lineIndex * 6 + 5] = posArray[j * 3 + 2]
            lineIndex += 1
          }
        }
      }
      linesGeo.setDrawRange(0, lineIndex * 2)
      linesGeo.attributes.position.needsUpdate = true
    }

    const resize = () => {
      const rect = container.getBoundingClientRect()
      if (!rect.width || !rect.height) {
        return
      }
      renderer.setSize(rect.width, rect.height, false)
      camera.aspect = rect.width / rect.height
      camera.updateProjectionMatrix()
    }

    resize()
    window.addEventListener('resize', resize)

    const animate = () => {
      updateThree()
      faceGroup.rotation.y = Math.sin(Date.now() * 0.001) * 0.1
      faceGroup.rotation.x = Math.cos(Date.now() * 0.001) * 0.05
      renderer.render(scene, camera)
      m2AnimationRef.current = window.requestAnimationFrame(animate)
    }
    m2AnimationRef.current = window.requestAnimationFrame(animate)

    return () => {
      if (m2AnimationRef.current !== null) {
        window.cancelAnimationFrame(m2AnimationRef.current)
        m2AnimationRef.current = null
      }
      window.removeEventListener('resize', resize)
      particlesGeo.dispose()
      pMaterial.dispose()
      skeletonGeo.dispose()
      skeletonMaterial.dispose()
      skeletonGlowGeo.dispose()
      skeletonGlowMaterial.dispose()
      linesGeo.dispose()
      lineMaterial.dispose()
      renderer.dispose()
      if (renderer.domElement.parentElement === container) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [activePage])

  useEffect(() => {
    if (activePage !== 'M4' || !m4Activated) {
      return
    }
    const container = m4ContainerRef.current
    if (!container) {
      return
    }

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000)
    const control = m4ControlRef.current
    control.currentZ = control.targetZ
    camera.position.z = control.currentZ

    const lowPerfMode = window.innerWidth < 1280 || window.devicePixelRatio > 1.8 || m4TrackingActive
    const renderer = new THREE.WebGLRenderer({ antialias: !lowPerfMode, alpha: true, powerPreference: 'high-performance' })
    renderer.setPixelRatio(window.devicePixelRatio)
    renderer.domElement.style.width = '100%'
    renderer.domElement.style.height = '100%'
    renderer.domElement.style.display = 'block'
    renderer.domElement.style.pointerEvents = 'none'
    container.appendChild(renderer.domElement)

    const particleGroup = new THREE.Group()
    scene.add(particleGroup)

    const coreColor = new THREE.Color(m4CoreParticleColorMap[m4BoneElement])
    const ringColor = new THREE.Color(resolveM4RingColorByEmotion(m2State))
    const fracturedRing = m2State === 'ANXIETY' || m2State === 'ANGER' || m2State === 'SADNESS'

    const coreGeo = new THREE.BufferGeometry()
    const coreCount = lowPerfMode ? 1800 : 2900
    const corePos = new Float32Array(coreCount * 3)
    const coreCol = new Float32Array(coreCount * 3)

    for (let i = 0; i < coreCount; i += 1) {
      const radius = 70 * Math.cbrt(Math.random())
      const theta = Math.random() * 2 * Math.PI
      const phi = Math.acos(2 * Math.random() - 1)
      corePos[i * 3] = radius * Math.sin(phi) * Math.cos(theta)
      corePos[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta)
      corePos[i * 3 + 2] = radius * Math.cos(phi)
      const mixed = coreColor.clone().multiplyScalar(0.92 + Math.random() * 0.1)
      coreCol[i * 3] = mixed.r
      coreCol[i * 3 + 1] = mixed.g
      coreCol[i * 3 + 2] = mixed.b
    }

    coreGeo.setAttribute('position', new THREE.BufferAttribute(corePos, 3))
    coreGeo.setAttribute('color', new THREE.BufferAttribute(coreCol, 3))

    const particleCanvas = document.createElement('canvas')
    particleCanvas.width = 128
    particleCanvas.height = 128
    const particleCtx = particleCanvas.getContext('2d')
    if (!particleCtx) {
      return
    }
    const center = particleCanvas.width / 2
    const radius = particleCanvas.width / 2
    const gradient = particleCtx.createRadialGradient(center, center, 0, center, center, radius)
    gradient.addColorStop(0, 'rgba(255, 255, 255, 1)')
    gradient.addColorStop(0.2, 'rgba(255, 255, 255, 1)')
    gradient.addColorStop(0.4, 'rgba(255, 255, 255, 0.4)')
    gradient.addColorStop(1, 'rgba(255, 255, 255, 0)')
    particleCtx.fillStyle = gradient
    particleCtx.fillRect(0, 0, particleCanvas.width, particleCanvas.height)
    const particleTexture = new THREE.CanvasTexture(particleCanvas)
    particleTexture.generateMipmaps = false
    particleTexture.minFilter = THREE.LinearFilter
    particleTexture.magFilter = THREE.LinearFilter
    particleTexture.needsUpdate = true

    const coreMat = new THREE.PointsMaterial({
      size: 2.9,
      vertexColors: true,
      blending: THREE.AdditiveBlending,
      transparent: true,
      opacity: 0.82,
      depthWrite: true,
      alphaTest: 0.18,
      sizeAttenuation: true,
      map: particleTexture,
    })
    const coreParticles = new THREE.Points(coreGeo, coreMat)
    coreParticles.renderOrder = 1
    particleGroup.add(coreParticles)

    const ringCount = lowPerfMode ? 1200 : 2100
    const ringTilt = Math.PI / 3
    const ringBackPos = new Float32Array(ringCount * 3)
    const ringFrontPos = new Float32Array(ringCount * 3)
    const ringRadius = 90
    const segmentCount = fracturedRing ? 4 : 1
    const segmentArc = (Math.PI * 2) / segmentCount
    const visibleArc = fracturedRing ? segmentArc * 0.62 : segmentArc * 0.96
    let backWriteIndex = 0
    let frontWriteIndex = 0
    for (let segment = 0; segment < segmentCount; segment += 1) {
      const segmentStart = segment * segmentArc
      const perSegment = Math.floor(ringCount / segmentCount)
      for (let i = 0; i < perSegment; i += 1) {
        const progress = perSegment <= 1 ? 0 : i / (perSegment - 1)
        const angle = segmentStart + progress * visibleArc
        const radius = ringRadius + (Math.random() - 0.5) * 1.6
        const x = Math.cos(angle) * radius
        const y = (Math.random() - 0.5) * 0.18
        const z = Math.sin(angle) * radius
        const zAfterTilt = y * Math.sin(ringTilt) + z * Math.cos(ringTilt)
        if (zAfterTilt >= 0) {
          ringFrontPos[frontWriteIndex * 3] = x
          ringFrontPos[frontWriteIndex * 3 + 1] = y
          ringFrontPos[frontWriteIndex * 3 + 2] = z
          frontWriteIndex += 1
        } else {
          ringBackPos[backWriteIndex * 3] = x
          ringBackPos[backWriteIndex * 3 + 1] = y
          ringBackPos[backWriteIndex * 3 + 2] = z
          backWriteIndex += 1
        }
      }
    }
    const ringBackGeo = new THREE.BufferGeometry()
    ringBackGeo.setAttribute('position', new THREE.BufferAttribute(ringBackPos, 3))
    ringBackGeo.setDrawRange(0, backWriteIndex)
    const ringFrontGeo = new THREE.BufferGeometry()
    ringFrontGeo.setAttribute('position', new THREE.BufferAttribute(ringFrontPos, 3))
    ringFrontGeo.setDrawRange(0, frontWriteIndex)
    const ringBackMat = new THREE.PointsMaterial({
      color: ringColor,
      size: 1.4,
      transparent: true,
      opacity: fracturedRing ? 0.32 : 0.46,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      depthTest: true,
      alphaTest: 0.18,
      sizeAttenuation: true,
      map: particleTexture,
    })
    const ringFrontMat = ringBackMat.clone()
    ringFrontMat.opacity = fracturedRing ? 0.78 : 0.92
    const ringBackParticles = new THREE.Points(ringBackGeo, ringBackMat)
    const ringFrontParticles = new THREE.Points(ringFrontGeo, ringFrontMat)
    ringBackParticles.rotation.x = ringTilt
    ringFrontParticles.rotation.x = ringTilt
    ringBackParticles.renderOrder = 0
    ringFrontParticles.renderOrder = 2
    particleGroup.add(ringBackParticles)
    particleGroup.add(ringFrontParticles)

    const farGeo = new THREE.BufferGeometry()
    const farCount = lowPerfMode ? 260 : 420
    const farPos = new Float32Array(farCount * 3)
    for (let i = 0; i < farCount; i += 1) {
      const radius = 180 + Math.random() * 140
      const theta = Math.random() * 2 * Math.PI
      const phi = Math.acos(2 * Math.random() - 1)
      farPos[i * 3] = radius * Math.sin(phi) * Math.cos(theta)
      farPos[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta)
      farPos[i * 3 + 2] = radius * Math.cos(phi)
    }
    farGeo.setAttribute('position', new THREE.BufferAttribute(farPos, 3))
    const farMat = new THREE.PointsMaterial({
      color: new THREE.Color('#d5ecff'),
      size: 0.92,
      transparent: true,
      opacity: 0.34,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      alphaTest: 0.1,
      sizeAttenuation: true,
      map: particleTexture,
    })
    const farParticles = new THREE.Points(farGeo, farMat)
    particleGroup.add(farParticles)

    const nebulaGeo = new THREE.BufferGeometry()
    const nebulaCount = lowPerfMode ? 90 : 160
    const nebulaPos = new Float32Array(nebulaCount * 3)
    const nebulaCol = new Float32Array(nebulaCount * 3)
    const nebulaPalette = ['#87CEFA', '#9FD3FF', '#D4AF37', '#7EC8FF']
    for (let i = 0; i < nebulaCount; i += 1) {
      const radius = 120 + Math.random() * 150
      const theta = Math.random() * 2 * Math.PI
      const phi = Math.acos(2 * Math.random() - 1)
      nebulaPos[i * 3] = radius * Math.sin(phi) * Math.cos(theta)
      nebulaPos[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta)
      nebulaPos[i * 3 + 2] = radius * Math.cos(phi)
      const color = new THREE.Color(nebulaPalette[i % nebulaPalette.length]).multiplyScalar(0.72 + Math.random() * 0.22)
      nebulaCol[i * 3] = color.r
      nebulaCol[i * 3 + 1] = color.g
      nebulaCol[i * 3 + 2] = color.b
    }
    nebulaGeo.setAttribute('position', new THREE.BufferAttribute(nebulaPos, 3))
    nebulaGeo.setAttribute('color', new THREE.BufferAttribute(nebulaCol, 3))
    const nebulaMat = new THREE.PointsMaterial({
      size: 5.8,
      vertexColors: true,
      transparent: true,
      opacity: 0.14,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      alphaTest: 0.06,
      sizeAttenuation: true,
      map: particleTexture,
    })
    const nebulaParticles = new THREE.Points(nebulaGeo, nebulaMat)
    particleGroup.add(nebulaParticles)

    const resize = () => {
      const rect = container.getBoundingClientRect()
      if (!rect.width || !rect.height) {
        return
      }
      renderer.setSize(rect.width, rect.height, false)
      camera.aspect = rect.width / rect.height
      camera.updateProjectionMatrix()
    }

    resize()
    window.addEventListener('resize', resize)

    let isDragging = false
    let previous = { x: 0, y: 0 }

    const handleDown = () => {
      isDragging = true
    }
    const handleUp = () => {
      isDragging = false
    }
    const handleMove = (event: MouseEvent) => {
      if (!isDragging || m4TrackingActive) {
        previous = { x: event.offsetX, y: event.offsetY }
        return
      }
      const deltaMove = { x: event.offsetX - previous.x, y: event.offsetY - previous.y }
      control.targetRotY += deltaMove.x * 0.01
      control.targetRotX += deltaMove.y * 0.01
      previous = { x: event.offsetX, y: event.offsetY }
      setM4GestureLabel('MOUSE OVERRIDE')
    }
    const handleWheel = (event: WheelEvent) => {
      if (m4TrackingActive) {
        return
      }
      control.targetZ += event.deltaY * 0.1
      control.targetZ = Math.max(60, Math.min(control.targetZ, 400))
      setM4GestureLabel('MOUSE OVERRIDE')
    }

    window.addEventListener('mousedown', handleDown)
    window.addEventListener('mouseup', handleUp)
    window.addEventListener('mousemove', handleMove)
    window.addEventListener('wheel', handleWheel, { passive: true })

    const animate = () => {
      const t = performance.now() * 0.001
      const breathScale = 1 + Math.sin(t * 1.2) * 0.05
      ringBackParticles.rotation.z += 0.0018
      ringFrontParticles.rotation.z += 0.0018
      coreParticles.rotation.y += 0.0014
      coreParticles.rotation.x += 0.0008
      farParticles.rotation.y -= 0.00045
      farParticles.rotation.x += 0.0002
      nebulaParticles.rotation.y += 0.00016
      nebulaParticles.rotation.z -= 0.00011

      const lerpSpeed = Math.abs(control.targetZ - control.currentZ) > 20 ? 0.05 : 0.02
      control.currentZ += (control.targetZ - control.currentZ) * lerpSpeed
      camera.position.z = control.currentZ

      control.currentRotX += (control.targetRotX - control.currentRotX) * 0.1
      control.currentRotY += (control.targetRotY - control.currentRotY) * 0.1

      let diffZ = control.targetRotZ - control.currentRotZ
      while (diffZ < -Math.PI) diffZ += Math.PI * 2
      while (diffZ > Math.PI) diffZ -= Math.PI * 2
      control.currentRotZ += diffZ * 0.1

      control.currentScale += (control.targetScale - control.currentScale) * 0.08
      control.currentBrightness += (control.targetBrightness - control.currentBrightness) * 0.1

      const finalScale = control.currentScale * breathScale
      coreParticles.scale.set(finalScale, finalScale, finalScale)
      const breathAlpha = 0.96 + Math.sin(t * 1.2) * 0.04
      coreMat.opacity = 0.82 * breathAlpha * control.currentBrightness
      ringBackMat.opacity = (fracturedRing ? 0.42 : 0.54) * (0.95 + Math.sin(t * 1.1) * 0.05) * control.currentBrightness
      ringFrontMat.opacity = (fracturedRing ? 0.68 : 0.82) * (0.95 + Math.sin(t * 1.1) * 0.05) * control.currentBrightness

      particleGroup.rotation.x = control.currentRotX
      particleGroup.rotation.y = control.currentRotY
      particleGroup.rotation.z = control.currentRotZ

      renderer.render(scene, camera)
      m4AnimationRef.current = window.requestAnimationFrame(animate)
    }

    m4AnimationRef.current = window.requestAnimationFrame(animate)

    return () => {
      if (m4AnimationRef.current !== null) {
        window.cancelAnimationFrame(m4AnimationRef.current)
        m4AnimationRef.current = null
      }
      window.removeEventListener('resize', resize)
      window.removeEventListener('mousedown', handleDown)
      window.removeEventListener('mouseup', handleUp)
      window.removeEventListener('mousemove', handleMove)
      window.removeEventListener('wheel', handleWheel)
      coreGeo.dispose()
      coreMat.dispose()
      ringBackGeo.dispose()
      ringBackMat.dispose()
      ringFrontGeo.dispose()
      ringFrontMat.dispose()
      farGeo.dispose()
      farMat.dispose()
      nebulaGeo.dispose()
      nebulaMat.dispose()
      particleTexture.dispose()
      renderer.dispose()
      if (renderer.domElement.parentElement === container) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [activePage, m4Activated, m4TrackingActive, m4Energy.volitionRatio, m2State, m4BoneElement])

  const t = {
    scan: {
      phase: lang === 'cn' ? 'Phase 00 : 仪式' : 'Phase 00 : The Ritual',
      titleLine1: lang === 'cn' ? '献出你的' : 'Offer Your',
      titleLine2: lang === 'cn' ? '映像' : 'Reflection',
      desc:
        lang === 'cn'
          ? '凝视镜头。留下你此刻的心境。物理坐标与文字流墨即将在此刻坍缩为你的数字宿命。'
          : 'Gaze into the lens. Leave your current state of mind. Coordinates and text flow collapse into your digital fate.',
      placeholder: lang === 'cn' ? '描述你此刻的心境...' : 'Describe your current state of mind...',
      initiate: lang === 'cn' ? '启动采样' : 'Initiate Capture',
      extracting: lang === 'cn' ? '提取_68_节点...' : 'EXTRACTING_68_NODES...',
      parsing: lang === 'cn' ? '解析_语义...' : 'PARSING_SEMANTICS...',
      ready: lang === 'cn' ? '准备进入' : 'Ready to Proceed',
      camera: lang === 'cn' ? '摄像' : 'CAMERA',
      capture: lang === 'cn' ? '捕捉' : 'CAPTURE',
      upload: lang === 'cn' ? '上传' : 'UPLOAD',
      close: lang === 'cn' ? '关闭' : 'CLOSE',
      uploadId: lang === 'cn' ? '上传编号' : 'UPLOAD ID',
      status: lang === 'cn' ? '状态' : 'STATUS',
    },
    m1: {
      header: lang === 'cn' ? 'M1 // 几何拓扑 // 拓扑提取' : 'M1 // The Geometry // Topological Extraction',
      title: lang === 'cn' ? '几何拓扑' : 'Geometry',
      sub: lang === 'cn' ? '解剖蓝图' : 'Anatomical Blueprint',
      desc:
        lang === 'cn'
          ? '将面容解构为客观的拓扑坐标。支持向量机 (SVM) 在此寻找潜藏在三庭五眼中的古典黄金分割与五行隐喻。'
          : 'Deconstruct the face into objective topological coordinates. SVM seeks classical golden ratios and five-element metaphors hidden within facial proportions.',
      angle: lang === 'cn' ? '提取角度' : 'Extracted Angle',
      archetype: lang === 'cn' ? 'SVM 原型' : 'SVM Archetype',
      capture: lang === 'cn' ? '骨相采集' : 'GEOMETRY CAPTURE',
      modeRealtime: lang === 'cn' ? '实时视界' : 'REAL-TIME',
      modeStatic: lang === 'cn' ? '静态矩阵' : 'STATIC',
      result: lang === 'cn' ? '结果' : 'RESULT',
      rowElement: lang === 'cn' ? '五行元素' : 'WU XING ELEMENT',
      rowAngle: lang === 'cn' ? '下颌角' : 'MANDIBULAR ANGLE',
      rowZygomatic: lang === 'cn' ? '颧骨结构' : 'ZYGOMATIC STRUCT',
      enterM2: lang === 'cn' ? '进入 M2：语义剖面' : 'ENTER M2: THE PROFILER',
      extract: lang === 'cn' ? '开始拓扑解构' : 'START TOPOLOGY EXTRACTION',
    },
    m2: {
      header: lang === 'cn' ? 'M2 // 语义溢散 // 心境剖面' : 'M2 // The Profiler // Semantic Smudging',
      stream: lang === 'cn' ? '意识流' : 'Consciousness Stream',
      streamHint:
        lang === 'cn'
          ? '准备就绪。请暴露你的内在状态，算法已就位。'
          : 'Ready and waiting. Expose your inner state to the algorithm.',
      confidence: lang === 'cn' ? '置信度' : 'Confidence Rate',
      state: lang === 'cn' ? '识别状态' : 'State Identified',
      awaiting: lang === 'cn' ? '等待' : 'AWAITING',
      inputPlaceholder: lang === 'cn' ? '你当下的心境是怎样的？（开心/焦虑/…）' : 'Describe your current state of mind...',
      moduleLabel: lang === 'cn' ? '自然语言处理模块' : 'Natural Language Processing Module',
      faceframe: lang === 'cn' ? 'FACEFRAME' : 'FACEFRAME',
      enterM3: lang === 'cn' ? '进入意志夺权' : 'ENTER M3: WILL OVERRIDE',
    },
    m3: {
      header: lang === 'cn' ? 'M3 // 人格锻造 // 琉璃变形' : 'M3 // The Persona // Glass Metamorphosis',
      left: lang === 'cn' ? '血肉剥离后，面相的倒影在此凝结为你专属的琉璃幻兽。' : 'After the flesh is stripped away, a glassbound totem of you condenses here.',
      right:
        lang === 'cn'
          ? '介入下方刻度，击碎代码偏见的毛玻璃，将原生灵魂从算法的迷雾中赎回。'
          : 'Intervene on the scale below to shatter algorithmic bias and reclaim the original soul.',
      identity: lang === 'cn' ? '你的身份：???' : 'Your Identity: ???',
      algorithm: lang === 'cn' ? '算法' : 'ALGORITHM',
      freeWill: lang === 'cn' ? '自由意志' : 'FREE WILL',
      currentState: lang === 'cn' ? '当前状态' : 'CURRENT STATE',
      stateSealed: lang === 'cn' ? '算法封印' : 'ALGO SEALED',
      stateHalf: lang === 'cn' ? '半自由' : 'PARTIAL FREE',
      stateFree: lang === 'cn' ? '完全自由' : 'FULL FREE',
      manifest: lang === 'cn' ? '确认夺权，生成宿命' : 'CONFIRM OVERRIDE, GENERATE FATE',
      loadingMask: lang === 'cn' ? '专属图腾凝结中' : 'TOTEM CONDENSING',
      generatedTotem: lang === 'cn' ? '生成图腾' : 'GENERATED_TOTEM',
    },
    m4: {
      header: lang === 'cn' ? 'M4 // 宿命场 // 关系蚀变' : 'M4 // The Destiny // Relational Eclipse',
      title: lang === 'cn' ? '宿命日食' : 'Destiny',
      sub: lang === 'cn' ? '关系日食' : 'Relational Eclipse',
      desc:
        lang === 'cn'
          ? '当异质的矩阵对齐。这是从深空虚无中涌现的、潜意识伴侣的能量投射。'
          : 'When heterogeneous matrices align, a projection of the subconscious companion emerges from deep space.',
      tracking: lang === 'cn' ? '[ 手势追踪已激活 ]' : '[ HAND TRACKING ACTIVE ]',
      activate: lang === 'cn' ? '激活' : 'ACTIVATE',
      instructions: lang === 'cn' ? '张手：释放 · 握拳：收束' : 'OPEN PALM: release · FIST: hold',
      manifest: lang === 'cn' ? '显化卡片' : 'MANIFEST CARD',
      resonance: lang === 'cn' ? '共鸣' : 'Resonance',
      gesture: lang === 'cn' ? '手势' : 'GESTURE',
      confidence: lang === 'cn' ? '置信度' : 'CONF',
      collapseTitle: lang === 'cn' ? 'M4 // 宿命场 // 坍缩' : 'M4 // THE DESTINY // COLLAPSE',
      activateField: lang === 'cn' ? '[ ACTIVATE / 激活场域 ]' : '[ ACTIVATE / FIELD ]',
      camHint:
        lang === 'cn'
          ? '* 请允许浏览器访问摄像头。如果由于 iframe 限制无法打开摄像头，将自动降级为鼠标交互（拖拽旋转，滚轮缩放）。'
          : '* Please allow camera access. If iframe policy blocks camera, it falls back to mouse interaction (drag to rotate, wheel to zoom).',
      enterObserver: lang === 'cn' ? '输入观察者 ID' : 'ENTER OBSERVER ID',
      cancel: lang === 'cn' ? '取消' : 'CANCEL',
      confirm: lang === 'cn' ? '确认' : 'CONFIRM',
      energyConvergence: lang === 'cn' ? '能量收束度' : 'ENERGY CONVERGENCE',
      entityControl: lang === 'cn' ? '实体控制' : 'Entity Control',
      weightBone: lang === 'cn' ? '骨相 60%' : 'BONE 60%',
      weightMood: lang === 'cn' ? '情绪 30%' : 'MOOD 30%',
      weightWill: lang === 'cn' ? '意志 10%' : 'WILL 10%',
      ringDispersion: lang === 'cn' ? '星环离散' : 'RING DISPERSION',
      generateBonus: lang === 'cn' ? '相生增益' : 'GENERATE BONUS',
      overcomePenalty: lang === 'cn' ? '相克抑制' : 'OVERCOME PENALTY',
      algorithmBias: lang === 'cn' ? '算法偏置' : 'ALGO BIAS',
      willRewrite: lang === 'cn' ? '意志改写' : 'WILL REWRITE',
      ringContinuous: lang === 'cn' ? '完整连续' : 'CONTINUOUS',
      ringLightBreak: lang === 'cn' ? '轻度断裂' : 'LIGHT FRACTURE',
      ringMediumBreak: lang === 'cn' ? '中度断裂' : 'MEDIUM FRACTURE',
      ringHeavyBreak: lang === 'cn' ? '高离散坍缩' : 'HIGH DISPERSION',
      tipOpen: lang === 'cn' ? '张开: 靠近' : 'OPEN: APPROACH',
      tipFist: lang === 'cn' ? '握拳: 远离' : 'FIST: RETREAT',
      tipRotate: lang === 'cn' ? '旋转: 拨动时间' : 'ROTATE: SHIFT TIME',
    },
    engine: {
      title: lang === 'cn' ? '引擎' : 'The Engine',
      loading: lang === 'cn' ? '正在加载节点阈值矩阵...' : 'Loading node threshold matrix...',
    },
  }
  const m1TypeDisplay =
    lang === 'cn'
      ? ({ METAL: '金型', WOOD: '木型', WATER: '水型', FIRE: '火型', EARTH: '土型' }[m1ResultType.split(' ')[0]] ?? m1ResultType)
      : m1ResultType
  const updateLogicPanel = useCallback(() => {
    const moduleId: 'M1' | 'M2' | 'M3' | 'M4' = activePage === 'M1' || activePage === 'M2' || activePage === 'M3' || activePage === 'M4' ? activePage : 'M1'
    const m2AnchorCount = m2SemanticCategories.reduce((sum, category) => sum + m2EmotionLexicon[category].size, 0)
    const m1Landmarks = m1CurrentLandmarksRef.current
    const canthusRatio =
      m1Landmarks && m1Landmarks[133] && m1Landmarks[362] && m1Landmarks[234] && m1Landmarks[454]
        ? (MathUtils.distance(m1Landmarks[133], m1Landmarks[362]) / Math.max(MathUtils.distance(m1Landmarks[234], m1Landmarks[454]), 0.0001)).toFixed(4)
        : '--'
    const agencyRatio = Math.max(0, Math.min(1, agencyValue / 100))
    const grayscale = (0.2 + (1 - agencyRatio) * 0.5).toFixed(3)
    const saturation = (0.92 + agencyRatio * 0.35).toFixed(3)
    const contrast = (1 + agencyRatio * 0.12).toFixed(3)
    const brightness = (0.98 + agencyRatio * 0.06).toFixed(3)
    const ringColor = resolveM4RingColorByEmotion(m2State)
    const decoupleLines = [
      'SENSOR (Data Capture): MediaPipe Face Mesh',
      'Type: Deep Learning (Black Box)',
      'Status: [VISUAL ONLY / LOGIC BYPASSED] - 410 redundant nodes discarded.',
      'ENGINE (Decision Logic): 68-Point Geometric Reduction',
      'Type: Traditional AI Expert System (White Box)',
      'Status: [ACTIVE / DECISION MAKER]',
      'CRITIQUE: We intentionally decouple sensor and logic. 478 neural landmarks are reduced to 68 rigid boundaries for deterministic judgment.',
    ]
    const moduleLines: Record<'M1' | 'M2' | 'M3' | 'M4', string[]> = {
      M1: [
        ...decoupleLines,
        `Logic: ${logicData.M1.logic}`,
        `Data: ${logicData.M1.data}`,
        `Rule: ${logicData.M1.rule}`,
        `Mapped Value / Jaw_Angle: ${m1ResultAngle}`,
        `Mapped Value / EyeSpan_Ratio(133-362 / 234-454): ${canthusRatio}`,
        `Mapped Value / WuXing_Output: ${m1TypeDisplay}`,
      ],
      M2: [
        `Logic: ${logicData.M2.logic}`,
        `Process: ${logicData.M2.data}`,
        `Output Rule: ${logicData.M2.rule}`,
        `Mapped Value / Lexicon_Anchor_Count: ${m2AnchorCount}`,
        `Mapped Value / Text_Log_Count: ${m2Logs.length}`,
        `Mapped Value / Matched_Keywords: ${m2Keywords.length} [${m2Keywords.join(', ') || '--'}]`,
        `Mapped Value / Emotion_State: ${m2State} @ ${m2Confidence.toFixed(1)}%`,
      ],
      M3: [
        `Logic: ${logicData.M3.logic}`,
        `Mechanism: ${logicData.M3.data}`,
        `Visual Rule: ${logicData.M3.rule}`,
        `Mapped Value / Human_Agency: ${agencyValue.toFixed(1)} (${(agencyRatio * 100).toFixed(1)}%)`,
        `Mapped Value / Persona_Grayscale: ${grayscale}`,
        `Mapped Value / Persona_Saturate: ${saturation}`,
        `Mapped Value / Persona_Contrast_Brightness: ${contrast} / ${brightness}`,
      ],
      M4: [
        `Logic: ${logicData.M4.logic}`,
        `Formula: ${logicData.M4.data}`,
        `Export: ${logicData.M4.rule}`,
        `Mapped Value / Weights: bone=${m4Energy.algorithmWeight}, mood=${m4Energy.moodWeight}, will=${m4Energy.willWeight}`,
        `Mapped Value / M1(Bone->Color): ${m4BoneElement.toUpperCase()} -> ${m4CoreParticleColorMap[m4BoneElement]}`,
        `Mapped Value / M2(State->Ring): ${m2State} -> ring=${ringColor}, break=${(m4Energy.ringBreakRatio * 100).toFixed(1)}%`,
        `Mapped Value / M3(Agency->Bias): ${agencyValue.toFixed(1)} -> algo_bias=+${m4Energy.explain.algorithmBiasDelta.toFixed(3)}, will_bias=+${m4Energy.explain.freedomBiasDelta.toFixed(3)}`,
        `Mapped Value / ShengKe: +${m4Energy.explain.generateAdded.toFixed(3)} / -${m4Energy.explain.overcomeReduced.toFixed(3)}`,
        `Mapped Value / Final Universe: dominant=${m4Energy.dominant.toUpperCase()}, confidence=${m4Energy.confidence.toFixed(1)}%`,
        `Mapped Value / Export_Params: html2canvas(bg=#050505,useCORS=true,scale=${Math.min(2, window.devicePixelRatio || 1.5).toFixed(2)})`,
      ],
    }
    setLogicPanelContent({ moduleId, lines: moduleLines[moduleId] })
  }, [activePage, agencyValue, m1ResultAngle, m1TypeDisplay, m2Confidence, m2Keywords, m2Logs.length, m2State, m4BoneElement, m4Energy])
  const toggleLogicOverlay = () => {
    const overlay = logicOverlayRef.current
    if (!overlay) {
      return
    }
    const isActive = overlay.classList.toggle('active')
    setEngineOpen(isActive)
    if (isActive) {
      updateLogicPanel()
    }
  }
  useEffect(() => {
    if (engineOpen) {
      updateLogicPanel()
    }
  }, [engineOpen, updateLogicPanel])

  const renderScanPage = () => (
    <div className="h-full w-full animate-in fade-in duration-[1500ms]">
      <div className="flex h-full w-full flex-col items-center justify-center px-12 lg:flex-row lg:px-32">
        <div className="z-10 flex w-full flex-col pr-0 lg:w-1/2 lg:pr-20">
          <div className="mb-8 flex items-center gap-4">
            <div className="h-[1px] w-12 bg-[#d4af37]" />
            <span className="font-mono text-[9px] tracking-[0.5em] text-[#d4af37] uppercase">{t.scan.phase}</span>
          </div>
          <h1 className="mb-8 font-serif text-6xl leading-none text-[#EAEAEA] lg:text-8xl">
            {t.scan.titleLine1} <br />
            <span className="opacity-80 italic">{t.scan.titleLine2}</span>
          </h1>
          <p className="mb-16 max-w-md border-l border-[#333] pl-6 text-xs leading-loose font-light tracking-widest text-[#888]">
            {t.scan.desc}
          </p>

          <div className="group relative w-full max-w-md border-b border-[#333] pb-4">
            <div className="absolute bottom-0 left-0 h-[1px] w-0 bg-[#06b6d4] transition-all duration-700 group-focus-within:w-full group-hover:w-full" />
            <input
              type="text"
              placeholder={t.scan.placeholder}
              value={moodText}
              onChange={(event) => setMoodText(event.target.value)}
              disabled={scanState === 'scanning'}
              className="w-full bg-transparent font-serif text-2xl italic tracking-wide text-[#EAEAEA] placeholder-[#444] outline-none disabled:opacity-50"
              onFocus={() => setGazeMode('analyzing')}
              onBlur={() => setGazeMode('idle')}
            />
          </div>
          {scanError && <p className="mt-4 text-xs tracking-wide text-rose-300">{scanError}</p>}
          {uploadResult && (
            <p className="mt-4 text-xs tracking-wide text-[#d4af37]">
              {t.scan.uploadId}: {uploadResult.uploadId ?? 'N/A'} · {t.scan.status}: {uploadResult.status ?? 'OK'}
            </p>
          )}
        </div>

        <div className="relative mt-12 flex h-[50vh] w-full items-center justify-center lg:mt-0 lg:h-full lg:w-1/2">
          <div
            className={`relative h-[400px] w-[300px] transition-all duration-1000 lg:h-[480px] lg:w-[380px] ${
              scanState === 'scanning'
                ? 'scale-105 shadow-[0_0_50px_rgba(6,182,212,0.1)]'
                : 'scale-100'
            }`}
          >
            <div className="absolute left-0 top-0 h-8 w-8 border-l-2 border-t-2 border-[#555]" />
            <div className="absolute right-0 top-0 h-8 w-8 border-r-2 border-t-2 border-[#555]" />
            <div className="absolute bottom-0 left-0 h-8 w-8 border-b-2 border-l-2 border-[#555]" />
            <div className="absolute bottom-0 right-0 h-8 w-8 border-b-2 border-r-2 border-[#555]" />

            <div
              className="absolute inset-4 flex cursor-pointer items-center justify-center overflow-hidden border border-white/5 bg-black/40 backdrop-blur-md"
              onClick={handleScan}
              onMouseEnter={() => setGazeMode('analyzing')}
              onMouseMove={(event) => {
                const rect = event.currentTarget.getBoundingClientRect()
                const dx = (event.clientX - rect.left) / rect.width - 0.5
                const dy = (event.clientY - rect.top) / rect.height - 0.5
                setScanTotemTilt({ x: dy * 8, y: -dx * 12 })
              }}
              onMouseLeave={() => {
                setGazeMode('idle')
                setScanTotemTilt({ x: 0, y: 0 })
              }}
            >
              {imagePreview && (
                <img src={imagePreview} alt="scan-source" className="absolute inset-0 h-full w-full object-cover opacity-30" />
              )}
              {cameraOpen && !imagePreview && (
                <video ref={videoRef} autoPlay muted playsInline className="absolute inset-0 h-full w-full object-cover opacity-35" />
              )}
              {scanState === 'idle' && (
                <div
                  className="flex flex-col items-center gap-3 opacity-60 transition-opacity duration-500 hover:opacity-100"
                  style={{ transform: `perspective(700px) rotateX(${scanTotemTilt.x}deg) rotateY(${scanTotemTilt.y}deg)` }}
                >
                  <div className="flex items-center gap-3">
                    <Maximize size={24} strokeWidth={1} className="text-[#EAEAEA]" />
                    <Fingerprint size={22} strokeWidth={1} className="text-[#888]" />
                  </div>
                  <span className="font-mono text-[10px] tracking-[0.4em] text-[#888] uppercase">{t.scan.initiate}</span>
                </div>
              )}

              {scanState === 'scanning' && (
                <>
                  <div className="absolute left-0 top-0 h-[2px] w-full animate-[scan_2s_ease-in-out_infinite] bg-cyan-400 shadow-[0_0_20px_#06b6d4]" />
                  <div className="absolute inset-0 animate-pulse bg-[radial-gradient(circle_at_center,rgba(6,182,212,0.1)_0%,transparent_70%)]" />
                  <Crosshair
                    size={64}
                    strokeWidth={0.5}
                    className="text-[#06b6d4] opacity-80 animate-[spin_4s_linear_infinite]"
                  />
                  <div className="absolute bottom-6 left-6 flex flex-col gap-2 font-mono text-[9px] tracking-widest text-[#06b6d4] opacity-80">
                    <span>{t.scan.extracting}</span>
                    <span>[||||||||||] 89%</span>
                    <span>{t.scan.parsing}</span>
                  </div>
                </>
              )}

              {scanState === 'complete' && (
                <div className="flex flex-col items-center gap-6">
                  <div className="flex h-20 w-20 items-center justify-center rounded-full border border-[#d4af37] bg-[#d4af37]/10 shadow-[0_0_40px_rgba(212,175,55,0.3)]">
                    <ArrowRight size={32} strokeWidth={1} className="text-[#d4af37]" />
                  </div>
                  <span className="font-mono text-[10px] tracking-[0.4em] text-[#d4af37] uppercase">{t.scan.ready}</span>
                </div>
              )}
            </div>
            <div className="absolute left-4 right-4 bottom-4 z-20 flex items-center justify-between gap-2 border border-white/10 bg-black/55 px-3 py-2 backdrop-blur-md">
              <button
                type="button"
                onClick={() => void openCamera()}
                className="font-mono text-[9px] tracking-[0.25em] text-[#9bc6d4] uppercase hover:text-[#06b6d4]"
                onMouseEnter={() => setGazeMode('analyzing')}
                onMouseLeave={() => setGazeMode('idle')}
              >
                {t.scan.camera}
              </button>
              <button
                type="button"
                onClick={captureFromCamera}
                className="font-mono text-[9px] tracking-[0.25em] text-[#c9c9c9] uppercase hover:text-white"
                onMouseEnter={() => setGazeMode('analyzing')}
                onMouseLeave={() => setGazeMode('idle')}
              >
                {t.scan.capture}
              </button>
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="font-mono text-[9px] tracking-[0.25em] text-[#d4af37] uppercase hover:text-[#f5d27b]"
                onMouseEnter={() => setGazeMode('analyzing')}
                onMouseLeave={() => setGazeMode('idle')}
              >
                {t.scan.upload}
              </button>
              {cameraOpen && (
                <button
                  type="button"
                  onClick={closeCamera}
                  className="font-mono text-[9px] tracking-[0.25em] text-[#888] uppercase hover:text-[#ccc]"
                  onMouseEnter={() => setGazeMode('analyzing')}
                  onMouseLeave={() => setGazeMode('idle')}
                >
                  {t.scan.close}
                </button>
              )}
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleImageFileChange}
            />
          </div>
        </div>
      </div>
    </div>
  )

  const renderM1Page = () => (
    <div className="absolute inset-0 h-full w-full">
      <video ref={m1VideoRef} autoPlay playsInline className="hidden" />
      <img ref={m1UploadImageRef} alt="upload-source" className="hidden" />
      <input ref={m1FileInputRef} type="file" accept="image/*" className="hidden" onChange={handleM1ImageFileChange} />

      <div
        ref={m1ContainerRef}
        className="pointer-events-none absolute inset-0 z-0"
        style={{ background: 'radial-gradient(circle at center, rgba(10, 20, 35, 0.6) 0%, #000000 90%)' }}
      />

      <div className="relative z-10 flex h-full w-full flex-col justify-between px-4 pb-8 pt-6 sm:px-6 lg:px-10 lg:pb-10 lg:pt-8">
        <div className="flex items-start justify-between">
          <div>
            <p className="m-page-title">{t.m1.header}</p>
            <div className="mt-4 h-[1px] w-12 bg-blue-500/40" />
          </div>
        </div>

        <div className="flex h-full flex-col items-start justify-start gap-8 pb-24 pt-10 lg:flex-row lg:items-center lg:justify-between lg:pt-16">
          <div className="m1-left-panel">
            <h2 className="m1-panel-title">{t.m1.capture}</h2>
            <div className="space-y-4">
              <button
                type="button"
                onClick={() => void handleM1Camera()}
                className={`m1-mode-button ${m1CaptureMode === 'camera' ? 'is-active' : ''}`}
                onMouseEnter={() => setGazeMode('analyzing')}
                onMouseLeave={() => setGazeMode('idle')}
              >
                <span className="m1-mode-indicator">{m1CaptureMode === 'camera' ? '>' : ''}</span>
                <span>{t.m1.modeRealtime}</span>
              </button>
              <button
                type="button"
                onClick={handleM1UploadClick}
                className={`m1-mode-button ${m1CaptureMode === 'static' ? 'is-active' : ''}`}
                onMouseEnter={() => setGazeMode('analyzing')}
                onMouseLeave={() => setGazeMode('idle')}
              >
                <span className="m1-mode-indicator">{m1CaptureMode === 'static' ? '>' : ''}</span>
                <span>{t.m1.modeStatic}</span>
              </button>
            </div>
            <div className="mt-4 pt-2 text-left">
              <div className="mb-4 h-px w-14 bg-white/20" />
              <p className="h-4 font-mono text-[9px] tracking-[2px] text-white/70 uppercase">{m1SystemLog}</p>
            </div>
          </div>

          <div
            className={`m1-right-panel transition-all duration-700 ${
              m1ResultVisible ? 'translate-x-0 opacity-100' : 'translate-x-10 opacity-0'
            }`}
          >
            <div className="mb-8 flex items-center gap-3 font-mono text-[11px] tracking-[0.25em] text-yellow-500/90">
              <div className="h-1.5 w-1.5 animate-pulse rounded-sm bg-yellow-500" />
              [ {t.m1.result} ]
            </div>
            <div className="space-y-4 font-mono text-[10px] uppercase tracking-wider">
              <div className="m1-data-row">
                <span className="m1-data-label">{t.m1.rowElement}</span>
                <span className="m1-data-value m1-data-value-accent">{m1TypeDisplay}</span>
              </div>
              <div className="m1-data-row">
                <span className="m1-data-label">{t.m1.rowAngle}</span>
                <span className="m1-data-value">{m1ResultAngle}</span>
              </div>
              <div className="m1-data-row">
                <span className="m1-data-label">{t.m1.rowZygomatic}</span>
                <span className="m1-data-value">{m1ResultBone}</span>
              </div>
            </div>
            {m1ShowEnterM2 && (
              <button
                type="button"
                className="mt-8 w-full rounded-sm border border-yellow-500/50 bg-transparent py-3.5 font-mono text-[10px] tracking-[0.2em] text-yellow-500 uppercase transition-all hover:bg-yellow-500/10"
                onClick={() => setActivePage('M2')}
                onMouseEnter={() => setGazeMode('analyzing')}
                onMouseLeave={() => setGazeMode('idle')}
              >
                {t.m1.enterM2}
              </button>
            )}
          </div>
        </div>

        <div className="pointer-events-none absolute bottom-10 left-1/2 w-full max-w-lg -translate-x-1/2 px-4 text-center lg:bottom-14 lg:px-0">
          <button
            type="button"
            onClick={handleM1Deconstruct}
            className={`m1-extract-trigger pointer-events-auto transition-all ${
              m1DeconstructVisible
                ? 'is-visible'
                : 'pointer-events-none opacity-0'
            }`}
            onMouseEnter={() => setGazeMode('analyzing')}
            onMouseLeave={() => setGazeMode('idle')}
          >
            <span className="m1-extract-text">{t.m1.extract}</span>
            <span className="m1-extract-line" />
          </button>
        </div>
      </div>
    </div>
  )

  const renderM2Page = () => {
    const displayKeywords = m2Keywords.length ? m2Keywords : [t.m2.awaiting]
    const displayLogs = m2Logs
    return (
      <div className="absolute inset-0 h-full w-full animate-in fade-in duration-[1500ms]">
        <div className="flex h-full w-full flex-col px-4 pb-8 pt-6 sm:px-6 lg:px-24 lg:pb-10 lg:pt-8">
          <div className="flex items-start justify-between">
            <div className="tracking-widest">
              <p className="m-page-title">{t.m2.header}</p>
              <div className="mt-4 h-[1px] w-12 bg-gray-600" />
            </div>
          </div>

          <div className="flex flex-1 flex-col items-stretch justify-start gap-8 pt-6 lg:flex-row lg:items-center lg:justify-between lg:gap-10 lg:pt-12">
            <div className="w-full max-w-none space-y-4 lg:w-1/4 lg:max-w-xs lg:space-y-6">
              <div className="flex h-32 flex-col justify-end overflow-hidden font-serif text-sm leading-relaxed text-gray-300">
                {displayLogs.map((item, index) => (
                  <p
                    key={`${item}-${index}`}
                    className={
                      m2Logs.length && index === displayLogs.length - 1
                        ? 'mt-2 border-l-2 border-yellow-500 pl-2 text-white'
                        : 'opacity-40'
                    }
                  >
                    {item}
                  </p>
                ))}
              </div>
            </div>

            <div
              className={`relative order-1 flex flex-1 -translate-y-36 items-center justify-center transition-all duration-[850ms] sm:-translate-y-32 md:-translate-y-44 lg:order-none lg:translate-y-0 ${
                m2Collapsing ? 'scale-0 opacity-0 blur-sm' : 'scale-100 opacity-100 blur-0'
              }`}
            >
              <div className="absolute h-[86vw] w-[86vw] max-h-[76vh] max-w-[76vh] rounded-full bg-transparent lg:h-[80vh] lg:w-[80vh]" />
              <div
                ref={m2ContainerRef}
                className="absolute z-10 h-[82vw] w-[82vw] max-h-[72vh] max-w-[72vh] lg:h-[76vh] lg:w-[76vh]"
              />
              <div className="pointer-events-none absolute h-[42vw] w-[42vw] max-h-[36vh] max-w-[36vh] rounded-full border border-white/10 opacity-70 lg:h-[40vh] lg:w-[40vh]" />
            </div>

            <div className="w-full max-w-none space-y-5 text-left lg:w-1/4 lg:max-w-xs lg:space-y-6 lg:text-right">
              <div>
                <div className="mb-1 text-[10px] uppercase tracking-widest text-gray-500">{t.m2.confidence}</div>
                <div className="text-4xl font-light tracking-tighter lg:text-6xl" style={{ color: m2Accent }}>
                  {m2Confidence.toFixed(1)}
                  <span className="text-2xl text-gray-500">%</span>
                </div>
              </div>
              <div>
                <div className="mb-1 text-[10px] uppercase tracking-widest text-gray-500">{t.m2.state}</div>
                <div className="text-2xl font-bold tracking-widest" style={{ color: m2Accent }}>
                  {m2StateCopy.label}
                </div>
                <div className="mt-2 text-[10px] uppercase tracking-[0.2em] text-gray-500">{m2StateCopy.sub}</div>
              </div>
              <div className="mt-4 flex flex-wrap justify-end gap-2 text-[10px] uppercase text-gray-600">
                {displayKeywords.map((keyword) => (
                  <span
                    key={keyword}
                    className="rounded border border-gray-700 px-2 py-1"
                    style={{ color: m2Accent, borderColor: `${m2Accent}80` }}
                  >
                    {keyword}
                  </span>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-auto flex flex-col items-center pb-3">
            <div className="w-full max-w-lg">
              <div className="flex items-center rounded-full border border-white/10 bg-white/[0.03] px-4 py-3 backdrop-blur-md transition-all duration-300 focus-within:border-yellow-500/50 focus-within:shadow-[0_0_20px_rgba(255,215,0,0.1)] sm:px-6 sm:py-4">
                <span className="mr-4 text-yellow-500">{'>'}</span>
                <input
                  type="text"
                  value={m2Input}
                  onChange={(event) => setM2Input(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') {
                      handleM2Submit()
                    }
                  }}
                  placeholder={t.m2.inputPlaceholder}
                  className="w-full bg-transparent text-sm font-serif text-white placeholder-gray-600 outline-none"
                  onMouseEnter={() => setGazeMode('analyzing')}
                  onMouseLeave={() => setGazeMode('idle')}
                />
                <button
                  type="button"
                  onClick={handleM2Submit}
                  className="ml-4 text-gray-500 transition-colors hover:text-white"
                  onMouseEnter={() => setGazeMode('analyzing')}
                  onMouseLeave={() => setGazeMode('idle')}
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M5 12h14M12 5l7 7-7 7" />
                  </svg>
                </button>
              </div>
              <div className="mt-4 text-center text-[10px] uppercase tracking-widest text-gray-600">{t.m2.moduleLabel}</div>
              {m2ReadyToAdvance && (
                <div className="mt-6 flex justify-center">
                  <button
                    type="button"
                    onClick={handleM2Advance}
                    className="w-full rounded-full border border-[#d4af37]/80 bg-[#d4af37]/12 px-6 py-3 font-mono text-[11px] tracking-[0.26em] text-[#f5d27b] uppercase shadow-[0_0_18px_rgba(212,175,55,0.28)] transition-all hover:bg-[#d4af37]/18 hover:text-[#ffe6a3] sm:w-auto sm:px-10"
                    onMouseEnter={() => setGazeMode('analyzing')}
                    onMouseLeave={() => setGazeMode('idle')}
                  >
                    {t.m2.enterM3}
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    )
  }

  const renderM3Page = () => {
    const currentState =
      agencyValue <= 33 ? t.m3.stateSealed : agencyValue <= 66 ? t.m3.stateHalf : t.m3.stateFree
    const m3SliderNormalized = Math.max(0, Math.min(1, agencyValue / 100))
    const m3WillDominant = m3SliderNormalized > 0.5

    return (
      <div className="relative h-full w-full animate-in fade-in duration-[1500ms]">
        <div className="absolute left-4 top-[-28px] z-20 sm:left-6 sm:top-[-32px] lg:left-24 lg:top-[-36px]">
          <p className="m-page-title">{t.m3.header}</p>
          <div className="mt-4 h-[1px] w-12 bg-gray-600" />
        </div>
        <p className="pointer-events-none absolute left-12 top-1/2 z-10 hidden -translate-y-1/2 text-[11px] leading-relaxed tracking-[0.3em] text-white/40 [writing-mode:vertical-rl] lg:block">
          {t.m3.left}
        </p>
        <div className="absolute left-4 right-4 top-10 z-30 rounded border border-white/12 bg-black/45 p-3 backdrop-blur-sm sm:top-12 md:left-auto md:right-8 md:top-24 md:w-52 md:translate-y-0 md:p-4 lg:right-10 lg:top-1/2 lg:-translate-y-1/2">
          <p className="font-mono text-[9px] tracking-[0.28em] text-white/45 uppercase">{t.m3.currentState}</p>
          <p className="mt-2 font-serif text-base text-[#d4af37] md:text-xl">{currentState}</p>
        </div>
        <div className="absolute left-1/2 top-[16%] z-10 -translate-x-1/2 text-right font-mono text-[7px] tracking-[0.2em] text-white/30 uppercase md:top-[18%] md:text-[8px] md:tracking-[0.22em]">
          {t.m3.identity}
        </div>

        <main className="relative z-20 flex h-full flex-col items-center justify-center gap-8">
          <section className="[perspective:1000px]">
            <div
              className="relative h-[54vw] min-h-[280px] w-[92vw] max-h-[400px] max-w-[720px] overflow-hidden border border-white/10 bg-[linear-gradient(135deg,rgba(255,255,255,0.08)_0%,rgba(255,255,255,0.02)_50%,rgba(255,255,255,0.05)_100%)] shadow-[0_0_40px_rgba(0,0,0,0.5)] backdrop-blur-[8px] transition-all duration-[800ms]"
              style={{ transform: `rotateY(${prismRotateY}deg) rotateX(${prismRotateX}deg)` }}
            >
              <div
                className="absolute inset-0 z-10 bg-[repeating-linear-gradient(0deg,rgba(0,255,255,0.03)_0px,rgba(0,255,255,0.03)_1px,transparent_1px,transparent_2px)] transition-opacity duration-500"
                style={{ opacity: 1 - agencyRatio * 0.8 }}
              />
              <div className="absolute inset-0 z-0 bg-[radial-gradient(circle,transparent_20%,#000_150%)] opacity-20" />
              <div className="absolute inset-0 flex items-center justify-center">
                {m3MaskLoading && (
                  <div className="absolute z-20 flex w-[320px] flex-col items-center gap-4">
                    <div className="font-mono text-[12px] tracking-[0.36em] text-white/55 uppercase">
                      [ {t.m3.loadingMask} <span className="text-[#F3E5AB] [text-shadow:0_0_12px_rgba(212,175,55,0.6)]">{m3MaskProgress}%</span> ]
                    </div>
                    <div className="h-px w-[240px] bg-white/10">
                      <div
                        className="h-full bg-[#D4AF37] transition-all duration-300 [box-shadow:0_0_8px_rgba(212,175,55,0.6),0_0_20px_rgba(212,175,55,0.2)]"
                        style={{ width: `${m3MaskProgress}%` }}
                      />
                    </div>
                  </div>
                )}
                {m3MaskSrc && (
                  <img
                    id="m3-mask-display"
                    src={m3MaskSrc}
                    alt="3D glass totem"
                    loading="eager"
                    decoding="async"
                    className="h-[46vw] w-[78vw] max-h-[320px] max-w-[520px] object-contain transition-all duration-500"
                    style={{
                      filter: personaFilter,
                      transform: `translateZ(${6 + agencyRatio * 12}px)`,
                    }}
                  />
                )}
              </div>
            </div>
          </section>

          <section className="flex w-full max-w-[500px] -translate-y-2 flex-col gap-5 px-4 md:-translate-y-4 md:px-0">
            <div className="flex items-center justify-between font-mono text-[12px] tracking-[0.28em] uppercase">
              <span style={{ color: m3WillDominant ? 'rgba(255,255,255,0.3)' : 'rgba(255,255,255,0.4)' }}>{t.m3.algorithm}</span>
              <span
                style={{
                  color: m3WillDominant ? '#FFD700' : 'rgba(255, 215, 0, 0.4)',
                  textShadow: m3WillDominant ? '0 0 10px rgba(212, 175, 55, 0.2)' : 'none',
                }}
              >
                {t.m3.freeWill}
              </span>
            </div>
            <div
              className="relative flex h-10 w-full items-center"
              onMouseEnter={() => setGazeMode('analyzing')}
              onMouseLeave={() => setGazeMode('idle')}
            >
              <input
                type="range"
                min={0}
                max={1000}
                value={Math.round(agencyValue * 10)}
                onChange={(event) => {
                  setAgencyValue(Number(event.target.value) / 10)
                  setM3Touched(true)
                }}
                onMouseUp={() => loadM3Mask(m3MaskCandidates)}
                onTouchEnd={() => loadM3Mask(m3MaskCandidates)}
                className="absolute inset-0 z-20 h-full w-full cursor-pointer appearance-none bg-transparent opacity-0"
              />
              <div className="pointer-events-none absolute inset-0 z-10 flex items-center">
                <div className="absolute left-0 h-1.5 w-px bg-white/20" />
                <div className="absolute right-0 h-1.5 w-px bg-white/20" />
                <div className="absolute left-0 h-px w-full bg-white/10" />
                <div
                  className="absolute left-0 h-px bg-[#D4AF37] shadow-[0_0_8px_rgba(212,175,55,0.6)]"
                  style={{ width: `calc(${m3SliderNormalized} * 100%)` }}
                />
                <div
                  className="absolute w-[2px] -translate-x-1/2 rounded-[1px] bg-[#F3E5AB] shadow-[0_0_12px_rgba(212,175,55,0.6),0_0_4px_rgba(255,255,255,0.8)] transition-all duration-200"
                  style={{ left: `calc(${m3SliderNormalized} * 100%)`, height: '32px' }}
                />
              </div>
            </div>
            {m3Touched && (
              <div className="mt-6 flex justify-center">
                <button
                  type="button"
                  onClick={handleM3Manifest}
                  className="rounded-full border border-[#d4af37]/60 px-8 py-3 font-mono text-[11px] tracking-[0.24em] text-[#d4af37] uppercase transition-all hover:bg-[#d4af37]/12 hover:text-[#f5d27b]"
                  onMouseEnter={() => setGazeMode('analyzing')}
                  onMouseLeave={() => setGazeMode('idle')}
                >
                  {t.m3.manifest}
                </button>
              </div>
            )}
          </section>
        </main>
      </div>
    )
  }

  const renderM4Page = () => {
    const gestureColor =
      m4GestureLabel.startsWith('FIST') ? '#FF4500' : m4GestureLabel.startsWith('OPEN') ? '#FFD700' : m4GestureLabel
          .startsWith('MOUSE')
        ? '#4169E1'
        : '#6b7280'
    const destinyAccent = mixHex(m4Energy.primaryColor, m4Energy.overlayColor, 0.45)
    const cardCoreColor = m4CoreParticleColorMap[m4BoneElement]
    const cardRingColor = resolveM4RingColorByEmotion(m2State)
    const totemSpecies = m3TotemSpeciesMap[m4BoneElement]
    const elementLabelMap: Record<M4EnergyElement, string> = {
      metal: 'METAL ARCHETYPE',
      wood: 'WOOD ARCHETYPE',
      water: 'WATER ARCHETYPE',
      fire: 'FIRE ARCHETYPE',
      earth: 'EARTH ARCHETYPE',
    }
    const emotionLabelMap: Record<M2State, string> = {
      ANXIETY: 'ANXIETY',
      ANGER: 'ANGER',
      SADNESS: 'SADNESS',
      JOY: 'JOY',
      EXCITEMENT: 'EXCITEMENT',
      CALM: 'CALM',
      NEUTRAL: 'NEUTRAL',
      CALCULATING: 'NEUTRAL',
      NULL: 'NEUTRAL',
    }
    const ringState =
      m4Energy.ringBreakRatio < 0.15
        ? t.m4.ringContinuous
        : m4Energy.ringBreakRatio < 0.45
          ? t.m4.ringLightBreak
          : m4Energy.ringBreakRatio < 0.75
            ? t.m4.ringMediumBreak
            : t.m4.ringHeavyBreak
    const m4DominantLabel =
      lang === 'cn'
        ? ({ metal: '金', wood: '木', water: '水', fire: '火', earth: '土' }[m4Energy.dominant] ?? m4Energy.dominant)
        : m4Energy.dominant.toUpperCase()
    const m4MoodLabel =
      lang === 'cn'
        ? ({ anxiety: '焦虑', joy: '喜悦', calm: '平静', neutral: '中性' }[m4Mood] ?? m4Mood)
        : ({ anxiety: 'ANXIETY', joy: 'JOY', calm: 'CALM', neutral: 'NEUTRAL' }[m4Mood] ?? m4Mood.toUpperCase())
    const m4EnergyTitle = lang === 'cn' ? `${m4DominantLabel}曜${m4MoodLabel}态` : `${m4DominantLabel} ${m4MoodLabel} STATE`
    return (
      <div className="relative h-full w-full animate-in fade-in duration-[1500ms]">
        {m4AmbientAudioSrc && <audio ref={m4AudioRef} loop src={m4AmbientAudioSrc} />}
        <div
          ref={m4ContainerRef}
          className="pointer-events-none absolute inset-0 z-0"
          style={{ background: 'radial-gradient(circle at center, rgba(30, 20, 10, 0.2) 0%, #000000 80%)' }}
        />

        {m4ExportEntryOpen && (
          <div className="absolute inset-0 z-40 flex flex-col items-center justify-center bg-black/90">
            <p className="mb-10 text-sm tracking-[0.5em] text-white/60">{t.m4.enterObserver}</p>
            <div className="w-[300px]">
              <input
                type="text"
                value={m4ObserverInput}
                onChange={(event) => setM4ObserverInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    void handleM4ExportConfirm()
                  }
                }}
                placeholder="NICKNAME"
                className="w-full border-0 border-b border-[#d4af37]/40 bg-transparent px-0 py-2 text-center font-serif text-2xl text-[#d4af37] outline-none transition-all focus:border-[#d4af37]"
                autoFocus
              />
            </div>
            <div className="mt-8 flex items-center gap-4">
              <button
                type="button"
                onClick={() => setM4ExportEntryOpen(false)}
                className="rounded-full border border-white/20 px-5 py-2 text-xs tracking-[0.2em] text-white/70 uppercase hover:border-white/40"
              >
                {t.m4.cancel}
              </button>
              <button
                type="button"
                onClick={() => void handleM4ExportConfirm()}
                className="rounded-full border border-[#d4af37]/70 px-5 py-2 text-xs tracking-[0.2em] text-[#d4af37] uppercase hover:bg-[#d4af37]/10"
              >
                {t.m4.confirm}
              </button>
            </div>
          </div>
        )}

        <div className="absolute bottom-5 right-4 z-20 h-[88px] w-[118px] overflow-hidden rounded-lg border border-yellow-500/30 bg-black/80 transition-opacity duration-300 md:bottom-8 md:right-8 md:h-[120px] md:w-[160px]">
          <video ref={videoRef} autoPlay muted playsInline className="h-full w-full object-cover -scale-x-100" />
          {m4CamStatus && (
            <div className="absolute inset-0 z-10 flex items-center justify-center bg-black/50 px-2 text-center text-[10px]" style={{ color: m4CamStatusColor }}>
              {m4CamStatus}
            </div>
          )}
        </div>

        {!m4Activated && (
          <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-black/85 backdrop-blur-md">
            <p className="mb-12 text-xs tracking-widest text-gray-400">{t.m4.collapseTitle}</p>
            <button
              type="button"
              onClick={handleM4Start}
              className="mb-6 border border-yellow-500 px-12 py-4 text-[14px] tracking-[0.2em] text-yellow-500 transition-all duration-300 hover:bg-yellow-500/10 hover:shadow-[0_0_20px_rgba(255,215,0,0.4)]"
            >
              {t.m4.activateField}
            </button>
            <p className="max-w-md text-center text-[10px] text-gray-500">
              {t.m4.camHint}
            </p>
          </div>
        )}

        <div ref={m4CardRef} className="relative z-10 flex h-full w-full flex-col justify-between px-4 pb-6 pt-0 sm:px-6 sm:pt-0 lg:px-24 lg:pb-8 lg:pt-0">
          <div className="absolute right-4 top-6 z-30 lg:hidden">
            <button
              type="button"
              onClick={handleM4Export}
              disabled={m4Exporting}
              className="rounded-full border border-[#d4af37]/70 bg-[#d4af37]/10 px-4 py-2 font-mono text-[10px] tracking-[0.18em] text-[#f5d27b] uppercase transition-all hover:bg-[#d4af37]/18 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {m4Exporting ? 'EXPORTING' : 'DOWNLOAD CARD'}
            </button>
            {m4ExportHint && <p className="mt-2 text-right text-[9px] tracking-wider text-gray-400 uppercase">{m4ExportHint}</p>}
            {m4ExportHint === 'EXPORT FAILED' && m4ExportError && <p className="mt-1 max-w-[180px] text-right text-[9px] tracking-wider text-red-400/80 uppercase">{m4ExportError}</p>}
          </div>
          <div className="flex -mt-11 flex-col items-start justify-between gap-5 md:-mt-[56px] md:flex-row md:gap-0 lg:-mt-[64px]">
            <div className="tracking-widest">
              <p className="m-page-title">{t.m4.header}</p>
              <div className="mt-4 h-[1px] w-12 bg-gray-600" />
              <p className="mt-5 font-serif text-2xl tracking-wide" style={{ color: destinyAccent }}>
                {m4EnergyTitle}
              </p>
              <p className="mt-2 text-[10px] tracking-[0.22em] text-gray-400 uppercase">{t.m4.energyConvergence} {m4Energy.confidence}%</p>
            </div>
            <div className="hidden text-right lg:block">
              <div className="mb-1 text-[10px] uppercase tracking-widest text-gray-500">{t.m4.entityControl}</div>
              <div
                className="text-2xl font-bold tracking-widest drop-shadow-[0_0_10px_rgba(255,215,0,0.5)]"
                style={{ color: gestureColor }}
                data-gesture={m4GestureState}
              >
                {m4GestureLabel}
              </div>
              <div className="mt-5 rounded border border-white/12 bg-black/45 px-4 py-3 text-left text-[10px] tracking-widest text-gray-300">
                <div className="mb-2 flex justify-between gap-4">
                  <span className="text-gray-500 uppercase">{t.m4.weightBone}</span>
                  <span className="uppercase">{lang === 'cn' ? m4DominantLabel : m4BoneElement} +{m4Energy.explain.boneAdded.toFixed(2)}</span>
                </div>
                <div className="mb-2 flex justify-between gap-4">
                  <span className="text-gray-500 uppercase">{t.m4.weightMood}</span>
                  <span className="uppercase">{m4Mood} +{m4Energy.explain.moodAdded.toFixed(2)}</span>
                </div>
                <div className="mb-2 flex justify-between gap-4">
                  <span className="text-gray-500 uppercase">{t.m4.weightWill}</span>
                  <span className="uppercase">
                    {Math.round(m4Energy.volitionRatio * 100)}% / +{m4Energy.explain.freedomBiasDelta.toFixed(2)}
                  </span>
                </div>
                <div className="mb-2 flex justify-between gap-4">
                  <span className="text-gray-500 uppercase">{t.m4.ringDispersion}</span>
                  <span>{ringState}</span>
                </div>

              </div>
              <div className="mt-4 flex justify-end">
                <button
                  type="button"
                  onClick={handleM4Export}
                  disabled={m4Exporting}
                  className="rounded-full border border-[#d4af37]/60 px-5 py-2 font-mono text-[10px] tracking-[0.2em] text-[#d4af37] uppercase transition-all hover:bg-[#d4af37]/12 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {m4Exporting ? 'EXPORTING' : 'DOWNLOAD CARD'}
                </button>
              </div>
              {m4ExportHint && <p className="mt-2 text-[10px] tracking-widest text-gray-400 uppercase">{m4ExportHint}</p>}
              {m4ExportHint === 'EXPORT FAILED' && m4ExportError && (
                <p className="mt-1 text-[10px] tracking-wider text-red-400/80 uppercase">{m4ExportError}</p>
              )}
            </div>
          </div>

          <div className="absolute bottom-6 left-1/2 w-full max-w-lg -translate-x-1/2 px-3 text-center md:bottom-10 md:px-0">
            <div className="flex flex-wrap justify-center gap-4 text-[9px] uppercase tracking-widest text-gray-500 md:gap-8 md:text-[10px]">
              <span className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-yellow-500" />
                {t.m4.tipOpen}
              </span>
              <span className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-red-500" />
                {t.m4.tipFist}
              </span>
              <span className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-blue-500" />
                {t.m4.tipRotate}
              </span>
            </div>
          </div>
        </div>

        {shouldRenderM4ExportCard && <div className="pointer-events-none absolute -left-[99999px] top-0 z-[-1]">
          <div
            ref={m4ExportCardRef}
            data-export-card="m4"
            style={{
              position: 'relative',
              display: 'flex',
              flexDirection: 'column',
              width: '380px',
              height: '680px',
              background:
                'linear-gradient(180deg, #070707 0%, #030303 55%, #020202 100%) padding-box, linear-gradient(135deg, rgba(135,206,250,0.95) 0%, rgba(212,175,55,0.95) 100%) border-box',
              padding: '32px',
              boxSizing: 'border-box',
              color: '#ffffff',
              border: '1px solid transparent',
              outline: '1px solid rgba(212,175,55,0.15)',
              outlineOffset: '-8px',
              boxShadow: '0 20px 60px rgba(0,0,0,0.9), 0 0 32px rgba(135,206,250,0.16), 0 0 24px rgba(212,175,55,0.14)',
            }}
          >
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                fontFamily: '"Space Mono", monospace',
                fontSize: '9px',
                letterSpacing: '2px',
                color: 'rgba(255,255,255,0.3)',
                textTransform: 'uppercase',
              }}
            >
              <span>FaceFrame_Art</span>
              <span>M4_Resolution</span>
            </div>
            <div
              style={{
                marginTop: '28px',
                position: 'relative',
                height: '190px',
                width: '100%',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
              }}
            >
              <div style={{ position: 'relative', width: '204px', height: '204px' }}>
                <svg width="204" height="204" viewBox="0 0 280 280" style={{ overflow: 'visible', position: 'absolute', left: 0, top: 0 }}>
                  <g transform="translate(140 140) scale(1.24) translate(-140 -140)">
                    {m4ExportFallbackStar.ringBackPoints.map((point, index) => (
                      <circle
                        key={`fallback-ring-back-${index}`}
                        cx={140 + point.x}
                        cy={140 + point.y}
                        r={point.r}
                        fill={cardRingColor}
                        fillOpacity={point.a}
                      />
                    ))}
                    {m4ExportFallbackStar.corePoints.map((point, index) => (
                      <circle
                        key={`fallback-core-${index}`}
                        cx={140 + point.x}
                        cy={140 + point.y}
                        r={point.r}
                        fill={cardCoreColor}
                        fillOpacity={point.a}
                      />
                    ))}
                    {m4ExportFallbackStar.ringFrontPoints.map((point, index) => (
                      <circle
                        key={`fallback-ring-front-${index}`}
                        cx={140 + point.x}
                        cy={140 + point.y}
                        r={point.r}
                        fill={cardRingColor}
                        fillOpacity={point.a}
                      />
                    ))}
                  </g>
                </svg>
              </div>
            </div>
            <div style={{ marginTop: '16px', textAlign: 'center', paddingBottom: '20px' }}>
              <p
                style={{
                  marginBottom: '8px',
                  fontFamily: '"Space Mono", monospace',
                  fontSize: '9px',
                  letterSpacing: '4px',
                  color: 'rgba(255,255,255,0.5)',
                  textTransform: 'uppercase',
                }}
              >
                OBSERVER / ENTITY
              </p>
              <p style={{ fontSize: '27px', letterSpacing: '5px', color: '#f2f2f2' }}>{m4ObserverName}</p>
              <div
                style={{
                  margin: '8px auto 0',
                  width: '96px',
                  height: '1px',
                  borderRadius: '999px',
                  background: 'linear-gradient(90deg, rgba(212,175,55,0) 0%, rgba(212,175,55,0.25) 18%, rgba(212,175,55,0.98) 50%, rgba(212,175,55,0.25) 82%, rgba(212,175,55,0) 100%)',
                  boxShadow: '0 0 8px rgba(212,175,55,0.55)',
                }}
              />
            </div>
            <div style={{ marginTop: '24px', display: 'flex', flexDirection: 'column', flex: 1, position: 'relative', paddingRight: '34px', paddingLeft: '12px' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <span style={{ fontFamily: '"Space Mono", monospace', fontSize: '9px', color: 'rgba(255,255,255,0.35)' }}>GEOMETRY_STRUCTURE</span>
                  <span style={{ fontSize: '13px', letterSpacing: '2px', color: 'rgba(255,255,255,0.9)' }}>{elementLabelMap[m4BoneElement]}</span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <span style={{ fontFamily: '"Space Mono", monospace', fontSize: '9px', color: 'rgba(255,255,255,0.35)' }}>STATE_FIELD</span>
                  <span style={{ fontSize: '13px', letterSpacing: '2px', color: 'rgba(255,255,255,0.9)' }}>{emotionLabelMap[m2State]}</span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <span style={{ fontFamily: '"Space Mono", monospace', fontSize: '9px', color: 'rgba(255,255,255,0.35)' }}>{t.m3.generatedTotem}</span>
                  <span style={{ fontSize: '13px', letterSpacing: '2px', color: 'rgba(255,255,255,0.92)' }}>{totemSpecies.en}</span>
                </div>
              </div>
              <div
                style={{
                  position: 'absolute',
                  right: '2px',
                  bottom: '74px',
                  writingMode: 'vertical-rl',
                  textOrientation: 'mixed',
                  fontFamily: '"Space Mono", monospace',
                  fontSize: '7px',
                  letterSpacing: '1.5px',
                  color: 'rgba(212,175,55,0.72)',
                  textTransform: 'uppercase',
                  textAlign: 'center',
                }}
              >
                [ ENERGY_WEIGHTS ] BONE: 60% | EMOTION: 30% | WILL: 10%
              </div>
              <div
                style={{
                  marginTop: 'auto',
                  paddingTop: '24px',
                  fontFamily: '"Space Mono", monospace',
                  fontSize: '8px',
                  letterSpacing: '3.2px',
                  color: '#87cefa',
                  textTransform: 'uppercase',
                  textAlign: 'center',
                }}
              >
                ONE FACE ONE CLICK ONE DESTINY
              </div>
            </div>
          </div>
        </div>}
      </div>
    )
  }

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-[#050505] font-sans text-[#EAEAEA] selection:bg-[#d4af37] selection:text-black">
      <AlgorithmicEyeCursor mode={gazeMode} />
      <div
        className="pointer-events-none fixed inset-0 z-0 bg-repeat opacity-[0.03] mix-blend-overlay"
        style={{ backgroundImage: 'url("https://www.transparenttextures.com/patterns/stardust.png")' }}
      />
      <div
        className="pointer-events-none fixed inset-0 z-0 opacity-10"
        style={{
          backgroundImage:
            'linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px)',
          backgroundSize: '40px 40px',
        }}
      />
      <div
        className="pointer-events-none fixed z-0 h-[800px] w-[800px] rounded-full opacity-40 mix-blend-screen transition-opacity duration-500"
        style={{
          background:
            'radial-gradient(circle, rgba(6,182,212,0.05) 0%, rgba(212,175,55,0.02) 40%, transparent 70%)',
          transform: `translate(calc(-50% + ${smoothPos.x}px), calc(-50% + ${smoothPos.y}px))`,
        }}
      />
      <header className="fixed top-0 z-50 flex w-full items-center justify-end p-6 mix-blend-difference lg:px-12">
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
            ref={engineButtonRef}
            type="button"
            onClick={toggleLogicOverlay}
            className="engine-icon text-[#888] transition-colors hover:text-[#d4af37]"
            title={lang === 'cn' ? '引擎：算法透明' : 'The Engine: Algorithm Transparency'}
          >
            <span className="pr-1 text-lg italic">∑</span>
          </button>
        </div>
      </header>

      <main className="relative z-10 h-full w-full flex-1 pb-24 pt-16">
        {activePage === 'SCAN' && renderScanPage()}
        {activePage === 'M1' && renderM1Page()}
        {activePage === 'M2' && renderM2Page()}
        {activePage === 'M3' && renderM3Page()}
        {activePage === 'M4' && renderM4Page()}
      </main>

      <button
        type="button"
        onClick={handleReturnToOrigin}
        onMouseEnter={() => {
          setGazeMode('analyzing')
          setReturnSigilActive(true)
        }}
        onMouseLeave={() => {
          setGazeMode('idle')
          setReturnSigilActive(false)
        }}
        className="absolute bottom-5 left-4 z-50 flex items-center gap-2 md:bottom-6 md:left-6 md:gap-2.5 lg:bottom-10 lg:left-10 lg:gap-3"
      >
        <span className="relative flex h-8 w-8 items-center justify-center md:h-9 md:w-9 lg:h-10 lg:w-10">
          <span className="absolute h-1.5 w-1.5 rounded-full bg-[#d4af37] shadow-[0_0_12px_rgba(212,175,55,0.9)] md:h-2 md:w-2" />
          <span
            className={`absolute rounded-full border border-dashed border-[#d4af37]/60 transition-all duration-300 ${
              returnSigilActive ? 'h-0 w-0 opacity-0' : 'h-4 w-4 opacity-100 md:h-5 md:w-5'
            }`}
            style={{ animation: 'spin 28s linear infinite' }}
          />
          <span
            className={`absolute rounded-full border border-dashed border-[#d4af37]/35 transition-all duration-300 ${
              returnSigilActive ? 'h-0 w-0 opacity-0' : 'h-6 w-6 opacity-100 md:h-7 md:w-7 lg:h-8 lg:w-8'
            }`}
            style={{ animation: 'spin 40s linear infinite reverse' }}
          />
        </span>
        <span
          className={`font-mono text-[7px] tracking-[0.24em] text-[#d4af37] uppercase transition-all duration-300 md:text-[8px] md:tracking-[0.28em] lg:text-[9px] lg:tracking-[0.34em] ${
            returnSigilActive ? 'translate-x-0 opacity-100' : '-translate-x-2 opacity-0'
          }`}
        >
          [ RETURN TO ORIGIN ]
        </span>
      </button>

      <div ref={logicOverlayRef} className={`logic-overlay ${engineOpen ? 'active' : ''}`}>
        <div className="logic-panel">
          <div className="logic-panel-head">
            <span className="logic-panel-title">∑ ALGORITHM WHITE-BOX / {logicPanelContent.moduleId}</span>
            <button
              type="button"
              onClick={toggleLogicOverlay}
              className="logic-panel-close"
            >
              <X size={16} strokeWidth={1.5} />
            </button>
          </div>
          <div className="logic-panel-subtitle">{logicData[logicPanelContent.moduleId].title}</div>
          <div className="logic-panel-body">
            {logicPanelContent.lines.map((line) => (
              <div key={line} className="logic-line">
                {line}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
