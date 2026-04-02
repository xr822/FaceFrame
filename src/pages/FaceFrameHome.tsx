import { Homepage } from './Homepage'

type FaceFrameHomeProps = {
  onEnterExhibition: () => void
}

export default function FaceFrameHome({ onEnterExhibition }: FaceFrameHomeProps) {
  return <Homepage onEnterExhibition={onEnterExhibition} />
}

