# FaceFrame

A cyber-meditation interactive web artwork built with React + Three.js.  
Through four stages (M1 → M4), the project maps face topology, emotional semantics, free-will modulation, and destiny-card synthesis into one continuous ritual-like experience.

## ✨ Overview

FaceFrame is an interactive digital art experience with four modules:

- **M1 / The Geometry**  
  Facial topology extraction and structural decomposition.
- **M2 / The Profiler**  
  Semantic-emotional flow field and state interpretation.
- **M3 / The Persona**  
  Glass totem reconstruction with free-will slider.
- **M4 / The Destiny**  
  Particle planet + destiny card generation/export.

## 🧱 Tech Stack

- **Frontend**: React, TypeScript, Vite
- **3D / Visual**: Three.js, GLSL-style shader effects
- **Image Export**: html2canvas
- **UI / Styling**: TailwindCSS + custom CSS
- **Lint/Build**: ESLint, TypeScript build

## 🚀 Quick Start

### 1) Install dependencies

```bash
npm install
```

### 2) Run development server

```bash
npm run dev -- --host 0.0.0.0 --port 4173
```

### 3) Build for production

```bash
npm run build
```

### 4) Lint

```bash
npm run lint
```

## 📁 Project Structure (simplified)

```text
src/
  pages/
    Homepage.tsx
    FaceFrameExhibition.tsx
  components/
  ...
```

## 🎮 Interaction Notes

- Enter from homepage to begin scan and module flow.
- M2 accepts text input to drive emotion/state transitions.
- M3 free-will slider changes persona rendering.
- M4 supports destiny card export (`DOWNLOAD CARD`).

## 📸 Screenshots

You can add screenshots here:

```md
![Homepage](./docs/screenshots/homepage.png)
![M4 Destiny](./docs/screenshots/m4-destiny.png)
```

## 🛠️ Publish to GitHub

After creating your GitHub repo:

```bash
git init
git add .
git commit -m "Initial commit: FaceFrame"
git branch -M main
git remote add origin https://github.com/<your-username>/FaceFrame.git
git push -u origin main
```

## 📄 License

For portfolio/demo use by default.  
If needed, add a formal LICENSE file (MIT / All Rights Reserved / custom).
