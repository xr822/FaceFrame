import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { randomUUID } from 'node:crypto'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    {
      name: 'persona-api',
      configureServer(server) {
        const personas = ['/personas/fox.svg', '/personas/deer.svg', '/personas/wolf.svg', '/personas/rabbit.svg']

        const readBody = (req: import('http').IncomingMessage) =>
          new Promise<string>((resolve) => {
            let data = ''
            req.on('data', (chunk) => {
              data += chunk
            })
            req.on('end', () => resolve(data))
          })

        const pickPersona = (seed: string) => {
          let hash = 0
          for (let i = 0; i < seed.length; i += 1) {
            hash = (hash * 31 + seed.charCodeAt(i)) % personas.length
          }
          return personas[hash]
        }

        server.middlewares.use('/api/scan', async (req, res, next) => {
          if (req.method !== 'POST') {
            next()
            return
          }
          await readBody(req)
          const uploadId = randomUUID()
          res.statusCode = 200
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify({ uploadId, status: 'ok' }))
        })

        server.middlewares.use('/api/persona', async (req, res, next) => {
          if (req.method !== 'POST') {
            next()
            return
          }
          const raw = await readBody(req)
          let payload: { uploadId?: string } = {}
          try {
            payload = raw ? JSON.parse(raw) : {}
          } catch {
            payload = {}
          }
          const uploadId = payload.uploadId ?? randomUUID()
          const imageUrl = pickPersona(uploadId)
          res.statusCode = 200
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify({ personaId: uploadId, imageUrl, status: 'ok' }))
        })
      },
    },
  ],
})
