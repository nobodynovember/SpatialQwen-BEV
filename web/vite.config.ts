import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

const defaultQwenBaseUrl = 'http://10.79.128.145:8001/v1'

export default defineConfig(({ mode }) => {
  const environment = loadEnv(mode, process.cwd(), '')
  const qwenEndpoint = new URL(environment.QWEN_BASE_URL || defaultQwenBaseUrl)
  const qwenPathPrefix = qwenEndpoint.pathname.replace(/\/$/, '')

  return {
    plugins: [react()],
    server: {
      proxy: {
        '/api/qwen': {
          target: qwenEndpoint.origin,
          changeOrigin: true,
          headers: {
            Authorization: `Bearer ${environment.QWEN_API_KEY || 'EMPTY'}`,
          },
          rewrite: (path) => `${qwenPathPrefix}${path.replace(/^\/api\/qwen/, '')}`,
        },
      },
    },
  }
})
