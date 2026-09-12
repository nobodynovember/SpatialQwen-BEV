import { useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import {
  Activity,
  BatteryMedium,
  Bot,
  BrainCircuit,
  Camera,
  ChevronRight,
  CircleCheck,
  CircleDot,
  Clock3,
  Download,
  FileText,
  Gauge,
  Layers3,
  Loader2,
  Map,
  Maximize2,
  MessageSquareText,
  Network,
  Pause,
  Play,
  RotateCcw,
  ScanLine,
  SendHorizontal,
  Settings2,
  SkipBack,
  SkipForward,
  Wifi,
} from 'lucide-react'
import {
  instruction,
  mapLayers,
  markerPositions,
  replaySteps,
  semanticLegend,
  type MapLayerId,
  type ReplayStep,
} from './data/replay'

const totalSteps = replaySteps.length - 1
const qwenModel = import.meta.env.VITE_QWEN_MODEL || 'qwen3-vl-8b'

type ChatRole = 'assistant' | 'user' | 'error'

interface ChatMessage {
  id: string
  role: ChatRole
  content: string
}

interface QwenCompletion {
  choices?: Array<{ message?: { content?: unknown } }>
  error?: { message?: string }
}

function formatElapsed(seconds: number) {
  const minutes = Math.floor(seconds / 60)
  const remainder = seconds % 60
  return `00:${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`
}

function createMessageId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function getCompletionText(payload: QwenCompletion) {
  const content = payload.choices?.[0]?.message?.content
  if (typeof content === 'string') return content.trim()
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === 'string') return part
        if (typeof part === 'object' && part && 'text' in part && typeof part.text === 'string') return part.text
        return ''
      })
      .join('')
      .trim()
  }
  return ''
}

function buildQwenSystemPrompt(step: ReplayStep) {
  return [
    'You are the local Qwen assistant embedded in the SpatialQwen-BEV navigation replay console.',
    'Answer concisely and accurately. You can explain the replay state, the spatial reasoning, or the likely next navigation action.',
    'Do not claim to control a real robot or that an action has been executed outside this replay.',
    `Navigation instruction: ${instruction}`,
    `Current viewpoint: VP-${step.node.toString().padStart(2, '0')} (${step.location}).`,
    `Current action: ${step.action}.`,
    `Current subtask: ${step.subtask}.`,
    `Navigation confidence: ${step.confidence.toFixed(2)}.`,
  ].join('\n')
}

function App() {
  const [currentIndex, setCurrentIndex] = useState(3)
  const [isPlaying, setIsPlaying] = useState(false)
  const [mapLayer, setMapLayer] = useState<MapLayerId>('color')
  const [isExpanded, setIsExpanded] = useState(false)
  const [command, setCommand] = useState('')
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content: 'Ask about the current replay state, the route, or the navigation decision.',
    },
  ])
  const [isChatLoading, setIsChatLoading] = useState(false)
  const [notice, setNotice] = useState('')
  const noticeTimeout = useRef<number | undefined>(undefined)
  const chatHistoryRef = useRef<HTMLDivElement | null>(null)

  const currentStep = replaySteps[currentIndex]
  const selectedLayer = useMemo(
    () => mapLayers.find((layer) => layer.id === mapLayer) ?? mapLayers[0],
    [mapLayer],
  )
  const marker = markerPositions[currentIndex]
  const subtaskNumber = currentIndex < 3 ? 1 : currentIndex < 5 ? 2 : 3

  useEffect(() => {
    if (!isPlaying) return undefined

    const interval = window.setInterval(() => {
      setCurrentIndex((index) => {
        if (index >= totalSteps) {
          setIsPlaying(false)
          return index
        }
        return index + 1
      })
    }, 1450)

    return () => window.clearInterval(interval)
  }, [isPlaying])

  useEffect(() => {
    return () => window.clearTimeout(noticeTimeout.current)
  }, [])

  useEffect(() => {
    chatHistoryRef.current?.scrollTo({ top: chatHistoryRef.current.scrollHeight, behavior: 'smooth' })
  }, [chatMessages, isChatLoading])

  const showNotice = (message: string) => {
    window.clearTimeout(noticeTimeout.current)
    setNotice(message)
    noticeTimeout.current = window.setTimeout(() => setNotice(''), 2600)
  }

  const restart = () => {
    setCurrentIndex(0)
    setIsPlaying(false)
    showNotice('Replay reset to the first viewpoint.')
  }

  const togglePlayback = () => {
    if (currentIndex === totalSteps) setCurrentIndex(0)
    setIsPlaying((playing) => !playing)
  }

  const handleCommand = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const trimmed = command.trim()
    if (!trimmed || isChatLoading) return

    const userMessage: ChatMessage = {
      id: createMessageId(),
      role: 'user',
      content: trimmed,
    }
    const nextMessages = [...chatMessages, userMessage]

    setChatMessages(nextMessages)
    setCommand('')
    setIsChatLoading(true)

    try {
      const response = await fetch('/api/qwen/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: qwenModel,
          temperature: 0.2,
          max_tokens: 512,
          messages: [
            { role: 'system', content: buildQwenSystemPrompt(currentStep) },
            ...nextMessages
              .filter((message) => message.role !== 'error')
              .map(({ role, content }) => ({ role, content })),
          ],
        }),
      })
      const payload = await response.json() as QwenCompletion

      if (!response.ok) {
        throw new Error(payload.error?.message || `Qwen endpoint returned ${response.status}`)
      }

      const content = getCompletionText(payload)
      if (!content) throw new Error('Qwen returned an empty response.')

      setChatMessages((messages) => [
        ...messages,
        { id: createMessageId(), role: 'assistant', content },
      ])
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Unknown local Qwen error.'
      setChatMessages((messages) => [
        ...messages,
        { id: createMessageId(), role: 'error', content: `Local Qwen is unavailable: ${detail}` },
      ])
      showNotice('The local Qwen request did not complete.')
    } finally {
      setIsChatLoading(false)
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true">
            <ScanLine size={23} strokeWidth={1.8} />
          </div>
          <div>
            <h1>Robot VLN Demo</h1>
            <p className="eyebrow">Developed by UC Geomatics</p>
          </div>
        </div>

        <div className="system-strip" aria-label="Replay system status">
          <span><Bot size={15} /> Agent: Dataset replay</span>
          <span><Wifi size={15} className="status-good" /> Stream: Local</span>
          <span><BatteryMedium size={16} className="status-good" /> Memory: Ready</span>
          <span><Activity size={15} className="status-accent" /> Model: Qwen3-VL</span>
          <span><Clock3 size={15} /> {formatElapsed(currentStep.elapsedSeconds)}</span>
        </div>
      </header>

      <div className="workspace">
        <aside className="left-rail">
          <section className="panel observation-panel">
            <div className="panel-heading">
              <div className="heading-title"><Camera size={18} /> Robot Observation</div>
              <span className="panel-kicker">RGB</span>
            </div>
            <div className="observation-frame">
              <img src="/assets/demo-bev.gif" alt="Recorded Matterport RGB observation" />
              <div className="feed-chip"><span /> Replay feed</div>
            </div>
            <div className="observation-meta">
              <span><i className="online-dot" /> Perspective view</span>
              <span>15 Hz recorded</span>
            </div>
          </section>

          <section className="panel session-panel">
            <div className="panel-heading">
              <div className="heading-title"><Bot size={18} /> Replay Trajectory</div>
              <span className="status-label">Ready</span>
            </div>
            <div className="session-grid">
              <div><span>Scene</span><strong>VLzqgDo317F</strong></div>
              <div><span>Episode</span><strong>R2R-001</strong></div>
              <div><span>Viewpoint</span><strong>VP-{currentStep.node.toString().padStart(2, '0')}</strong></div>
              <div><span>Mode</span><strong>Offline replay</strong></div>
            </div>
            <div className="instruction-block">
              <span>Navigation instruction</span>
              <p>{instruction}</p>
            </div>
            <div className="task-progress" aria-label={`Subtask ${subtaskNumber} of 3`}>
              <div className="progress-caption">
                <span>Subtask {subtaskNumber}/3</span>
                <strong>{currentStep.subtask}</strong>
              </div>
              <div className="step-dots">
                {[1, 2, 3].map((task) => <i key={task} className={task <= subtaskNumber ? 'active' : ''} />)}
              </div>
            </div>
          </section>

          <section className="panel memory-panel">
            <div className="panel-heading">
              <div className="heading-title"><Layers3 size={18} /> Memory Layers</div>
              <span className="panel-kicker">BEV</span>
            </div>
            <div className="memory-row"><span className="layer-icon metric" /> Metric memory <strong>Active</strong></div>
            <div className="memory-row"><span className="layer-icon topology" /> Topology graph <strong>{currentIndex + 1} nodes</strong></div>
            <div className="memory-row"><span className="layer-icon semantic" /> Semantic grounding <strong>18 classes</strong></div>
          </section>
        </aside>

        <section className="center-stage">
          <section className={`panel map-panel ${isExpanded ? 'map-expanded' : ''}`}>
            <div className="panel-heading map-heading">
              <div className="heading-title"><Map size={19} /> BEV Navigation Map</div>
              <div className="map-actions">
                <div className="segmented-control" aria-label="Map layer">
                  {mapLayers.map((layer) => (
                    <button
                      key={layer.id}
                      type="button"
                      className={mapLayer === layer.id ? 'selected' : ''}
                      onClick={() => setMapLayer(layer.id)}
                    >
                      {layer.label}
                    </button>
                  ))}
                </div>
                <button
                  type="button"
                  className="icon-button"
                  onClick={() => showNotice('Map calibration settings are part of V1.')}
                  title="Map settings"
                  aria-label="Map settings"
                >
                  <Settings2 size={17} />
                </button>
                <button
                  type="button"
                  className={`icon-button ${isExpanded ? 'selected' : ''}`}
                  onClick={() => setIsExpanded((expanded) => !expanded)}
                  title="Expand map"
                  aria-label="Expand map"
                >
                  <Maximize2 size={17} />
                </button>
              </div>
            </div>

            <div className="map-stage" aria-label={`${selectedLayer.label} BEV map, current viewpoint ${currentStep.node}`}>
              <img src={selectedLayer.src} alt={`${selectedLayer.label} bird's-eye-view map`} />
              <div
                className="current-marker"
                style={{ left: `${marker.x}%`, top: `${marker.y}%` }}
                aria-hidden="true"
              >
                <span />
                <b>{currentStep.node}</b>
              </div>
              <div className="map-status"><CircleDot size={15} /> Current node VP-{currentStep.node.toString().padStart(2, '0')}</div>
              <div className="map-floor">Floor 1</div>
            </div>

            <div className="map-footer">
              <div className="legend-items" aria-label="Map legend">
                <span><i className="legend-line" /> Trajectory</span>
                <span><i className="legend-current" /> Current node</span>
                <span><i className="legend-goal" /> Goal region</span>
              </div>
              {mapLayer === 'semantic' && (
                <div className="semantic-legend" aria-label="Semantic classes">
                  {semanticLegend.map(([label, color]) => <span key={label}><i style={{ backgroundColor: color }} />{label}</span>)}
                </div>
              )}
            </div>
          </section>

          <section className="panel playback-panel">
            <div className="playback-controls">
              <div className="transport-buttons">
                <button type="button" className="icon-button" onClick={restart} title="Restart replay" aria-label="Restart replay"><RotateCcw size={18} /></button>
                <button
                  type="button"
                  className="icon-button"
                  disabled={currentIndex === 0}
                  onClick={() => setCurrentIndex((index) => Math.max(0, index - 1))}
                  title="Previous viewpoint"
                  aria-label="Previous viewpoint"
                ><SkipBack size={18} /></button>
                <button type="button" className="play-button" onClick={togglePlayback} title={isPlaying ? 'Pause replay' : 'Play replay'} aria-label={isPlaying ? 'Pause replay' : 'Play replay'}>
                  {isPlaying ? <Pause size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" />}
                </button>
                <button
                  type="button"
                  className="icon-button"
                  disabled={currentIndex === totalSteps}
                  onClick={() => setCurrentIndex((index) => Math.min(totalSteps, index + 1))}
                  title="Next viewpoint"
                  aria-label="Next viewpoint"
                ><SkipForward size={18} /></button>
              </div>
              <div className="scrubber-wrap">
                <input
                  type="range"
                  min="0"
                  max={totalSteps}
                  value={currentIndex}
                  onChange={(event) => {
                    setCurrentIndex(Number(event.target.value))
                    setIsPlaying(false)
                  }}
                  aria-label="Replay progress"
                  style={{ '--progress': `${(currentIndex / totalSteps) * 100}%` } as React.CSSProperties}
                />
                <div className="scrubber-labels"><span>VP-00</span><span>{formatElapsed(currentStep.elapsedSeconds)} / {formatElapsed(replaySteps[totalSteps].elapsedSeconds)}</span><span>VP-10</span></div>
              </div>
              <div className="playback-state"><span className={isPlaying ? 'pulse-dot' : 'idle-dot'} /> {isPlaying ? 'Playing' : currentIndex === totalSteps ? 'Complete' : 'Paused'}</div>
            </div>
            <div className="replay-actions">
              <a className="action-button" href="/assets/map-zoom-color.png" download="spatialqwen-bev-map.png"><Download size={17} /> Export Map</a>
              <button type="button" className="action-button" onClick={() => showNotice('Replay report is ready for V1 session exports.')}><FileText size={17} /> Export Report</button>
            </div>
          </section>
        </section>

        <aside className="right-rail">
          <section className="panel inference-panel">
            <div className="panel-heading">
              <div className="heading-title"><BrainCircuit size={19} /> Navigation Inference</div>
              <span className="live-label">Replay state</span>
            </div>
            <div className="current-action">
              <span>Current action</span>
              <strong>{currentStep.action}</strong>
              <small>{currentStep.location}</small>
            </div>
            <div className="reasoning-grid">
              <div className="reasoning-column">
                <h2>Temporal reasoning</h2>
                <ul>{currentStep.temporalReasoning.map((reason) => <li key={reason}>{reason}</li>)}</ul>
              </div>
              <div className="reasoning-column">
                <h2>Spatial reasoning</h2>
                <ul>{currentStep.spatialReasoning.map((reason) => <li key={reason}>{reason}</li>)}</ul>
              </div>
            </div>
            <div className="confidence-row">
              <span>Navigation confidence</span>
              <div className="confidence-track"><i style={{ width: `${Math.round(currentStep.confidence * 100)}%` }} /></div>
              <strong>{currentStep.confidence.toFixed(2)}</strong>
            </div>
          </section>

          <section className="panel trace-panel">
            <div className="panel-heading">
              <div className="heading-title"><Gauge size={18} /> Action Trace</div>
              <span className="panel-kicker">{currentIndex + 1}/{replaySteps.length}</span>
            </div>
            <ol className="trace-list">
              {replaySteps.slice(Math.max(0, currentIndex - 2), currentIndex + 1).map((step) => (
                <li key={step.node} className={step.node === currentStep.node ? 'current' : 'complete'}>
                  {step.node === currentStep.node ? <CircleDot size={16} /> : <CircleCheck size={16} />}
                  <div><span>VP-{step.node.toString().padStart(2, '0')}</span><strong>{step.action}</strong></div>
                  <ChevronRight size={15} />
                </li>
              ))}
            </ol>
          </section>

          <section className="panel command-panel">
            <div className="panel-heading">
              <div className="heading-title"><MessageSquareText size={18} /> Task Console</div>
              <span className="live-label">Local Qwen</span>
            </div>
            <div className="command-history" ref={chatHistoryRef} aria-live="polite">
              {chatMessages.map((message) => (
                <div key={message.id} className={`chat-message ${message.role}`}>
                  {message.role === 'assistant' ? <Bot size={15} /> : <span className="chat-avatar">{message.role === 'user' ? 'You' : '!'}</span>}
                  <div>
                    <span>{message.role === 'assistant' ? qwenModel : message.role === 'user' ? 'Operator' : 'Connection'}</span>
                    <p>{message.content}</p>
                  </div>
                </div>
              ))}
              {isChatLoading && (
                <div className="chat-message assistant pending">
                  <Loader2 size={15} className="spin" />
                  <div><span>{qwenModel}</span><p>Thinking...</p></div>
                </div>
              )}
            </div>
            <form className="command-form" onSubmit={handleCommand}>
              <input
                value={command}
                onChange={(event) => setCommand(event.target.value)}
                placeholder="Ask the local Qwen..."
                aria-label="Ask the local Qwen"
                disabled={isChatLoading}
              />
              <button type="submit" disabled={isChatLoading} title="Send to local Qwen" aria-label="Send to local Qwen">
                {isChatLoading ? <Loader2 size={17} className="spin" /> : <SendHorizontal size={17} />}
              </button>
            </form>
          </section>

          <section className="panel model-panel">
            <Network size={18} />
            <div><span>Local model</span><strong>{qwenModel} via /api/qwen</strong></div>
            <span className="connection-good">Chat enabled</span>
          </section>
        </aside>
      </div>

      <div className={`toast ${notice ? 'visible' : ''}`} role="status">{notice}</div>
    </main>
  )
}

export default App
