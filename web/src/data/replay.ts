export type MapLayerId = 'color' | 'semantic' | 'candidates'

export interface ReplayStep {
  node: number
  elapsedSeconds: number
  action: string
  subtask: string
  location: string
  temporalReasoning: string[]
  spatialReasoning: string[]
  confidence: number
}

export const instruction =
  'Walk forward, turn right at the stairs, then go down the stairs and stop at the bottom.'

export const mapLayers: Array<{ id: MapLayerId; label: string; src: string }> = [
  { id: 'color', label: 'Color', src: '/assets/map-zoom-color.png' },
  { id: 'semantic', label: 'Semantic', src: '/assets/map-zoom-semantic.png' },
  { id: 'candidates', label: 'Candidates', src: '/assets/map-candidates-zoom-color.png' },
]

export const replaySteps: ReplayStep[] = [
  {
    node: 0,
    elapsedSeconds: 0,
    action: 'Initialize memory',
    subtask: 'Build the first local BEV',
    location: 'Entry viewpoint',
    temporalReasoning: [
      'The replay begins at the entry viewpoint.',
      'Initialize metric and topological memory before moving.',
    ],
    spatialReasoning: [
      'The forward route is observed and traversable.',
      'No obstruction is detected in the near field.',
    ],
    confidence: 0.72,
  },
  {
    node: 1,
    elapsedSeconds: 13,
    action: 'Walk forward',
    subtask: 'Approach the stair junction',
    location: 'Hallway segment',
    temporalReasoning: [
      'The first forward action has completed.',
      'The next landmark remains ahead in the route.',
    ],
    spatialReasoning: [
      'The corridor continues directly ahead.',
      'The right-side opening remains visible in memory.',
    ],
    confidence: 0.8,
  },
  {
    node: 2,
    elapsedSeconds: 27,
    action: 'Walk forward',
    subtask: 'Continue toward the stairs',
    location: 'Main corridor',
    temporalReasoning: [
      'The agent is still executing the forward instruction.',
      'The staircase landmark is becoming more prominent.',
    ],
    spatialReasoning: [
      'The path ahead remains connected to the current node.',
      'The right turn is not yet the safest next move.',
    ],
    confidence: 0.83,
  },
  {
    node: 3,
    elapsedSeconds: 42,
    action: 'Turn right at the stairs',
    subtask: 'Align with the stair approach',
    location: 'Stair junction',
    temporalReasoning: [
      'The forward segment is complete.',
      'The route now requires a right turn at the stairs.',
    ],
    spatialReasoning: [
      'Stairs are visible on the right side of the local map.',
      'The candidate heading aligns with the instruction.',
    ],
    confidence: 0.88,
  },
  {
    node: 4,
    elapsedSeconds: 56,
    action: 'Turn right at the stairs',
    subtask: 'Confirm stair descent route',
    location: 'Stair approach',
    temporalReasoning: [
      'The turn has been initiated and the route is being verified.',
      'The descent task remains active after the turn.',
    ],
    spatialReasoning: [
      'The staircase edge is within the observed local region.',
      'The selected waypoint preserves a connected path.',
    ],
    confidence: 0.9,
  },
  {
    node: 5,
    elapsedSeconds: 71,
    action: 'Go down the stairs',
    subtask: 'Begin descent',
    location: 'Top landing',
    temporalReasoning: [
      'The turn is complete and the descent phase begins.',
      'Continue until the lower landing can be confirmed.',
    ],
    spatialReasoning: [
      'The stair direction is reachable from the current node.',
      'No competing waypoint has stronger route alignment.',
    ],
    confidence: 0.87,
  },
  {
    node: 6,
    elapsedSeconds: 87,
    action: 'Go down the stairs',
    subtask: 'Continue descent',
    location: 'Stairwell',
    temporalReasoning: [
      'The lower level is not yet confirmed.',
      'The agent must keep descending before it can stop.',
    ],
    spatialReasoning: [
      'The current local radius does not contain the destination.',
      'The route remains constrained by the stair geometry.',
    ],
    confidence: 0.84,
  },
  {
    node: 7,
    elapsedSeconds: 103,
    action: 'Go down the stairs',
    subtask: 'Search for lower landing',
    location: 'Lower stair segment',
    temporalReasoning: [
      'The destination has not entered the visible local area.',
      'Continue the same high-level instruction.',
    ],
    spatialReasoning: [
      'The path is still supported by the metric memory.',
      'The lower route is clearer than backtracking.',
    ],
    confidence: 0.82,
  },
  {
    node: 8,
    elapsedSeconds: 119,
    action: 'Go down the stairs',
    subtask: 'Reach the lower landing',
    location: 'Lower landing approach',
    temporalReasoning: [
      'The replay is nearing the final target region.',
      'Stop only after the lower landing is spatially verified.',
    ],
    spatialReasoning: [
      'The selected node is closer to the inferred destination.',
      'No dynamic obstruction is present in the replay asset.',
    ],
    confidence: 0.89,
  },
  {
    node: 9,
    elapsedSeconds: 135,
    action: 'Verify stop position',
    subtask: 'Check destination proximity',
    location: 'Bottom landing',
    temporalReasoning: [
      'The action sequence is complete pending final confirmation.',
      'Evaluate whether the current node satisfies the stop condition.',
    ],
    spatialReasoning: [
      'The inferred target region overlaps the local destination area.',
      'The route is complete without requiring a detour.',
    ],
    confidence: 0.93,
  },
  {
    node: 10,
    elapsedSeconds: 151,
    action: 'Stop',
    subtask: 'Navigation complete',
    location: 'Bottom landing',
    temporalReasoning: [
      'All requested navigation steps have been completed.',
      'The replay can now be exported or restarted.',
    ],
    spatialReasoning: [
      'The final viewpoint lies in the inferred destination region.',
      'The final position is consistent with the completed route.',
    ],
    confidence: 0.96,
  },
]

export const markerPositions = [
  { x: 54, y: 7 },
  { x: 56, y: 15 },
  { x: 98, y: 13 },
  { x: 89, y: 18 },
  { x: 76, y: 25 },
  { x: 66, y: 30 },
  { x: 54, y: 37 },
  { x: 43, y: 42 },
  { x: 62, y: 54 },
  { x: 55, y: 58 },
  { x: 50, y: 50 },
]

export const semanticLegend = [
  ['Wall', '#aac4e4'],
  ['Floor', '#667788'],
  ['Chair', '#91d57e'],
  ['Table', '#ff7b17'],
  ['Sofa', '#2ba333'],
  ['Bed', '#d269ba'],
]
