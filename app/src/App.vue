<script setup lang="ts">
import Chart from 'chart.js/auto'
import type { ChartData } from 'chart.js'
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { read, utils } from 'xlsx'
import {
  classifyEmotion,
  classifyEmotionBatch,
  type ClassifyEmotionResult,
  MAX_BATCH_SIZE,
  chunkIntoBlocks,
  LANGUAGE_LABELS,
  type LanguageCode,
} from '../gemini'

type Mode = 'single' | 'excel'

const MAX_EXCEL_SENTENCES = 10
const EXCEL_BATCH_SIZE = MAX_BATCH_SIZE
const NAV_ITEMS: Array<{ key: Mode; label: string; description: string }> = [
  { key: 'single', label: 'Single Sentence', description: 'Manual entry for one message at a time' },
  {
    key: 'excel',
    label: 'Excel Upload',
    description: `Upload up to ${MAX_EXCEL_SENTENCES} rows (processed ${EXCEL_BATCH_SIZE} at a time)`,
  },
]
const CHART_LABELS = ['joy', 'love', 'anger', 'fear', 'sadness', 'surprise', 'unknown'] as const
type ChartLabel = (typeof CHART_LABELS)[number]
const CHART_COLORS: Record<ChartLabel, string> = {
  joy: '#22c55e',
  love: '#f97316',
  anger: '#ef4444',
  fear: '#6366f1',
  sadness: '#0ea5e9',
  surprise: '#eab308',
  unknown: '#94a3b8',
}

const LANGUAGE_OPTIONS = (Object.entries(LANGUAGE_LABELS) as [LanguageCode, string][]).map(
  ([key, label]) => ({
    key,
    label,
  }),
)

const apiKey = ref('')
const activeMode = ref<Mode>('single')
const sentence = ref('')
const singleContext = ref('')
const selectedLanguage = ref<LanguageCode>('english')
const excelSentences = ref<string[]>([])
const excelFileName = ref<string | null>(null)
const excelNotice = ref<string | null>(null)
const predictions = ref<ClassifyEmotionResult[]>([])
const loading = ref(false)
const error = ref<string | null>(null)
const showTutorial = ref(false)

const chartCanvas = ref<HTMLCanvasElement | null>(null)
let chartInstance: Chart<'pie'> | null = null
const resolveColor = (label: string | null) => {
  const key = (label ?? 'unknown') as ChartLabel
  return CHART_COLORS[key]
}

const sanitizedApiKey = computed(() => apiKey.value.trim())
const hasApiKey = computed(() => Boolean(sanitizedApiKey.value))
const hasResults = computed(() => predictions.value.length > 0)
const canSubmitSingle = computed(
  () => Boolean(sentence.value.trim()) && !loading.value && hasApiKey.value,
)
const canSubmitExcel = computed(
  () => excelSentences.value.length > 0 && !loading.value && hasApiKey.value,
)
const latestSingleResult = computed(() =>
  activeMode.value === 'single' ? predictions.value[0] ?? null : null,
)
const highlightedEmotion = computed(() => latestSingleResult.value?.predictedEmotion ?? null)
const highlightedSentence = computed(() => latestSingleResult.value?.sentence ?? null)
const excelBlocks = computed(() =>
  chunkIntoBlocks(excelSentences.value, EXCEL_BATCH_SIZE).map((block, blockIndex) => ({
    blockId: blockIndex + 1,
    sentences: block.map((text, localIndex) => ({
      localId: localIndex + 1,
      text,
      globalNumber: blockIndex * EXCEL_BATCH_SIZE + localIndex + 1,
    })),
  })),
)
const predictionBlocks = computed(() =>
  chunkIntoBlocks(predictions.value, EXCEL_BATCH_SIZE).map((block, blockIndex) => ({
    blockId: blockIndex + 1,
    results: block.map((result, localIndex) => ({
      localId: localIndex + 1,
      globalNumber: blockIndex * EXCEL_BATCH_SIZE + localIndex + 1,
      result,
    })),
  })),
)

const emotionCounts = computed(() => {
  const counts: Record<ChartLabel, number> = {
    joy: 0,
    love: 0,
    anger: 0,
    fear: 0,
    sadness: 0,
    surprise: 0,
    unknown: 0,
  }

  for (const item of predictions.value) {
    const label = (item.predictedEmotion ?? 'unknown') as ChartLabel
    counts[label] += 1
  }

  return counts
})

const pieData = computed<ChartData<'pie'> | null>(() => {
  const labels = CHART_LABELS.filter((label) => emotionCounts.value[label] > 0)
  if (!labels.length) return null

  return {
    labels,
    datasets: [
      {
        data: labels.map((label) => emotionCounts.value[label]),
        backgroundColor: labels.map((label) => CHART_COLORS[label]),
        borderWidth: 0,
      },
    ],
  }
})

const statusMessage = computed(() => {
  if (!loading.value) return null
  if (activeMode.value === 'excel') {
    return 'Analyzing Excel file...'
  }
  return 'Analyzing sentence...'
})

const tutorialSections = [
  {
    title: 'Single sentence mode',
    steps: [
      'Enter your Gemini API key to unlock the analyze buttons.',
      'Type or paste one sentence into the text area.',
      'Choose "Analyze this sentence" to see the predicted emotion on the right.',
    ],
  },
  {
    title: 'Excel upload mode',
    steps: [
      'Switch to Excel Upload in the sidebar.',
      `Upload a .xlsx or .xls file with up to ${MAX_EXCEL_SENTENCES} non-empty cells.`,
      'Press "Analyze selected sentences" to populate the chart and list below.',
    ],
  },
  {
    title: 'Tips',
    steps: [
      'Only non-empty cells are included and trimmed automatically.',
      'Results stay local to your browser session.',
      'Switching modes clears existing outputs so you can start clean.',
    ],
  },
] as const

const teardownChart = () => {
  if (chartInstance) {
    chartInstance.destroy()
    chartInstance = null
  }
}

const updateChart = async () => {
  await nextTick()
  const canvas = chartCanvas.value
  const data = pieData.value

  if (!canvas || !data) {
    teardownChart()
    return
  }

  const context = canvas.getContext('2d')
  if (!context) return

  if (!chartInstance) {
    chartInstance = new Chart<'pie'>(context, {
      type: 'pie',
      data,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              color: '#cbd5f5',
              usePointStyle: true,
            },
          },
          tooltip: {
            callbacks: {
              label: (tooltipItem) => {
                const label = tooltipItem.label ?? ''
                const value = tooltipItem.raw as number
                const unit = value === 1 ? 'sentence' : 'sentences'
                return `${label}: ${value} ${unit}`
              },
            },
          },
        },
      },
    })
    return
  }

  chartInstance.data.labels = data.labels
  chartInstance.data.datasets = data.datasets
  chartInstance.update()
}

const resetStateForMode = (mode: Mode) => {
  error.value = null
  predictions.value = []

  if (mode === 'single') {
    excelSentences.value = []
    excelFileName.value = null
    excelNotice.value = null
  } else {
    sentence.value = ''
    singleContext.value = ''
  }
}

watch([pieData, chartCanvas], () => {
  updateChart()
})

watch(activeMode, (mode) => {
  resetStateForMode(mode)
})

watch(apiKey, () => {
  const currentError = error.value
  if (currentError && currentError.toLowerCase().includes('api key')) {
    error.value = null
  }
})

const requireApiKey = (): string | null => {
  const key = sanitizedApiKey.value
  if (!key) {
    error.value = 'Please enter your Gemini API key.'
    return null
  }
  return key
}

const classifySingle = async () => {
  if (!sentence.value.trim() || loading.value) return

  const key = requireApiKey()
  if (!key) return

  loading.value = true
  error.value = null
  predictions.value = []

  try {
    const contextPayload = singleContext.value.trim() || undefined
    const response = await classifyEmotion(
      sentence.value.trim(),
      key,
      contextPayload,
      selectedLanguage.value,
    )
    predictions.value = [response]
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Unable to analyze the sentence.'
  } finally {
    loading.value = false
  }
}

const classifyExcel = async () => {
  if (!excelSentences.value.length || loading.value) return

  const key = requireApiKey()
  if (!key) return

  loading.value = true
  error.value = null
  predictions.value = []

  try {
    const results = await classifyEmotionBatch(excelSentences.value, key, selectedLanguage.value)
    if (!results.length) {
      throw new Error('No valid sentences found in the file.')
    }
    predictions.value = results
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Unable to analyze the Excel file.'
  } finally {
    loading.value = false
  }
}

const handleExcelUpload = async (event: Event) => {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0] ?? null
  input.value = ''
  if (!file) return

  excelFileName.value = file.name
  error.value = null
  excelNotice.value = null
  predictions.value = []

  try {
    const data = await file.arrayBuffer()
    const workbook = read(data, { type: 'array' })
    const sheetName = workbook.SheetNames[0]
    if (!sheetName) throw new Error('The workbook is empty.')

    const worksheet = workbook.Sheets[sheetName]
    if (!worksheet) throw new Error('Could not read the first sheet.')

    const rows = utils.sheet_to_json<(string | number | null)[]>(worksheet, { header: 1 })

    const picked: string[] = []
    let validCount = 0

    for (const row of rows) {
      for (const cell of row) {
        const normalized =
          cell === null || cell === undefined
            ? ''
            : typeof cell === 'string'
              ? cell.trim()
              : String(cell).trim()

        if (!normalized) continue
        validCount += 1
        if (picked.length < MAX_EXCEL_SENTENCES) {
          picked.push(normalized)
        }
      }
    }

    if (!picked.length) {
      excelSentences.value = []
      throw new Error('No valid sentences found in the file.')
    }

    excelSentences.value = picked
    excelNotice.value =
      validCount > MAX_EXCEL_SENTENCES
        ? `Captured the first ${MAX_EXCEL_SENTENCES}/${validCount} sentences from the sheet.`
        : null
  } catch (err) {
    excelSentences.value = []
    excelNotice.value = null
    error.value = err instanceof Error ? err.message : 'Unable to read the Excel file.'
  }
}

const closeTutorial = () => {
  showTutorial.value = false
}

const openTutorial = () => {
  showTutorial.value = true
}

const handleEscapeKey = (event: KeyboardEvent) => {
  if (event.key === 'Escape') {
    closeTutorial()
  }
}

onMounted(() => {
  window.addEventListener('keydown', handleEscapeKey)
})

onUnmounted(() => {
  teardownChart()
  window.removeEventListener('keydown', handleEscapeKey)
})
</script>

<template>
  <div class="flex h-screen w-screen bg-white text-slate-900">
    <aside class="flex w-72 flex-col border-r border-slate-200 bg-slate-50">
      <div class="px-6 pb-6 pt-8">
        <p class="text-xs font-semibold uppercase tracking-[0.3em] text-slate-500">EmoGuest</p>
        <h1 class="mt-2 text-2xl font-bold text-slate-900">Emotion Lab</h1>
        <p class="mt-2 text-sm text-slate-500">Choose how you want to feed sentences into the analyzer.</p>
      </div>
      <nav class="space-y-2 px-4">
        <button
          v-for="item in NAV_ITEMS"
          :key="item.key"
          type="button"
          class="w-full rounded-2xl border px-4 py-4 text-left transition"
          :class="
            activeMode === item.key
              ? 'border-slate-900/10 bg-white shadow-lg shadow-slate-200 text-slate-900'
              : 'border-transparent text-slate-500 hover:border-slate-200 hover:bg-white'
          "
          @click="activeMode = item.key"
        >
          <p class="text-base font-semibold">{{ item.label }}</p>
          <p class="text-sm text-slate-400">{{ item.description }}</p>
        </button>
      </nav>
      <div class="mt-auto px-6 pb-8 pt-6 text-xs text-slate-500">
        <p>API: Gemini {{ activeMode === 'single' ? 'Sentence' : 'Excel' }} mode</p>
        <p class="mt-1">
          Handles up to {{ MAX_EXCEL_SENTENCES }} sentences per upload ({{ EXCEL_BATCH_SIZE }} analyzed per request).
        </p>
      </div>
    </aside>

    <main class="flex flex-1 flex-col overflow-hidden bg-slate-50">
      <header class="border-b border-slate-200 px-10 py-6 bg-white">
        <div class="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
          <div>
            <p class="text-xs font-semibold uppercase tracking-[0.3em] text-slate-500">Insights</p>
            <h2 class="text-2xl font-bold leading-tight">
              {{ activeMode === 'single' ? 'Analyze a single sentence' : 'Analyze sentences from Excel' }}
            </h2>
          </div>
          <div class="flex items-center justify-end gap-3 text-sm text-slate-500">
            <span v-if="statusMessage">{{ statusMessage }}</span>
            <span v-else>Gemini 2.5 Flash · Live pie chart</span>
            <button
              type="button"
              class="inline-flex h-9 w-9 items-center justify-center rounded-full border border-slate-200 bg-white text-base font-semibold text-slate-600 transition hover:bg-slate-100 focus:outline-none focus:ring-2 focus:ring-sky-200"
              aria-label="Open tutorial"
              @click="openTutorial"
            >
              i
            </button>
          </div>
        </div>
      </header>

      <section class="flex-1 overflow-y-auto px-6 py-8 sm:px-10">
        <div class="mx-auto flex max-w-6xl flex-col gap-6">
          <div class="grid gap-6 lg:grid-cols-[minmax(0,420px)_minmax(0,1fr)]">
            <div class="rounded-3xl border border-slate-200 bg-white p-6 shadow-[0_25px_60px_rgba(15,23,42,0.08)]">
              <h3 class="text-lg font-semibold text-slate-900">
                {{ activeMode === 'single' ? 'Quick single-sentence run' : 'Batch Excel run' }}
              </h3>
              <p class="mt-2 text-sm text-slate-500">
  {{
    activeMode === 'single'
      ? 'Type or paste any sentence to predict its dominant emotion using Gemini.'
      : `Upload .xlsx/.xls where each sentence lives in its own cell. Up to ${MAX_EXCEL_SENTENCES} sentences will be processed in batches of ${EXCEL_BATCH_SIZE}.`
  }}
              </p>

              <div class="mt-6 space-y-4">
                <div class="space-y-2">
                  <label class="text-sm font-semibold text-slate-700" for="api-key-input">Gemini API key</label>
                  <input
                    id="api-key-input"
                    v-model="apiKey"
                    type="password"
                    autocomplete="off"
                    spellcheck="false"
                    placeholder="Paste your Gemini API key"
                    class="w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 placeholder:text-slate-400 focus:border-sky-400 focus:outline-none focus:ring-4 focus:ring-sky-200"
                  />
                  <p class="text-xs text-slate-500">Stored locally and only used for the requests you trigger.</p>
                  <p
                    v-if="!hasApiKey"
                    class="text-xs font-medium text-amber-600"
                  >
                    Enter your API key to enable the analysis actions.
                  </p>
                </div>

                <div class="space-y-2">
                  <label class="text-sm font-semibold text-slate-700" for="language-select">Input language</label>
                  <select
                    id="language-select"
                    v-model="selectedLanguage"
                    class="w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-900 focus:border-sky-400 focus:outline-none focus:ring-4 focus:ring-sky-200"
                  >
                    <option
                      v-for="option in LANGUAGE_OPTIONS"
                      :key="option.key"
                      :value="option.key"
                    >
                      {{ option.label }}
                    </option>
                  </select>
                  <p class="text-xs text-slate-500">Gemini will expect sentences in the selected language.</p>
                </div>

                <form v-if="activeMode === 'single'" class="space-y-4" @submit.prevent="classifySingle">
                  <label class="text-sm font-semibold text-slate-700" for="sentence-input">Sentence</label>
                  <textarea
                    id="sentence-input"
                    v-model="sentence"
                    rows="5"
                    placeholder="Example: I feel incredibly happy about finishing this project."
                    class="w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 placeholder:text-slate-400 focus:border-sky-400 focus:outline-none focus:ring-4 focus:ring-sky-200"
                  ></textarea>
                  <label class="text-sm font-semibold text-slate-700" for="context-input">Context (optional)</label>
                  <textarea
                    id="context-input"
                    v-model="singleContext"
                    rows="3"
                    placeholder="Add background or scenario details that help interpret the sentence."
                    class="w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 placeholder:text-slate-400 focus:border-sky-400 focus:outline-none focus:ring-4 focus:ring-sky-200"
                  ></textarea>

                  <button
                    type="submit"
                    :disabled="!canSubmitSingle"
                    class="inline-flex w-full items-center justify-center gap-3 rounded-2xl bg-gradient-to-r from-sky-500 to-blue-600 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-sky-200 transition hover:from-sky-400 hover:to-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    <span
                      v-if="loading"
                      class="h-4 w-4 animate-spin rounded-full border-2 border-white/70 border-t-transparent"
                      aria-hidden="true"
                    ></span>
                    {{ loading ? 'Analyzing...' : 'Analyze this sentence' }}
                  </button>
                </form>

                <div v-else class="space-y-4">
                  <label class="text-sm font-semibold text-slate-700">Upload Excel file</label>
                  <label
                    class="flex cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-slate-300 bg-slate-100 px-4 py-8 text-center transition hover:border-slate-400 hover:bg-white"
                  >
                    <input
                      accept=".xlsx,.xls"
                      class="sr-only"
                      type="file"
                      @change="handleExcelUpload"
                    />
                    <svg
                      class="h-10 w-10 text-sky-500"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      stroke-width="1.5"
                    >
                      <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 9 12 4.5 7.5 9M12 4.5V15"
                      />
                    </svg>
                    <div>
                      <p class="font-semibold text-slate-900">Choose Excel file</p>
                      <p class="text-xs text-slate-500">
                        Supports .xlsx and .xls · up to {{ MAX_EXCEL_SENTENCES }} sentences ({{ EXCEL_BATCH_SIZE }} per request)
                      </p>
                    </div>
                  </label>

                  <div v-if="excelFileName" class="rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm">
                    <p class="truncate text-slate-900">File: {{ excelFileName }}</p>
                    <p v-if="excelNotice" class="text-xs text-amber-600">{{ excelNotice }}</p>
                  </div>

                  <div v-if="excelBlocks.length" class="space-y-3 text-sm">
                    <div class="flex items-center justify-between">
                      <p class="font-semibold text-slate-700">Blocks queued for analysis:</p>
                      <span class="text-xs text-slate-500">Each block holds up to {{ EXCEL_BATCH_SIZE }} sentences</span>
                    </div>
                    <div class="space-y-2">
                      <article
                        v-for="block in excelBlocks"
                        :key="`block-${block.blockId}`"
                        class="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3"
                      >
                        <div class="flex items-center justify-between text-xs uppercase tracking-[0.2em] text-slate-500">
                          <span>Block {{ block.blockId }}</span>
                          <span>{{ block.sentences.length }}/{{ EXCEL_BATCH_SIZE }} sentences</span>
                        </div>
                        <ol class="mt-2 space-y-2 text-slate-600">
                          <li
                            v-for="sentence in block.sentences"
                            :key="`block-${block.blockId}-${sentence.localId}`"
                            class="flex gap-3"
                          >
                            <span class="text-slate-400">{{ block.blockId }}.{{ sentence.localId }}</span>
                            <span class="text-slate-900">{{ sentence.text }}</span>
                          </li>
                        </ol>
                      </article>
                    </div>
                  </div>

                  <button
                    type="button"
                    :disabled="!canSubmitExcel"
                    class="inline-flex w-full items-center justify-center gap-3 rounded-2xl bg-gradient-to-r from-emerald-500 to-green-500 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-emerald-200 transition hover:from-emerald-400 hover:to-green-500 disabled:cursor-not-allowed disabled:opacity-50"
                    @click="classifyExcel"
                  >
                    <span
                      v-if="loading"
                      class="h-4 w-4 animate-spin rounded-full border-2 border-white/70 border-t-transparent"
                      aria-hidden="true"
                    ></span>
                    {{ loading ? 'Processing file...' : 'Analyze selected sentences' }}
                  </button>
                </div>

                <div
                  v-if="error"
                  class="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
                >
                  {{ error }}
                </div>
              </div>
            </div>

            <div class="rounded-3xl border border-slate-200 bg-white p-6 shadow-[0_25px_60px_rgba(15,23,42,0.08)]">
              <div class="flex items-center justify-between">
                <div>
                  <p class="text-xs uppercase tracking-[0.3em] text-slate-500">
                    {{ activeMode === 'single' ? 'Result' : 'Chart' }}
                  </p>
                  <h3 class="mt-1 text-xl font-semibold text-slate-900">
                    {{ activeMode === 'single' ? 'Emotion insight' : 'Emotion distribution' }}
                  </h3>
                </div>
                <span class="rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-500">
                  <template v-if="activeMode === 'single'">
                    {{ highlightedEmotion ? 'Result ready' : 'Awaiting sentence' }}
                  </template>
                  <template v-else>
                    {{ hasResults ? `${predictions.length} sentences` : 'Awaiting data' }}
                  </template>
                </span>
              </div>
              <div class="mt-4 flex h-80 items-center justify-center">
                <template v-if="activeMode === 'single'">
                  <div
                    v-if="highlightedEmotion"
                    class="flex w-full max-w-md flex-col items-center gap-4 rounded-3xl border border-slate-200 bg-slate-50 p-8 text-center"
                  >
                    <p class="text-xs uppercase tracking-[0.3em] text-slate-500">Predicted emotion</p>
                    <p class="text-4xl font-black capitalize text-slate-900">{{ highlightedEmotion }}</p>
                    <p v-if="highlightedSentence" class="text-sm text-slate-600">
                      “{{ highlightedSentence }}”
                    </p>
                  </div>
                  <p v-else class="text-sm text-slate-500">Submit a sentence to preview its emotion here.</p>
                </template>
                <template v-else>
                  <canvas
                    v-if="hasResults"
                    ref="chartCanvas"
                    class="h-full w-full"
                  ></canvas>
                  <p v-else class="text-sm text-slate-500">Run an analysis to generate the pie chart.</p>
                </template>
              </div>
            </div>
          </div>

          <div class="rounded-3xl border border-slate-200 bg-white p-6 shadow-[0_25px_60px_rgba(15,23,42,0.08)]">
            <div class="flex items-center justify-between">
              <div>
                <p class="text-xs uppercase tracking-[0.3em] text-slate-500">Details</p>
                <h3 class="text-lg font-semibold text-slate-900">Analyzed sentences</h3>
              </div>
              <span class="text-sm text-slate-500">{{ hasResults ? 'Complete' : 'Waiting for input' }}</span>
            </div>

            <div class="mt-4 space-y-4" v-if="hasResults">
              <article
                v-for="block in predictionBlocks"
                :key="`prediction-block-${block.blockId}`"
                class="rounded-2xl border border-slate-200 bg-slate-50 p-4"
              >
                <div class="flex flex-wrap items-center justify-between gap-3 text-sm">
                  <p class="font-semibold text-slate-900">Block {{ block.blockId }}</p>
                  <span class="text-xs text-slate-500">{{ block.results.length }}/{{ EXCEL_BATCH_SIZE }} sentences</span>
                </div>
                <div class="mt-3 space-y-3">
                  <div
                    v-for="entry in block.results"
                    :key="`prediction-${block.blockId}-${entry.localId}`"
                    class="rounded-2xl border border-slate-200 bg-white px-4 py-3"
                  >
                    <div class="flex flex-wrap items-center justify-between gap-3 text-sm">
                      <p class="font-semibold text-slate-900">
                        Block {{ block.blockId }} · Sentence {{ entry.localId }} (#{{ entry.globalNumber }})
                      </p>
                      <span
                        class="rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wide"
                        :style="{
                          backgroundColor: `${resolveColor(entry.result.predictedEmotion)}20`,
                          color: resolveColor(entry.result.predictedEmotion),
                        }"
                      >
                        {{ (entry.result.predictedEmotion ?? 'unknown').toUpperCase() }}
                      </span>
                    </div>
                    <p class="mt-2 text-sm text-slate-600">{{ entry.result.sentence }}</p>
                  </div>
                </div>
              </article>
            </div>
            <p v-else class="text-sm text-slate-500">Run an analysis and every sentence will appear here with its label.</p>
          </div>
        </div>
      </section>
    </main>

    <teleport to="body">
      <div
        v-if="showTutorial"
        class="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 px-4 py-8"
        @click.self="closeTutorial"
      >
        <div class="w-full max-w-3xl rounded-3xl bg-white p-6 shadow-2xl">
          <div class="flex items-start justify-between gap-4">
            <div>
              <p class="text-xs font-semibold uppercase tracking-[0.3em] text-slate-500">Tutorial</p>
              <h3 class="mt-1 text-2xl font-bold text-slate-900">Getting started with EmoGuest</h3>
              <p class="mt-2 text-sm text-slate-500">
                Follow these quick steps whenever you need a refresher on how to use the analyzer.
              </p>
            </div>
            <button
              type="button"
              class="inline-flex h-9 w-9 items-center justify-center rounded-full border border-slate-200 text-sm font-semibold text-slate-500 transition hover:bg-slate-100 focus:outline-none focus:ring-2 focus:ring-sky-200"
              aria-label="Close tutorial"
              @click="closeTutorial"
            >
              &times;
            </button>
          </div>
          <div class="mt-6 grid gap-6 md:grid-cols-2">
            <article
              v-for="section in tutorialSections"
              :key="section.title"
              class="rounded-2xl border border-slate-200 bg-slate-50 p-4"
            >
              <h4 class="text-base font-semibold text-slate-900">{{ section.title }}</h4>
              <ul class="mt-3 space-y-2 text-sm text-slate-600">
                <li v-for="step in section.steps" :key="step">{{ step }}</li>
              </ul>
            </article>
          </div>
          <div class="mt-6 text-right">
            <button
              type="button"
              class="inline-flex items-center justify-center rounded-2xl bg-slate-900 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-800"
              @click="closeTutorial"
            >
              Got it
            </button>
          </div>
        </div>
      </div>
    </teleport>
  </div>
</template>
