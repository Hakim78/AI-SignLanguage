interface Prediction {
  label: string;
  probability: number;
}

interface PredictionPanelProps {
  status: 'loading' | 'ready' | 'running' | 'error';
  statusText: string;
  loadingProgress: number;
  prediction: string;
  confidence: number;
  topPredictions: Prediction[];
  fps: number;
  latency: number;
  t: (key: string) => string;
}

export function PredictionPanel({
  status,
  statusText,
  loadingProgress,
  prediction,
  confidence,
  topPredictions,
  fps,
  latency,
  t
}: PredictionPanelProps) {
  const statusClass = {
    loading: 'text-yellow-400 border-yellow-500/30',
    ready: 'text-green-400 border-green-500/30',
    running: 'text-violet-400 border-violet-500/30',
    error: 'text-red-400 border-red-500/30'
  }[status];

  return (
    <div className="glass-card p-6 flex flex-col gap-5">
      {/* Status */}
      <div>
        <div className={`text-center py-3 px-4 rounded-xl bg-white/5 border ${statusClass}`}>
          {statusText}
        </div>
        {status === 'loading' && (
          <div className="mt-3 h-1.5 bg-white/10 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-violet-500 to-pink-500 rounded-full transition-all duration-300"
              style={{ width: `${loadingProgress}%` }}
            />
          </div>
        )}
      </div>

      {/* Main Prediction */}
      <div className="text-center py-10 px-4 bg-gradient-to-br from-violet-500/5 to-pink-500/5 rounded-2xl border border-violet-500/10">
        <div className={`font-extrabold leading-none ${
          prediction === '-'
            ? 'text-5xl text-gray-600'
            : 'text-[7rem] gradient-text'
        }`}>
          {prediction}
        </div>
        <div className="mt-4 text-xl font-semibold">
          {confidence > 0 ? (
            <span className="gradient-text">{confidence}% {t('confidence')}</span>
          ) : (
            <span className="text-gray-500">--</span>
          )}
        </div>
      </div>

      {/* Top Predictions */}
      <div>
        <h4 className="text-sm text-gray-400 mb-3 flex items-center gap-2">
          <svg viewBox="0 0 24 24" className="w-4 h-4" fill="currentColor">
            <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
          </svg>
          {t('topPredictions')}
        </h4>
        <div className="space-y-2">
          {topPredictions.length > 0 ? (
            topPredictions.map((pred, i) => (
              <div key={i} className="flex items-center gap-3">
                <span className="w-12 font-semibold text-sm">{pred.label}</span>
                <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-violet-500 to-pink-500 rounded-full transition-all duration-200"
                    style={{ width: `${pred.probability}%` }}
                  />
                </div>
                <span className="w-10 text-right text-xs text-gray-400">
                  {pred.probability}%
                </span>
              </div>
            ))
          ) : (
            <div className="text-gray-600 text-sm py-4 text-center">--</div>
          )}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 pt-4 border-t border-white/5">
        <div className="text-center">
          <div className="text-2xl font-bold gradient-text">{fps}</div>
          <div className="text-[0.65rem] text-gray-500 uppercase tracking-wider mt-1">FPS</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold gradient-text">{latency}</div>
          <div className="text-[0.65rem] text-gray-500 uppercase tracking-wider mt-1">{t('latency')}</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold gradient-text">79K</div>
          <div className="text-[0.65rem] text-gray-500 uppercase tracking-wider mt-1">{t('params')}</div>
        </div>
      </div>
    </div>
  );
}
