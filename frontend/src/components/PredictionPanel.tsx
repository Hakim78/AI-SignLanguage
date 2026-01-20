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
  return (
    <div className="card p-4 sm:p-5 h-full flex flex-col">
      {/* Status */}
      <div className="mb-4">
        <div className={`flex items-center justify-center gap-2 py-2.5 px-4 rounded-lg bg-white/[0.02] border border-white/[0.06] text-sm ${
          status === 'loading' ? 'status-loading' :
          status === 'ready' ? 'status-ready' :
          status === 'running' ? 'status-running' : 'status-error'
        }`}>
          {status === 'loading' && (
            <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
            </svg>
          )}
          {status === 'ready' && (
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          )}
          {status === 'running' && (
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-violet-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-violet-500"></span>
            </span>
          )}
          <span className="font-medium">{statusText}</span>
        </div>
        {status === 'loading' && (
          <div className="mt-2 h-1 bg-white/5 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-violet-600 to-purple-600 rounded-full transition-all duration-300"
              style={{ width: `${loadingProgress}%` }}
            />
          </div>
        )}
      </div>

      {/* Main prediction */}
      <div className={`flex-1 flex flex-col items-center justify-center py-6 sm:py-8 rounded-xl border mb-4 transition-all duration-300 ${
        confidence >= 90
          ? 'bg-green-500/10 border-green-500/30'
          : 'bg-white/[0.01] border-white/[0.04]'
      }`}>
        <div className={`font-bold leading-none transition-transform duration-200 ${
          prediction === '-'
            ? 'text-4xl sm:text-5xl text-zinc-700'
            : confidence >= 90
              ? 'text-6xl sm:text-7xl lg:text-8xl text-gradient scale-110'
              : 'text-6xl sm:text-7xl lg:text-8xl text-gradient'
        }`}>
          {prediction}
        </div>
        {confidence > 0 && (
          <div className="mt-3 flex items-center gap-2">
            {confidence >= 90 && (
              <svg className="w-4 h-4 text-green-500 animate-bounce" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            )}
            <span className={`text-sm ${confidence >= 90 ? 'text-green-400 font-semibold' : 'text-zinc-400'}`}>
              <span className={confidence >= 90 ? 'text-green-400' : 'text-violet-400'}>{confidence}%</span> {t('confidence')}
            </span>
          </div>
        )}
      </div>

      {/* Top predictions */}
      <div className="mb-4">
        <h4 className="text-xs font-medium text-zinc-500 uppercase tracking-wide mb-3">{t('topPredictions')}</h4>
        <div className="space-y-2">
          {topPredictions.length > 0 ? (
            topPredictions.slice(0, 5).map((pred, i) => (
              <div key={i} className={`flex items-center gap-3 p-2 rounded-lg transition-all duration-200 ${
                i === 0 && pred.probability >= 80 ? 'bg-violet-500/10' : ''
              }`}>
                <span className={`w-8 text-sm font-medium ${
                  i === 0 && pred.probability >= 80 ? 'text-violet-300' : 'text-zinc-300'
                }`}>{pred.label}</span>
                <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-300 ${
                      i === 0
                        ? 'bg-gradient-to-r from-violet-600 to-purple-500'
                        : 'bg-gradient-to-r from-zinc-600 to-zinc-500'
                    }`}
                    style={{ width: `${pred.probability}%` }}
                  />
                </div>
                <span className={`w-12 text-right text-xs font-medium ${
                  i === 0 && pred.probability >= 80 ? 'text-violet-400' : 'text-zinc-500'
                }`}>{pred.probability}%</span>
              </div>
            ))
          ) : (
            <div className="text-zinc-700 text-sm text-center py-6 bg-white/[0.02] rounded-lg">
              {t('waitingCam')}
            </div>
          )}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-2 pt-4 border-t border-white/[0.04]">
        <div className="text-center p-2 rounded-lg bg-white/[0.02] hover:bg-white/[0.04] transition-colors">
          <div className={`text-lg sm:text-xl font-semibold ${
            fps >= 25 ? 'text-green-400' : fps >= 15 ? 'text-yellow-400' : 'text-violet-400'
          }`}>{fps}</div>
          <div className="text-[10px] text-zinc-500 uppercase tracking-wide mt-0.5">FPS</div>
        </div>
        <div className="text-center p-2 rounded-lg bg-white/[0.02] hover:bg-white/[0.04] transition-colors">
          <div className={`text-lg sm:text-xl font-semibold ${
            latency <= 50 ? 'text-green-400' : latency <= 100 ? 'text-yellow-400' : 'text-violet-400'
          }`}>{latency}</div>
          <div className="text-[10px] text-zinc-500 uppercase tracking-wide mt-0.5">MS</div>
        </div>
        <div className="text-center p-2 rounded-lg bg-white/[0.02] hover:bg-white/[0.04] transition-colors">
          <div className="text-lg sm:text-xl font-semibold text-violet-400">79K</div>
          <div className="text-[10px] text-zinc-500 uppercase tracking-wide mt-0.5">{t('params')}</div>
        </div>
      </div>
    </div>
  );
}
