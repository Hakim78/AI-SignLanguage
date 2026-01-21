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
  t?: (key: string) => string;
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
}: PredictionPanelProps) {
  const getConfidenceColor = (conf: number) => {
    if (conf >= 90) return '#22c55e';
    if (conf >= 70) return '#6366f1';
    if (conf >= 50) return '#eab308';
    return '#71717a';
  };

  return (
    <div className="console h-full flex flex-col">
      {/* Console Header */}
      <div className="console-header">
        <div className="flex gap-1.5">
          <div className={`console-dot ${
            status === 'running' ? 'bg-green-500' :
            status === 'ready' ? 'bg-yellow-500' :
            status === 'loading' ? 'bg-yellow-500 animate-pulse' : 'bg-red-500'
          }`} />
          <div className="console-dot bg-zinc-700" />
          <div className="console-dot bg-zinc-700" />
        </div>
        <span className="text-xs text-zinc-500 ml-2">inference.monitor</span>
        <div className="ml-auto flex items-center gap-2">
          <span className={`status-dot ${
            status === 'running' ? 'status-running animate-pulse' :
            status === 'ready' ? 'status-ready' :
            status === 'loading' ? 'status-loading' : 'status-error'
          }`} />
          <span className="text-xs text-zinc-500">{statusText}</span>
        </div>
      </div>

      {/* Loading Progress */}
      {status === 'loading' && (
        <div className="px-4 py-2 border-b border-white/5">
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${loadingProgress}%` }} />
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 p-4 space-y-4 overflow-auto">
        {/* Primary Output */}
        <div className="text-center py-6">
          <div className="text-[10px] font-mono text-zinc-600 uppercase tracking-wider mb-2">
            output.prediction
          </div>
          <div
            className="prediction-display font-bold text-6xl sm:text-7xl tracking-tight"
            style={{
              color: prediction === '-' ? '#27272a' : getConfidenceColor(confidence),
              textShadow: confidence >= 90 ? `0 0 40px ${getConfidenceColor(confidence)}40` : 'none'
            }}
          >
            {prediction}
          </div>
          {confidence > 0 && (
            <div className="mt-3 inline-flex items-center gap-2 px-3 py-1.5 rounded-md bg-zinc-900/50 border border-white/5">
              <div
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: getConfidenceColor(confidence) }}
              />
              <span className="text-xs font-mono text-zinc-400">
                conf: <span style={{ color: getConfidenceColor(confidence) }}>{confidence}%</span>
              </span>
            </div>
          )}
        </div>

        {/* Probability Distribution */}
        <div>
          <div className="text-[10px] font-mono text-zinc-600 uppercase tracking-wider mb-3">
            probability.distribution
          </div>
          <div className="space-y-2">
            {topPredictions.length > 0 ? (
              topPredictions.slice(0, 5).map((pred, i) => (
                <div key={i} className="flex items-center gap-3">
                  <span className="w-6 text-xs font-mono text-zinc-500">{pred.label}</span>
                  <div className="flex-1 confidence-bar">
                    <div
                      className="confidence-fill"
                      style={{
                        width: `${pred.probability}%`,
                        backgroundColor: i === 0 ? '#6366f1' : '#3f3f46'
                      }}
                    />
                  </div>
                  <span className="w-10 text-right text-[11px] font-mono text-zinc-500">
                    {pred.probability.toFixed(1)}
                  </span>
                </div>
              ))
            ) : (
              <div className="text-xs font-mono text-zinc-700 text-center py-4">
                awaiting_input...
              </div>
            )}
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="grid grid-cols-3 gap-2 pt-4 border-t border-white/5">
          <div className="text-center">
            <div className="text-[10px] font-mono text-zinc-600 mb-1">fps</div>
            <div className={`text-lg font-semibold font-mono ${
              fps >= 25 ? 'text-green-400' : fps >= 15 ? 'text-yellow-400' : 'text-zinc-500'
            }`}>
              {fps || '--'}
            </div>
          </div>
          <div className="text-center">
            <div className="text-[10px] font-mono text-zinc-600 mb-1">latency</div>
            <div className={`text-lg font-semibold font-mono ${
              latency <= 20 ? 'text-green-400' : latency <= 50 ? 'text-yellow-400' : 'text-zinc-500'
            }`}>
              {latency || '--'}
              <span className="text-xs text-zinc-600">ms</span>
            </div>
          </div>
          <div className="text-center">
            <div className="text-[10px] font-mono text-zinc-600 mb-1">params</div>
            <div className="text-lg font-semibold font-mono text-indigo-400">
              79<span className="text-xs text-zinc-600">K</span>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-white/5 bg-zinc-950/50">
        <div className="flex items-center justify-between text-[10px] font-mono text-zinc-600">
          <span>S-TRM.v1.2</span>
          <span>ONNX.WebGL</span>
        </div>
      </div>
    </div>
  );
}
