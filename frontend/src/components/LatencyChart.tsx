interface LatencyChartProps {
  language: 'en' | 'fr';
}

const LATENCY_DATA = [
  { label: 'PyTorch CPU', value: 45, color: '#71717a' },
  { label: 'ONNX CPU', value: 28, color: '#a1a1aa' },
  { label: 'ONNX WebGL', value: 12, color: '#6366f1' },
];

const MAX_VALUE = 50;

export function LatencyChart({ language }: LatencyChartProps) {
  return (
    <div className="card p-5">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-sm font-medium text-zinc-300">
          {language === 'en' ? 'Inference Latency' : 'Latence d\'Inference'}
        </h4>
        <span className="text-xs text-zinc-600">ms/frame</span>
      </div>

      <div className="space-y-3">
        {LATENCY_DATA.map((item, i) => (
          <div key={i} className="space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-zinc-500">{item.label}</span>
              <span className="font-mono text-zinc-400">{item.value}ms</span>
            </div>
            <div className="h-2 bg-zinc-900 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${(item.value / MAX_VALUE) * 100}%`,
                  backgroundColor: item.color,
                }}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4 pt-4 border-t border-white/5">
        <div className="flex items-center gap-2">
          <svg className="w-3.5 h-3.5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
          </svg>
          <span className="text-xs text-zinc-500">
            {language === 'en'
              ? '3.75x faster with WebGL backend'
              : '3.75x plus rapide avec backend WebGL'}
          </span>
        </div>
      </div>
    </div>
  );
}
