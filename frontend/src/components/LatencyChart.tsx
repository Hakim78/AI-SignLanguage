import { useState, useEffect } from 'react';

interface LatencyChartProps {
  language: 'en' | 'fr';
}

interface LatencyItem {
  label: string;
  labelFr: string;
  value: number;
  color: string;
  isActive?: boolean;
}

const LATENCY_DATA: LatencyItem[] = [
  { label: 'PyTorch CPU', labelFr: 'PyTorch CPU', value: 45, color: '#52525b' },
  { label: 'ONNX CPU', labelFr: 'ONNX CPU', value: 28, color: '#71717a' },
  { label: 'ONNX WASM', labelFr: 'ONNX WASM', value: 15, color: '#8b5cf6', isActive: true },
  { label: 'ONNX WebGL', labelFr: 'ONNX WebGL', value: 12, color: '#6366f1' },
];

const MAX_VALUE = 50;

export function LatencyChart({ language }: LatencyChartProps) {
  const [animated, setAnimated] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setAnimated(true), 100);
    return () => clearTimeout(timer);
  }, []);

  const activeItem = LATENCY_DATA.find(d => d.isActive);
  const speedup = activeItem ? (LATENCY_DATA[0].value / activeItem.value).toFixed(1) : '3.0';

  return (
    <div className="card p-5 h-full flex flex-col">
      <div className="flex items-center justify-between mb-5">
        <div>
          <h3 className="font-semibold text-zinc-200">
            {language === 'en' ? 'Inference Latency' : 'Latence d\'Inférence'}
          </h3>
          <p className="text-[11px] text-zinc-600 mt-0.5">
            {language === 'en' ? 'Lower is better' : 'Plus bas = mieux'}
          </p>
        </div>
        <span className="text-xs px-2 py-1 rounded-full bg-zinc-800/50 text-zinc-500 font-mono border border-white/5">
          ms/frame
        </span>
      </div>

      <div className="flex-1 space-y-4">
        {LATENCY_DATA.map((item, i) => (
          <div key={i} className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className={`text-xs ${item.isActive ? 'text-violet-400 font-medium' : 'text-zinc-500'}`}>
                  {language === 'en' ? item.label : item.labelFr}
                </span>
                {item.isActive && (
                  <span className="text-[9px] px-1.5 py-0.5 rounded bg-violet-500/20 text-violet-400 uppercase tracking-wider font-medium">
                    {language === 'en' ? 'Active' : 'Actif'}
                  </span>
                )}
              </div>
              <span className={`font-mono text-xs ${item.isActive ? 'text-violet-400' : 'text-zinc-400'}`}>
                {item.value}ms
              </span>
            </div>
            <div className="h-2.5 bg-zinc-900/80 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-700 ease-out"
                style={{
                  width: animated ? `${(item.value / MAX_VALUE) * 100}%` : '0%',
                  backgroundColor: item.color,
                  transitionDelay: `${i * 100}ms`,
                }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Performance Summary */}
      <div className="mt-5 pt-4 border-t border-white/5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-full bg-emerald-500/10 flex items-center justify-center">
              <svg className="w-3.5 h-3.5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <span className="text-xs text-zinc-400">
              {language === 'en'
                ? `${speedup}x faster than PyTorch`
                : `${speedup}x plus rapide que PyTorch`}
            </span>
          </div>
          <div className="text-right">
            <div className="text-lg font-mono font-bold text-emerald-400">{activeItem?.value}ms</div>
            <div className="text-[10px] text-zinc-600">
              {language === 'en' ? 'current backend' : 'backend actuel'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
