interface ConfusionMatrixPreviewProps {
  language: 'en' | 'fr';
}

// Simulated confusion matrix data (8x8 subset of the full 31-class matrix)
// High diagonal values indicate good classification accuracy
const MATRIX_DATA = [
  [0.96, 0.01, 0.00, 0.01, 0.00, 0.01, 0.00, 0.01],
  [0.02, 0.94, 0.01, 0.00, 0.01, 0.00, 0.01, 0.01],
  [0.00, 0.01, 0.95, 0.02, 0.00, 0.01, 0.00, 0.01],
  [0.01, 0.00, 0.02, 0.93, 0.01, 0.01, 0.01, 0.01],
  [0.00, 0.01, 0.00, 0.01, 0.97, 0.00, 0.00, 0.01],
  [0.01, 0.00, 0.01, 0.01, 0.00, 0.94, 0.02, 0.01],
  [0.00, 0.01, 0.00, 0.01, 0.00, 0.02, 0.95, 0.01],
  [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.93],
];

const LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];

export function ConfusionMatrixPreview({ language }: ConfusionMatrixPreviewProps) {
  const getOpacity = (value: number) => {
    // Scale opacity based on value, with diagonal (correct predictions) being brightest
    return Math.max(0.05, value);
  };

  const getCellColor = (value: number, isDiagonal: boolean) => {
    if (isDiagonal) {
      // Diagonal cells (correct predictions) use accent color
      return `rgba(99, 102, 241, ${getOpacity(value)})`;
    }
    // Off-diagonal cells (misclassifications) use neutral/red tint
    return value > 0.02
      ? `rgba(239, 68, 68, ${getOpacity(value * 2)})`
      : `rgba(255, 255, 255, ${getOpacity(value)})`;
  };

  return (
    <div className="card p-5">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-sm font-medium text-zinc-300">
          {language === 'en' ? 'Confusion Matrix' : 'Matrice de Confusion'}
        </h4>
        <span className="text-xs text-zinc-600">8x8 {language === 'en' ? 'preview' : 'apercu'}</span>
      </div>

      <div className="relative">
        {/* Y-axis label */}
        <div className="absolute -left-6 top-1/2 -translate-y-1/2 -rotate-90 text-[10px] text-zinc-600 font-medium tracking-wider">
          {language === 'en' ? 'TRUE' : 'VRAI'}
        </div>

        {/* Matrix grid */}
        <div className="ml-4">
          {/* X-axis labels */}
          <div className="flex mb-1 pl-5">
            {LABELS.map((label, i) => (
              <div
                key={i}
                className="w-7 h-4 flex items-center justify-center text-[10px] font-mono text-zinc-600"
              >
                {label}
              </div>
            ))}
          </div>

          {/* Matrix rows */}
          {MATRIX_DATA.map((row, i) => (
            <div key={i} className="flex items-center">
              {/* Y-axis label */}
              <div className="w-5 h-7 flex items-center justify-center text-[10px] font-mono text-zinc-600">
                {LABELS[i]}
              </div>
              {/* Cells */}
              {row.map((value, j) => (
                <div
                  key={j}
                  className="w-7 h-7 m-[1px] rounded-sm flex items-center justify-center transition-all duration-200 hover:scale-110 cursor-default"
                  style={{
                    backgroundColor: getCellColor(value, i === j),
                  }}
                  title={`${LABELS[i]}→${LABELS[j]}: ${(value * 100).toFixed(1)}%`}
                >
                  {i === j && value > 0.9 && (
                    <span className="text-[8px] font-mono text-white/70">
                      {Math.round(value * 100)}
                    </span>
                  )}
                </div>
              ))}
            </div>
          ))}

          {/* X-axis label */}
          <div className="text-center text-[10px] text-zinc-600 font-medium tracking-wider mt-2">
            {language === 'en' ? 'PREDICTED' : 'PREDIT'}
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 mt-4 pt-4 border-t border-white/5">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-sm bg-indigo-500/80" />
          <span className="text-[10px] text-zinc-500">
            {language === 'en' ? 'Correct' : 'Correct'}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-sm bg-red-500/30" />
          <span className="text-[10px] text-zinc-500">
            {language === 'en' ? 'Misclassified' : 'Erreur'}
          </span>
        </div>
      </div>
    </div>
  );
}
