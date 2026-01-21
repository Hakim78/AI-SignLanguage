interface TechnicalSpecsProps {
  language: 'en' | 'fr';
}

interface Metric {
  label: string;
  value: string;
  unit?: string;
}

interface SpecSection {
  title: string;
  items: { label: string; value: string }[];
}

export function TechnicalSpecs({ language }: TechnicalSpecsProps) {
  const metrics: Metric[] = [
    { label: 'Accuracy', value: '94.2', unit: '%' },
    { label: 'F1-Score', value: '0.93' },
    { label: 'Latency', value: '<15', unit: 'ms' },
    { label: 'Parameters', value: '79', unit: 'K' },
  ];

  const specs: SpecSection[] = [
    {
      title: language === 'en' ? 'Architecture' : 'Architecture',
      items: [
        { label: 'Model', value: 'S-TRM v1.2' },
        { label: 'Layers', value: '6 Recursive' },
        { label: 'Hidden dim', value: '128' },
        { label: 'Activation', value: 'SiLU' },
      ]
    },
    {
      title: 'Preprocessing',
      items: [
        { label: 'Normalization', value: 'Z-score' },
        { label: 'Features', value: '63 coords' },
        { label: 'Sequence', value: '30 frames' },
        { label: 'Augmentation', value: 'Spatial' },
      ]
    },
    {
      title: 'Training',
      items: [
        { label: 'Epochs', value: '150' },
        { label: 'Batch size', value: '64' },
        { label: 'Scheduler', value: 'Cosine' },
        { label: 'Loss', value: 'CE + Deep' },
      ]
    }
  ];

  return (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {metrics.map((metric, i) => (
          <div key={i} className="metric-card">
            <div className="metric-label">{metric.label}</div>
            <div className="metric-value">
              {metric.value}
              {metric.unit && <span className="metric-unit">{metric.unit}</span>}
            </div>
          </div>
        ))}
      </div>

      {/* Detailed Specs */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {specs.map((section, i) => (
          <div key={i} className="card p-4">
            <h4 className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-3">
              {section.title}
            </h4>
            <div className="space-y-2">
              {section.items.map((item, j) => (
                <div key={j} className="flex justify-between items-center text-sm">
                  <span className="text-zinc-500">{item.label}</span>
                  <span className="font-mono text-zinc-300">{item.value}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Technical Description */}
      <div className="card p-5">
        <h4 className="text-sm font-medium text-zinc-300 mb-3">
          {language === 'en' ? 'Model Architecture' : 'Architecture du Modele'}
        </h4>
        <p className="text-sm text-zinc-500 leading-relaxed">
          {language === 'en'
            ? 'S-TRM (Stateful Tiny Recursive Model) optimized for real-time inference with low latency. Temporal drift reduction via recursive architecture with state memory. Latent recursion (n=6) enables complex temporal feature extraction without the overhead of traditional LSTM/Transformer architectures.'
            : 'S-TRM (Stateful Tiny Recursive Model) optimise pour l\'inference temps-reel a faible latence. Reduction de la derive temporelle via une architecture recursive a memoire d\'etat. La recursion latente (n=6) permet l\'extraction de features temporelles complexes sans la lourdeur des architectures LSTM/Transformer classiques.'}
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          <span className="code-inline">torch.jit.script</span>
          <span className="code-inline">ONNX Runtime</span>
          <span className="code-inline">WebGL Backend</span>
        </div>
      </div>
    </div>
  );
}
