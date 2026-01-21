import { useState } from 'react';

// Import des assets générés par le script Python
import reliabilityDiagram from '../assets/reliability_diagram.png';
import confidenceCalibration from '../assets/confidence_calibration.png';

interface TechnicalSpecsProps {
  language: 'en' | 'fr';
}

interface Metric {
  label: string;
  value: string;
  unit?: string;
  trend?: 'up' | 'down' | 'neutral';
}

export function TechnicalSpecs({ language }: TechnicalSpecsProps) {
  // On gère les onglets pour basculer entre les deux analyses techniques
  const [activeTab, setActiveTab] = useState<'reliability' | 'calibration'>('reliability');

  const metrics: Metric[] = [
    { label: 'Accuracy', value: '94.2', unit: '%', trend: 'up' },
    { label: 'F1-Score', value: '0.93', trend: 'up' },
    { label: 'Latency', value: '<15', unit: 'ms', trend: 'down' },
    { label: 'Parameters', value: '79', unit: 'K', trend: 'down' },
  ];

  return (
    <div className="space-y-8">
      
      {/* 1. High-Level Metrics (Top Bar) */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {metrics.map((metric, i) => (
          <div key={i} className="metric-card p-4 bg-zinc-900/50 border border-white/5 rounded-xl hover:border-violet-500/20 transition-colors">
            <div className="text-zinc-500 text-xs font-medium uppercase tracking-wider mb-1">{metric.label}</div>
            <div className="flex items-baseline gap-1">
              <span className="text-2xl font-mono text-white font-semibold">{metric.value}</span>
              {metric.unit && <span className="text-sm text-zinc-400 font-mono">{metric.unit}</span>}
            </div>
          </div>
        ))}
      </div>

      {/* 2. Deep Analysis Section (The Scientific Core) */}
      <div className="card bg-zinc-900/30 border border-white/10 rounded-2xl overflow-hidden">
        
        {/* Tabs Header */}
        <div className="flex border-b border-white/5">
          <button
            onClick={() => setActiveTab('reliability')}
            className={`flex-1 py-4 text-sm font-medium transition-colors ${
              activeTab === 'reliability' ? 'text-violet-400 bg-white/5' : 'text-zinc-500 hover:text-zinc-300'
            }`}
          >
            {language === 'en' ? 'Reliability Diagram' : 'Diagramme de Fiabilité'}
          </button>
          <div className="w-px bg-white/5" />
          <button
            onClick={() => setActiveTab('calibration')}
            className={`flex-1 py-4 text-sm font-medium transition-colors ${
              activeTab === 'calibration' ? 'text-violet-400 bg-white/5' : 'text-zinc-500 hover:text-zinc-300'
            }`}
          >
            {language === 'en' ? 'Confidence Dist.' : 'Dist. de Confiance'}
          </button>
        </div>

        {/* Tab Content */}
        <div className="p-6">
          
          {/* TAB 1: RELIABILITY DIAGRAM */}
          {activeTab === 'reliability' && (
            <div className="grid md:grid-cols-2 gap-8 items-center">
              {/* Image Container */}
              <div className="space-y-4">
                <div className="aspect-square bg-black/40 rounded-lg border border-white/5 overflow-hidden p-2 flex items-center justify-center">
                  <img 
                    src={reliabilityDiagram} 
                    alt="Reliability Diagram" 
                    className="max-w-full max-h-full object-contain opacity-90 hover:opacity-100 transition-opacity"
                  />
                </div>
                <div className="text-center">
                  <span className="text-xs text-zinc-500 font-mono">Fig 1. Reliability Curve (Observed vs Perfect)</span>
                </div>
              </div>

              {/* Insight Text */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-white">
                  {language === 'en' ? 'Calibration Analysis' : 'Analyse de Calibration'}
                </h3>
                <p className="text-sm text-zinc-400 leading-relaxed">
                  {language === 'en' 
                    ? "The reliability diagram reveals a typical 'S-shaped' deviation characteristic of Deep Learning models. The model tends to be under-confident on hard samples and over-confident on easy ones."
                    : "Le diagramme de fiabilité révèle une déviation caractéristique des modèles de Deep Learning. La courbe s'éloigne de la diagonale idéale, indiquant une sur-confiance sur les prédictions erronées."}
                </p>
                
                <div className="p-4 bg-violet-500/10 border border-violet-500/20 rounded-lg">
                  <h4 className="text-violet-400 text-xs font-bold uppercase mb-2">
                    {language === 'en' ? 'Engineering Action' : 'Action Ingénierie'}
                  </h4>
                  <p className="text-xs text-zinc-300">
                    {language === 'en'
                      ? "Implementation of Temperature Scaling (T=1.5) recommended to align softmax probabilities with empirical accuracy."
                      : "Implémentation recommandée du 'Temperature Scaling' (T=1.5) pour aligner les probabilités softmax avec la précision empirique."}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: CONFIDENCE CALIBRATION */}
          {activeTab === 'calibration' && (
            <div className="grid md:grid-cols-2 gap-8 items-center">
              {/* Image Container */}
              <div className="space-y-4">
                <div className="aspect-square bg-black/40 rounded-lg border border-white/5 overflow-hidden p-2 flex items-center justify-center">
                  <img 
                    src={confidenceCalibration} 
                    alt="Confidence Calibration Histogram" 
                    className="max-w-full max-h-full object-contain opacity-90 hover:opacity-100 transition-opacity"
                  />
                </div>
                <div className="text-center">
                  <span className="text-xs text-zinc-500 font-mono">Fig 2. Confidence Density (Correct vs Incorrect)</span>
                </div>
              </div>

              {/* Insight Text */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-white">
                  {language === 'en' ? 'Thresholding Strategy' : 'Stratégie de Seuil'}
                </h3>
                <p className="text-sm text-zinc-400 leading-relaxed">
                  {language === 'en'
                    ? "The histogram shows a clear separation capability, but with overlap in the 0.4-0.6 range. Incorrect predictions (red) still maintain relatively high confidence scores."
                    : "L'histogramme montre une capacité de séparation, mais avec un chevauchement dans la zone 0.4-0.6. Les prédictions incorrectes (rouge) maintiennent des scores de confiance relativement élevés."}
                </p>

                <div className="grid grid-cols-2 gap-3 mt-4">
                  <div className="p-3 bg-zinc-800/50 rounded border border-white/5">
                    <div className="text-xs text-zinc-500 mb-1">Optimal Threshold</div>
                    <div className="text-green-400 font-mono text-sm"> 0.75</div>
                  </div>
                  <div className="p-3 bg-zinc-800/50 rounded border border-white/5">
                    <div className="text-xs text-zinc-500 mb-1">Rejection Rate</div>
                    <div className="text-yellow-400 font-mono text-sm">~12%</div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 3. Specs Grid (Tables) */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-4 border-t border-white/5">
        <SpecList 
          title="Architecture" 
          items={[
            ['Model', 'S-TRM v1.2'],
            ['Layers', '6 Recursive'],
            ['Hidden dim', '128'],
            ['Activation', 'SiLU'],
          ]} 
        />
        <SpecList 
          title="Training Config" 
          items={[
            ['Epochs', '100'],
            ['Batch size', '64'],
            ['Optimizer', 'AdamW'],
            ['Loss', 'Focal + DeepSup'],
          ]} 
        />
        <SpecList 
          title="Deployment" 
          items={[
            ['Format', 'ONNX (Web)'],
            ['Quantization', 'FP16'],
            ['Runtime', 'WASM / WebGL'],
            ['Size', '340 KB'],
          ]} 
        />
      </div>
    </div>
  );
}

// Helper pour les listes de specs
function SpecList({ title, items }: { title: string, items: string[][] }) {
  return (
    <div>
      <h4 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-4">{title}</h4>
      <ul className="space-y-3">
        {items.map(([label, value], i) => (
          <li key={i} className="flex justify-between text-sm">
            <span className="text-zinc-600">{label}</span>
            <span className="text-zinc-300 font-mono">{value}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}