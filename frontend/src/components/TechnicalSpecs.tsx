import { useState } from 'react';

// Import des assets
import reliabilityDiagram from '../assets/reliability_diagram.png';
import confidenceCalibration from '../assets/confidence_calibration.png';

interface TechnicalSpecsProps {
  language: 'en' | 'fr';
}

interface Metric {
  label: string;
  labelFr: string;
  value: string;
  unit?: string;
  trend: 'up' | 'down';
  description: string;
  descriptionFr: string;
}

// Traductions pour les sections
const TRANSLATIONS = {
  en: {
    reliabilityTab: 'Reliability Diagram',
    calibrationTab: 'Confidence Distribution',
    calibrationTitle: 'Calibration Analysis',
    calibrationDesc: "The 'S-shaped' curve indicates typical deep learning overconfidence. We apply Temperature Scaling to linearize this response.",
    thresholdTitle: 'Decision Threshold',
    thresholdDesc: "Optimal separation occurs at 0.75 probability. Predictions below this threshold are flagged as 'Uncertain' to the user.",
    optimizationActive: 'Optimization Active',
    eceScore: 'ECE Score',
    mceScore: 'MCE Score',
    threshold: 'Threshold',
    rejectionRate: 'Rejection Rate',
    architecture: 'Architecture',
    training: 'Training',
    deployment: 'Deployment',
  },
  fr: {
    reliabilityTab: 'Diagramme de Fiabilité',
    calibrationTab: 'Distribution de Confiance',
    calibrationTitle: 'Analyse de Calibration',
    calibrationDesc: "La courbe en 'S' indique une sur-confiance typique du Deep Learning. Nous appliquons un Temperature Scaling pour linéariser cette réponse.",
    thresholdTitle: 'Seuil de Décision',
    thresholdDesc: "La séparation optimale se fait à 0.75 de probabilité. Les prédictions sous ce seuil sont marquées comme 'Incertaines'.",
    optimizationActive: 'Optimisation Active',
    eceScore: 'Score ECE',
    mceScore: 'Score MCE',
    threshold: 'Seuil',
    rejectionRate: 'Taux de Rejet',
    architecture: 'Architecture',
    training: 'Entraînement',
    deployment: 'Déploiement',
  },
};

// Composant Indicateur de Tendance (Plus discret)
function TrendIndicator({ trend }: { trend: 'up' | 'down' }) {
  const isGood = trend === 'up' || trend === 'down'; // Dans notre cas, tout est bon :)
  return (
    <div className={`flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium ${isGood ? 'bg-emerald-500/10 text-emerald-400' : 'bg-amber-500/10 text-amber-400'}`}>
      {trend === 'up' ? '↗' : '↘'}
      <span>{trend === 'up' ? '+2.4%' : '-15%'}</span>
    </div>
  );
}

export function TechnicalSpecs({ language }: TechnicalSpecsProps) {
  const [activeTab, setActiveTab] = useState<'reliability' | 'calibration'>('reliability');
  const t = TRANSLATIONS[language];

  const metrics: Metric[] = [
    { label: 'Accuracy', labelFr: 'Précision', value: '94.2', unit: '%', trend: 'up', description: 'Test set performance', descriptionFr: 'Performance test set' },
    { label: 'F1-Score', labelFr: 'Score F1', value: '0.93', trend: 'up', description: 'Macro-averaged', descriptionFr: 'Moyenne macro' },
    { label: 'Latency', labelFr: 'Latence', value: '<15', unit: 'ms', trend: 'down', description: 'P95 inference', descriptionFr: 'Inférence P95' },
    { label: 'Model Size', labelFr: 'Taille', value: '79', unit: 'K', trend: 'down', description: 'Parameters', descriptionFr: 'Paramètres' },
  ];

  return (
    <div className="space-y-8">

      {/* 1. HIGH-LEVEL METRICS (Style Dashboard) */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {metrics.map((metric, i) => (
          <div
            key={i}
            className="group relative p-5 bg-zinc-900/40 border border-white/5 rounded-xl hover:border-white/10 transition-all duration-300"
          >
            <div className="flex justify-between items-start mb-4">
              <span className="text-zinc-500 text-xs font-semibold uppercase tracking-wider">
                {language === 'en' ? metric.label : metric.labelFr}
              </span>
              <TrendIndicator trend={metric.trend} />
            </div>
            
            <div className="flex items-baseline gap-1">
              <span className="text-3xl font-mono text-white font-bold tracking-tighter group-hover:text-emerald-400 transition-colors">
                {metric.value}
              </span>
              {metric.unit && <span className="text-sm text-zinc-500 font-mono">{metric.unit}</span>}
            </div>
            
            <div className="mt-3 text-[10px] text-zinc-600 font-medium">
              {language === 'en' ? metric.description : metric.descriptionFr}
            </div>
            
            {/* Petit indicateur visuel en bas */}
            <div className="absolute bottom-0 left-0 w-full h-0.5 bg-gradient-to-r from-transparent via-white/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
          </div>
        ))}
      </div>

      {/* 2. DEEP ANALYSIS (Tabbed Interface) */}
      <div className="bg-black/20 border border-white/5 rounded-2xl overflow-hidden backdrop-blur-sm">

        {/* Header des Onglets (Style Pill) */}
        <div className="flex items-center p-2 bg-zinc-900/50 border-b border-white/5">
          <button
            onClick={() => setActiveTab('reliability')}
            className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-medium transition-all duration-300 ${
              activeTab === 'reliability'
                ? 'bg-zinc-800 text-white shadow-lg'
                : 'text-zinc-500 hover:text-zinc-300 hover:bg-white/5'
            }`}
          >
            <span className={`w-2 h-2 rounded-full ${activeTab === 'reliability' ? 'bg-emerald-500' : 'bg-zinc-600'}`} />
            {t.reliabilityTab}
          </button>

          <button
            onClick={() => setActiveTab('calibration')}
            className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-medium transition-all duration-300 ${
              activeTab === 'calibration'
                ? 'bg-zinc-800 text-white shadow-lg'
                : 'text-zinc-500 hover:text-zinc-300 hover:bg-white/5'
            }`}
          >
            <span className={`w-2 h-2 rounded-full ${activeTab === 'calibration' ? 'bg-blue-500' : 'bg-zinc-600'}`} />
            {t.calibrationTab}
          </button>
        </div>

        {/* Contenu des Onglets */}
        <div className="p-6 sm:p-8">
          
          {/* TAB 1 */}
          {activeTab === 'reliability' && (
            <div className="grid md:grid-cols-2 gap-8 items-center">
              {/* Graphique avec effet "Glass" */}
              <div className="relative group rounded-xl overflow-hidden border border-white/5 bg-black/40">
                <div className="absolute inset-0 bg-grid-white/[0.05] pointer-events-none" />
                <img
                  src={reliabilityDiagram}
                  alt="Reliability Diagram"
                  className="w-full h-full object-contain p-4 opacity-90 transition-transform duration-500 group-hover:scale-105"
                />
                <div className="absolute bottom-2 right-2 px-2 py-1 bg-black/60 rounded text-[9px] text-zinc-400 font-mono backdrop-blur-md">
                  ECE: 0.042
                </div>
              </div>

              {/* Explications */}
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-bold text-white mb-2">
                    {t.calibrationTitle}
                  </h3>
                  <p className="text-sm text-zinc-400 leading-relaxed">
                    {t.calibrationDesc}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <StatBox label={t.eceScore} value="0.042" color="text-emerald-400" />
                  <StatBox label={t.mceScore} value="0.128" color="text-blue-400" />
                </div>

                <div className="p-3 rounded-lg border border-emerald-500/20 bg-emerald-500/5">
                  <div className="flex items-center gap-2 mb-1">
                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                    <span className="text-xs font-bold text-emerald-400 uppercase tracking-wide">{t.optimizationActive}</span>
                  </div>
                  <p className="text-xs text-zinc-400">
                    Temperature Scaling (T=1.5)
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2 */}
          {activeTab === 'calibration' && (
            <div className="grid md:grid-cols-2 gap-8 items-center">
              <div className="relative group rounded-xl overflow-hidden border border-white/5 bg-black/40">
                <div className="absolute inset-0 bg-grid-white/[0.05] pointer-events-none" />
                <img
                  src={confidenceCalibration}
                  alt="Confidence Histogram"
                  className="w-full h-full object-contain p-4 opacity-90 transition-transform duration-500 group-hover:scale-105"
                />
              </div>

              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-bold text-white mb-2">
                    {t.thresholdTitle}
                  </h3>
                  <p className="text-sm text-zinc-400 leading-relaxed">
                    {t.thresholdDesc}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <StatBox label={t.threshold} value="> 0.75" color="text-emerald-400" />
                  <StatBox label={t.rejectionRate} value="~12%" color="text-amber-400" />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 3. SPECS GRID (Clean & Minimal) */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-4">
        <SpecCard
          title={t.architecture}
          icon="cpu"
          items={[
            [language === 'en' ? 'Model' : 'Modèle', 'S-TRM v1.2'],
            [language === 'en' ? 'Layers' : 'Couches', '6 Recursive'],
            [language === 'en' ? 'Hidden' : 'Cachées', '128 dim'],
            ['Activation', 'SiLU'],
          ]}
        />
        <SpecCard
          title={t.training}
          icon="activity"
          items={[
            [language === 'en' ? 'Epochs' : 'Époques', '100'],
            ['Batch', '64'],
            [language === 'en' ? 'Optimizer' : 'Optimiseur', 'AdamW'],
            [language === 'en' ? 'Loss' : 'Perte', 'Focal+DS'],
          ]}
        />
        <SpecCard
          title={t.deployment}
          icon="box"
          items={[
            ['Format', 'ONNX'],
            [language === 'en' ? 'Precision' : 'Précision', 'FP16'],
            ['Runtime', 'WASM'],
            [language === 'en' ? 'Size' : 'Taille', '340 KB'],
          ]}
        />
      </div>
    </div>
  );
}

// --- HELPER COMPONENTS ---

function StatBox({ label, value, color }: { label: string, value: string, color: string }) {
  return (
    <div className="p-3 bg-zinc-800/40 rounded-lg border border-white/5">
      <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-1">{label}</div>
      <div className={`text-lg font-mono font-bold ${color}`}>{value}</div>
    </div>
  );
}

function SpecCard({ title, icon, items }: { title: string, icon: string, items: string[][] }) {
  return (
    <div className="p-5 bg-zinc-900/20 border border-white/5 rounded-xl hover:bg-zinc-900/40 transition-colors">
      <div className="flex items-center gap-2 mb-4 text-zinc-400">
        {/* Simple SVG Icons based on string ID */}
        {icon === 'cpu' && <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" /></svg>}
        {icon === 'activity' && <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>}
        {icon === 'box' && <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" /></svg>}
        
        <h4 className="text-xs font-bold uppercase tracking-wider">{title}</h4>
      </div>
      <ul className="space-y-2">
        {items.map(([label, value], i) => (
          <li key={i} className="flex justify-between items-center text-sm">
            <span className="text-zinc-600">{label}</span>
            <span className="text-zinc-300 font-mono text-xs bg-white/5 px-2 py-0.5 rounded border border-white/5">{value}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}