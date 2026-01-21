import { useState, useRef, useCallback, useEffect } from 'react';
import confusionMatrixImg from '../assets/confusion_matrix_pro.png';

interface ConfusionMatrixPreviewProps {
  language: 'en' | 'fr';
}

// Stats data with translations
const STATS = [
  {
    labelEn: 'Accuracy',
    labelFr: 'Précision',
    value: '94.2%',
    color: 'text-white',
  },
  {
    labelEn: 'F1-Score',
    labelFr: 'Score F1',
    value: '0.93',
    color: 'text-emerald-400',
  },
  {
    labelEn: 'Top Confusion',
    labelFr: 'Confusion Max',
    value: 'M ↔ N',
    color: 'text-amber-400',
    tooltip: {
      en: 'Similar hand shapes cause occasional misclassification',
      fr: 'Les formes similaires causent des erreurs de classification',
    },
  },
];

export function ConfusionMatrixPreview({ language }: ConfusionMatrixPreviewProps) {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [zoomState, setZoomState] = useState({ active: false, x: 50, y: 50 });
  const [spotlightPos, setSpotlightPos] = useState({ x: 0, y: 0 });
  const [spotlightOpacity, setSpotlightOpacity] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const cardRef = useRef<HTMLDivElement>(null);

  // Magnifier effect handler
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!containerRef.current) return;
    const { left, top, width, height } = containerRef.current.getBoundingClientRect();
    const x = ((e.clientX - left) / width) * 100;
    const y = ((e.clientY - top) / height) * 100;
    setZoomState({ active: true, x, y });
  }, []);

  // Spotlight card effect handler
  const handleCardMouseMove = useCallback((e: React.MouseEvent) => {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    setSpotlightPos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
  }, []);

  // Keyboard handler for modal open
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      setIsModalOpen(true);
    }
  }, []);

  // ESC key handler for modal close
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isModalOpen) {
        setIsModalOpen(false);
      }
    };
    window.addEventListener('keydown', handleEsc);
    return () => window.removeEventListener('keydown', handleEsc);
  }, [isModalOpen]);

  return (
    <>
      {/* Main Card */}
      <div
        ref={cardRef}
        onMouseMove={handleCardMouseMove}
        onMouseEnter={() => setSpotlightOpacity(1)}
        onMouseLeave={() => setSpotlightOpacity(0)}
        className="relative h-full flex flex-col overflow-hidden rounded-2xl border border-white/5 bg-zinc-900/60 shadow-xl"
      >
        {/* Spotlight gradient effect */}
        <div
          className="pointer-events-none absolute -inset-px rounded-2xl transition-opacity duration-300"
          style={{
            opacity: spotlightOpacity,
            background: `radial-gradient(600px circle at ${spotlightPos.x}px ${spotlightPos.y}px, rgba(139, 92, 246, 0.08), transparent 40%)`,
          }}
        />

        {/* Header */}
        <div className="relative flex items-center justify-between px-5 py-4 border-b border-white/5 bg-black/20">
          <div>
            <h3 className="font-bold text-white text-sm tracking-wide">
              {language === 'en' ? 'CONFUSION MATRIX' : 'MATRICE DE CONFUSION'}
            </h3>
            <p className="text-[10px] text-zinc-500 font-mono mt-0.5">
              {language === 'en' ? 'NORMALIZED • VALIDATION SET' : 'NORMALISÉE • SET VALIDATION'}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-[10px] font-mono text-zinc-400">31 {language === 'en' ? 'Classes' : 'Classes'}</span>
            </div>
            {/* Fullscreen button */}
            <button
              onClick={() => setIsModalOpen(true)}
              onKeyDown={handleKeyDown}
              className="p-1.5 rounded-lg hover:bg-white/5 text-zinc-500 hover:text-white transition-colors"
              aria-label={language === 'en' ? 'View fullscreen' : 'Voir en plein écran'}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
              </svg>
            </button>
          </div>
        </div>

        {/* Image Container with Magnifier */}
        <div
          ref={containerRef}
          className="relative flex-1 bg-black/40 cursor-crosshair overflow-hidden"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setZoomState(prev => ({ ...prev, active: false }))}
          onClick={() => setIsModalOpen(true)}
          role="button"
          tabIndex={0}
          onKeyDown={handleKeyDown}
          aria-label={language === 'en' ? 'Click to view fullscreen' : 'Cliquez pour voir en plein écran'}
        >
          {/* Base Image */}
          <div className="absolute inset-0 flex items-center justify-center p-4">
            <img
              src={confusionMatrixImg}
              alt={language === 'en' ? 'S-TRM Confusion Matrix showing classification results' : 'Matrice de confusion S-TRM montrant les résultats de classification'}
              className="w-full h-full object-contain transition-all duration-300 opacity-90 hover:opacity-100"
              loading="lazy"
            />
          </div>

          {/* Magnifier Lens */}
          <div
            className="absolute w-28 h-28 rounded-full border-2 border-violet-500/40 shadow-[0_0_20px_rgba(139,92,246,0.3)] pointer-events-none transition-opacity duration-150 bg-no-repeat"
            style={{
              opacity: zoomState.active ? 1 : 0,
              left: `calc(${zoomState.x}% - 56px)`,
              top: `calc(${zoomState.y}% - 56px)`,
              backgroundImage: `url(${confusionMatrixImg})`,
              backgroundSize: '400%',
              backgroundPosition: `${zoomState.x}% ${zoomState.y}%`,
              backgroundColor: '#0a0a0a',
            }}
          />

          {/* Hover instruction */}
          <div
            className={`absolute bottom-3 right-3 flex items-center gap-1.5 bg-black/70 backdrop-blur-sm px-2.5 py-1.5 rounded-lg text-[10px] text-zinc-400 transition-opacity duration-300 ${
              zoomState.active ? 'opacity-0' : 'opacity-100'
            }`}
          >
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
            </svg>
            <span>{language === 'en' ? 'Hover to zoom • Click to expand' : 'Survoler pour zoomer • Cliquer pour agrandir'}</span>
          </div>
        </div>

        {/* Footer Stats */}
        <div className="relative grid grid-cols-3 divide-x divide-white/5 border-t border-white/5 bg-black/20">
          {STATS.map((stat, i) => (
            <div key={i} className="group relative p-3 flex flex-col items-center hover:bg-white/[0.02] transition-colors">
              <span className={`text-[10px] uppercase tracking-wider mb-0.5 ${stat.tooltip ? 'text-amber-500/80' : 'text-zinc-500'}`}>
                {language === 'en' ? stat.labelEn : stat.labelFr}
              </span>
              <span className={`text-sm font-mono font-bold ${stat.color}`}>
                {stat.value}
              </span>
              {/* Tooltip */}
              {stat.tooltip && (
                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-max max-w-[180px] px-2.5 py-1.5 bg-zinc-900 border border-white/10 rounded-lg text-[10px] text-zinc-300 text-center opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none shadow-xl z-10">
                  {language === 'en' ? stat.tooltip.en : stat.tooltip.fr}
                  <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-px border-4 border-transparent border-t-zinc-900" />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Fullscreen Modal */}
      {isModalOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-sm p-4 sm:p-8 animate-fade-in"
          onClick={() => setIsModalOpen(false)}
          role="dialog"
          aria-modal="true"
          aria-label={language === 'en' ? 'Confusion Matrix fullscreen view' : 'Matrice de confusion en plein écran'}
        >
          {/* Close button */}
          <button
            onClick={() => setIsModalOpen(false)}
            className="absolute top-4 right-4 p-2 rounded-full bg-white/10 hover:bg-white/20 text-white transition-colors"
            aria-label={language === 'en' ? 'Close' : 'Fermer'}
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>

          {/* Modal content */}
          <div
            className="relative max-w-5xl max-h-[90vh] w-full"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-xl font-bold text-white">
                  {language === 'en' ? 'Confusion Matrix' : 'Matrice de Confusion'}
                </h2>
                <p className="text-sm text-zinc-400 mt-1">
                  {language === 'en'
                    ? 'Normalized classification results across 31 ASL sign classes'
                    : 'Résultats de classification normalisés sur 31 classes de signes ASL'}
                </p>
              </div>
            </div>

            {/* Image */}
            <div className="rounded-xl overflow-hidden border border-white/10 bg-black/50">
              <img
                src={confusionMatrixImg}
                alt={language === 'en' ? 'S-TRM Confusion Matrix' : 'Matrice de Confusion S-TRM'}
                className="w-full h-auto"
              />
            </div>

            {/* Modal Footer Stats */}
            <div className="flex items-center justify-center gap-8 mt-4 p-3 bg-white/5 rounded-xl">
              {STATS.map((stat, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="text-xs text-zinc-400">
                    {language === 'en' ? stat.labelEn : stat.labelFr}:
                  </span>
                  <span className={`text-sm font-mono font-bold ${stat.color}`}>
                    {stat.value}
                  </span>
                </div>
              ))}
            </div>

            {/* Hint */}
            <p className="text-center text-xs text-zinc-500 mt-3">
              {language === 'en' ? 'Press ESC or click outside to close' : 'Appuyez sur ESC ou cliquez à l\'extérieur pour fermer'}
            </p>
          </div>
        </div>
      )}
    </>
  );
}
