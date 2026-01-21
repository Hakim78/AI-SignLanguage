import { useState, useRef, useCallback } from 'react';
import { Header } from './components/Header';
import { VideoFeed } from './components/VideoFeed';
import { PredictionPanel } from './components/PredictionPanel';
import { TechnicalSpecs } from './components/TechnicalSpecs';
import { ConfusionMatrixPreview } from './components/ConfusionMatrixPreview';
import { LatencyChart } from './components/LatencyChart';
import { translations } from './i18n/translations';
import { useSignRecognition } from './hooks/useSignRecognition';
import { useSectionAnalytics } from './lib/analytics/useSectionAnalytics';

type Language = 'en' | 'fr';

interface VideoFeedRef {
  video: HTMLVideoElement | null;
  canvas: HTMLCanvasElement | null;
}

function App() {
  const [lang, setLang] = useState<Language>('en');
  const videoFeedRef = useRef<VideoFeedRef>(null);

  // Initialize section analytics tracking
  useSectionAnalytics();

  const {
    isRunning,
    isModelReady,
    status,
    loadingProgress,
    prediction,
    confidence,
    topPredictions,
    fps,
    latency,
    handDetected,
    currentMode,
    setCurrentMode,
    toggleCamera
  } = useSignRecognition();

  const t = useCallback((key: string): string => {
    return (translations[lang] as Record<string, unknown>)[key] as string || key;
  }, [lang]);

  const handleToggleLang = useCallback(() => {
    setLang(prev => prev === 'en' ? 'fr' : 'en');
  }, []);

  const handleStartStop = useCallback(() => {
    const video = videoFeedRef.current?.video;
    const canvas = videoFeedRef.current?.canvas;
    if (video && canvas) {
      toggleCamera(video, canvas);
    }
  }, [toggleCamera]);

  const getStatusText = useCallback(() => {
    switch (status) {
      case 'loading':
        return loadingProgress < 50 ? 'Loading config...' : 'Loading ONNX...';
      case 'ready':
        return 'Ready';
      case 'running':
        return 'Running';
      case 'error':
        return 'Error';
      default:
        return 'Loading...';
    }
  }, [status, loadingProgress]);

  return (
    <div className="min-h-screen bg-app">
      <Header lang={lang} onToggleLang={handleToggleLang} />

      {/* Hero - Minimal */}
      <section id="hero" className="pt-24 sm:pt-28 pb-8 px-4 sm:px-6">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-3xl sm:text-4xl md:text-5xl font-bold tracking-tight mb-4">
            Sign Language Recognition
            <br />
            <span className="text-gradient">Powered by S-TRM</span>
          </h1>
          <p className="text-zinc-500 text-sm sm:text-base max-w-xl mx-auto mb-6">
            {lang === 'en'
              ? 'Lightweight recursive architecture optimized for browser-based inference. 79K parameters, <15ms latency.'
              : 'Architecture recursive legere optimisee pour l\'inference navigateur. 79K parametres, <15ms latence.'}
          </p>
          <div className="flex items-center justify-center gap-3">
            <a href="#demo" className="btn-primary">
              {lang === 'en' ? 'Try Demo' : 'Essayer'}
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </a>
            <a href="#specs" className="btn-ghost">
              {lang === 'en' ? 'View Specs' : 'Voir Specs'}
            </a>
          </div>
        </div>
      </section>

      {/* Demo Section - Bento Grid */}
      <section id="demo" className="py-12 sm:py-16 px-4 sm:px-6">
        <div className="max-w-6xl mx-auto">
          <div className="mb-8">
            <h2 className="text-xl sm:text-2xl font-bold tracking-tight mb-2">
              {lang === 'en' ? 'Live Demo' : 'Demo en Direct'}
            </h2>
            <p className="text-zinc-500 text-sm">
              {lang === 'en' ? 'All inference runs locally in your browser' : 'Toute l\'inference s\'execute localement dans votre navigateur'}
            </p>
          </div>

          {/* Bento Grid Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-5">
            {/* Video Feed - Large */}
            <div className="lg:col-span-2 lg:row-span-2">
              <VideoFeed
                ref={videoFeedRef}
                isRunning={isRunning}
                currentMode={currentMode}
                onModeChange={setCurrentMode}
                onStartStop={handleStartStop}
                isModelReady={isModelReady}
                handDetected={handDetected}
                t={t}
              />
            </div>

            {/* Prediction Panel */}
            <div className="lg:row-span-2">
              <PredictionPanel
                status={status}
                statusText={getStatusText()}
                loadingProgress={loadingProgress}
                prediction={prediction}
                confidence={confidence}
                topPredictions={topPredictions}
                fps={fps}
                latency={latency}
                t={t}
              />
            </div>
          </div>
        </div>
      </section>

      {/* Technical Specs Section */}
      <section id="specs" className="py-12 sm:py-16 px-4 sm:px-6">
        <div className="section-divider mb-12" />

        <div className="max-w-6xl mx-auto">
          <div className="mb-8">
            <h2 className="text-xl sm:text-2xl font-bold tracking-tight mb-2">
              {lang === 'en' ? 'Technical Specifications' : 'Specifications Techniques'}
            </h2>
            <p className="text-zinc-500 text-sm max-w-2xl">
              {lang === 'en'
                ? 'S-TRM optimized for real-time inference with low latency. Temporal drift reduction via recursive architecture with state memory.'
                : 'S-TRM optimise pour l\'inference temps-reel a faible latence. Reduction de la derive temporelle via une architecture recursive a memoire d\'etat.'}
            </p>
          </div>

          <TechnicalSpecs language={lang} />

          {/* Data Visualizations */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-5 mt-6">
            <ConfusionMatrixPreview language={lang} />
            <LatencyChart language={lang} />
          </div>
        </div>
      </section>

      {/* Connect Section */}
      <section id="connect" className="py-12 sm:py-16 px-4 sm:px-6">
        <div className="section-divider mb-12" />

        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-10">
            <h2 className="text-xl sm:text-2xl font-bold tracking-tight mb-2">
              {lang === 'en' ? 'Connect & Inspect' : 'Connectez & Inspectez'}
            </h2>
            <p className="text-zinc-500 text-sm">
              {lang === 'en'
                ? 'Review the implementation, read the technical report, or get in touch.'
                : 'Examinez l\'implementation, lisez le rapport technique, ou contactez-nous.'}
            </p>
          </div>

          {/* Links Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* GitHub */}
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="card p-5 group hover:border-white/10"
            >
              <div className="w-10 h-10 rounded-xl bg-zinc-900 border border-white/5 flex items-center justify-center mb-4 group-hover:border-white/10 transition-colors">
                <svg className="w-5 h-5 text-zinc-500 group-hover:text-white transition-colors" fill="currentColor" viewBox="0 0 24 24">
                  <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.17 6.839 9.49.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.604-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.464-1.11-1.464-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836c.85.004 1.705.114 2.504.336 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.167 22 16.418 22 12c0-5.523-4.477-10-10-10z" />
                </svg>
              </div>
              <h3 className="font-medium text-sm mb-1">GitHub Repository</h3>
              <p className="text-xs text-zinc-600">
                {lang === 'en' ? 'View source code' : 'Voir le code source'}
              </p>
            </a>

            {/* LinkedIn - Hakim */}
            <a
              href="https://www.linkedin.com/in/hakim-djaalal78000/"
              target="_blank"
              rel="noopener noreferrer"
              className="card p-5 group hover:border-white/10"
            >
              <div className="w-10 h-10 rounded-xl bg-zinc-900 border border-white/5 flex items-center justify-center mb-4 group-hover:border-white/10 transition-colors">
                <svg className="w-5 h-5 text-zinc-500 group-hover:text-[#0A66C2] transition-colors" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                </svg>
              </div>
              <h3 className="font-medium text-sm mb-1">Hakim Djaalal</h3>
              <p className="text-xs text-zinc-600">LinkedIn</p>
            </a>

            {/* LinkedIn - Mouad */}
            <a
              href="https://www.linkedin.com/in/mouad-aoughane/"
              target="_blank"
              rel="noopener noreferrer"
              className="card p-5 group hover:border-white/10"
            >
              <div className="w-10 h-10 rounded-xl bg-zinc-900 border border-white/5 flex items-center justify-center mb-4 group-hover:border-white/10 transition-colors">
                <svg className="w-5 h-5 text-zinc-500 group-hover:text-[#0A66C2] transition-colors" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                </svg>
              </div>
              <h3 className="font-medium text-sm mb-1">Mouad Aoughane</h3>
              <p className="text-xs text-zinc-600">LinkedIn</p>
            </a>

            {/* Technical Report */}
            <a
              href="#"
              className="card p-5 group hover:border-white/10 cursor-not-allowed opacity-60"
              onClick={(e) => e.preventDefault()}
            >
              <div className="w-10 h-10 rounded-xl bg-zinc-900 border border-white/5 flex items-center justify-center mb-4">
                <svg className="w-5 h-5 text-zinc-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <h3 className="font-medium text-sm mb-1">Technical Report</h3>
              <p className="text-xs text-zinc-600">
                {lang === 'en' ? 'Coming soon' : 'Bientot disponible'}
              </p>
            </a>
          </div>

          {/* Author Info */}
          <div className="mt-10 text-center">
            <p className="text-xs text-zinc-600 mb-2">
              {lang === 'en' ? 'Developed by' : 'Developpe par'}
            </p>
            <div className="flex items-center justify-center gap-4">
              <span className="text-sm text-zinc-400">Hakim Djaalal</span>
              <span className="text-zinc-700">·</span>
              <span className="text-sm text-zinc-400">Mouad Aoughane</span>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-6 px-4 sm:px-6 border-t border-white/5">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4 text-center sm:text-left">
          <div className="flex items-center gap-2 text-xs text-zinc-600">
            <span className="font-mono">S-TRM.v1.2</span>
            <span className="text-zinc-800">|</span>
            <span>IPSSI MIA4 2025</span>
          </div>
          <div className="text-xs text-zinc-700">
            {lang === 'en' ? 'Client-side processing · Anonymous stats · MIT License' : 'Traitement local · Stats anonymes · Licence MIT'}
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
