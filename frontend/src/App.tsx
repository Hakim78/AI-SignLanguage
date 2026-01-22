import { useState, useRef, useCallback } from 'react';
import { DotLottieReact } from '@lottiefiles/dotlottie-react';
import { Header } from './components/Header';
import { HeroSection } from './components/HeroSection';
import { VideoFeed } from './components/VideoFeed';
import { PredictionPanel } from './components/PredictionPanel';
import HowItWorksSection from './components/HowItWorksSection';
import { TechnicalSpecs } from './components/TechnicalSpecs';
import { ConfusionMatrixPreview } from './components/ConfusionMatrixPreview';
import { LatencyChart } from './components/LatencyChart';
import { ConnectSection } from './components/ConnectSection';
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

      {/* Hero Section */}
      <HeroSection language={lang} />

      {/* How It Works Section */}
      <HowItWorksSection language={lang} />

      {/* Demo Section - Bento Grid */}
      <section id="demo" className="py-12 sm:py-16 px-4 sm:px-6">
        <div className="max-w-6xl mx-auto">
          <div className="mb-8 flex items-center gap-4">
            {/* Lottie Animation */}
            <div className="w-12 h-12 flex-shrink-0">
              <DotLottieReact
                src="https://lottie.host/6416a8a8-714b-4335-aae1-8b9f6a3d7a82/ZQ68fHVUqK.lottie"
                loop
                autoplay
              />
            </div>
            <div>
              <h2 className="text-xl sm:text-2xl font-bold tracking-tight mb-1">
                {lang === 'en' ? 'Live Demo' : 'Demo en Direct'}
              </h2>
              <p className="text-zinc-500 text-sm">
                {lang === 'en' ? 'All inference runs locally in your browser' : 'Toute l\'inférence s\'exécute localement dans votre navigateur'}
              </p>
            </div>
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
                language={lang}
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
              {lang === 'en' ? 'Technical Specifications' : 'Spécifications Techniques'}
            </h2>
            <p className="text-zinc-500 text-sm max-w-2xl">
              {lang === 'en'
                ? 'S-TRM optimized for real-time inference with low latency. Temporal drift reduction via recursive architecture with state memory.'
                : 'S-TRM optimisé pour l\'inférence temps-réel à faible latence. Réduction de la dérive temporelle via une architecture récursive à mémoire d\'état.'}
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
      <ConnectSection language={lang} />

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
