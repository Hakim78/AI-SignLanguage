import { useState, useRef, useCallback } from 'react';
import { Header } from './components/Header';
import { VideoFeed } from './components/VideoFeed';
import { PredictionPanel } from './components/PredictionPanel';
import HowItWorksSection from './components/HowItWorksSection';
import { translations } from './i18n/translations';
import { useSignRecognition } from './hooks/useSignRecognition';
import { DotLottieReact } from '@lottiefiles/dotlottie-react';

type Language = 'en' | 'fr';

interface VideoFeedRef {
  video: HTMLVideoElement | null;
  canvas: HTMLCanvasElement | null;
}

function App() {
  const [lang, setLang] = useState<Language>('fr');
  const videoFeedRef = useRef<VideoFeedRef>(null);

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
        return loadingProgress < 50 ? t('loadingConfig') : t('loadingOnnx');
      case 'ready':
        return t('ready');
      case 'running':
        return t('running');
      case 'error':
        return 'Error';
      default:
        return t('loading');
    }
  }, [status, loadingProgress, t]);

  const specs = [
    { label: 'Architecture', value: 'S-TRM' },
    { label: lang === 'en' ? 'Parameters' : 'Parametres', value: '79K' },
    { label: lang === 'en' ? 'Size' : 'Taille', value: '0.3 MB' },
    { label: 'Classes', value: '31' },
    { label: 'Runtime', value: 'ONNX' },
    { label: 'Detection', value: 'MediaPipe' }
  ];

  return (
    <div className="min-h-screen bg-gradient-main">
      <Header lang={lang} onToggleLang={handleToggleLang} />

      {/* Hero */}
      <section className="pt-28 sm:pt-32 pb-12 sm:pb-16 px-4">
        <div className="max-w-3xl mx-auto text-center">
          <p className="text-violet-400 text-sm font-medium tracking-wide uppercase mb-4">
            {lang === 'en' ? 'Real-time AI Demo' : 'Demo IA Temps Reel'}
          </p>
          <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold leading-tight mb-6">
            {lang === 'en' ? 'Sign Language' : 'Reconnaissance'}
            <br />
            <span className="text-gradient">
              {lang === 'en' ? 'Recognition' : 'Langue des Signes'}
            </span>
          </h1>
          <p className="text-zinc-400 text-base sm:text-lg max-w-xl mx-auto mb-8">
            {lang === 'en'
              ? 'Experience ASL recognition powered by S-TRM architecture. Runs entirely in your browser.'
              : 'Decouvrez la reconnaissance ASL propulsee par S-TRM. Fonctionne entierement dans votre navigateur.'}
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
            <a href="#demo" className="btn-primary w-full sm:w-auto">
              {lang === 'en' ? 'Launch Demo' : 'Lancer la Demo'}
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </a>
            <a href="#about" className="btn-secondary w-full sm:w-auto">
              {lang === 'en' ? 'Learn more' : 'En savoir plus'}
            </a>
          </div>
        </div>
      </section>

      {/* How it works */}
      <HowItWorksSection language={lang} />

      {/* Demo */}
      <section id="demo" className="py-12 sm:py-16 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-8 sm:mb-10">
            <h2 className="text-2xl sm:text-3xl font-bold mb-3">
              {lang === 'en' ? 'Live Demo' : 'Demo en Direct'}
            </h2>
            <p className="text-zinc-500 text-sm sm:text-base">
              {lang === 'en' ? 'Try the model directly in your browser' : 'Testez le modele directement dans votre navigateur'}
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
            <div className="lg:col-span-3">
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
            <div className="lg:col-span-2">
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

      {/* About */}
      <section id="about" className="py-12 sm:py-16 px-4">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-10">
            <h2 className="text-2xl sm:text-3xl font-bold mb-3">
              {lang === 'en' ? 'About the Project' : 'A propos du Projet'}
            </h2>
            <p className="text-zinc-500 text-sm sm:text-base max-w-2xl mx-auto">
              {lang === 'en'
                ? 'S-TRM (Stateful Tiny Recursive Model) is a lightweight architecture designed for real-time sign language recognition. Developed as part of IPSSI MIA4 program.'
                : 'S-TRM (Stateful Tiny Recursive Model) est une architecture legere concue pour la reconnaissance de la langue des signes en temps reel. Developpe dans le cadre du programme IPSSI MIA4.'}
            </p>
          </div>

          {/* Specs */}
          <div className="card p-6 mb-8">
            <h3 className="text-sm font-medium text-zinc-400 uppercase tracking-wide mb-4">
              {lang === 'en' ? 'Technical Specifications' : 'Specifications Techniques'}
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-4">
              {specs.map((spec, i) => (
                <div key={i} className="text-center p-3 rounded-lg bg-white/[0.02]">
                  <div className="text-xs text-zinc-500 mb-1">{spec.label}</div>
                  <div className="font-semibold text-sm">{spec.value}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Privacy notice */}
          <div className="card p-6 mb-8 border-green-500/20">
            <div className="flex items-start gap-4">
              <div className="w-16 h-16 flex-shrink-0 -ml-2 -mt-2">
                <DotLottieReact
                  src="https://lottie.host/e1c2ab12-3852-4cbc-8eca-c6037469eea1/l9f1TWlH3g.lottie"
                  loop
                  autoplay
                />
              </div>
              <div>
                <h3 className="font-semibold mb-1">
                  {lang === 'en' ? '100% Private' : '100% Prive'}
                </h3>
                <p className="text-zinc-500 text-sm">
                  {lang === 'en'
                    ? 'All processing happens locally in your browser. No video data is ever sent to any server.'
                    : 'Tout le traitement se fait localement dans votre navigateur. Aucune video n\'est envoyee a un serveur.'}
                </p>
              </div>
            </div>
          </div>

          {/* Team */}
          <div className="text-center">
            <div className="w-24 h-24 mx-auto mb-4">
              <DotLottieReact
                src="https://lottie.host/f5b74eb2-0985-4c6c-8f67-6a1000b9bf0f/R6PHcntLFr.lottie"
                loop
                autoplay
              />
            </div>
            <h3 className="text-sm font-medium text-zinc-400 uppercase tracking-wide mb-6">
              {lang === 'en' ? 'Development Team' : 'Equipe de Developpement'}
            </h3>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <a
                href="https://www.linkedin.com/in/hakim-djaalal78000/"
                target="_blank"
                rel="noopener noreferrer"
                className="card flex items-center gap-4 px-5 py-4 w-full sm:w-auto hover:border-violet-500/30"
              >
                <div className="w-11 h-11 rounded-full bg-gradient-to-br from-violet-600 to-purple-600 flex items-center justify-center font-semibold">
                  H
                </div>
                <div className="text-left">
                  <div className="font-medium text-sm">Hakim Djaalal</div>
                  <div className="text-zinc-500 text-xs flex items-center gap-1.5">
                    <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                    </svg>
                    LinkedIn
                  </div>
                </div>
              </a>
              <a
                href="https://www.linkedin.com/in/mouad-aoughane-2943b6208/"
                target="_blank"
                rel="noopener noreferrer"
                className="card flex items-center gap-4 px-5 py-4 w-full sm:w-auto hover:border-violet-500/30"
              >
                <div className="w-11 h-11 rounded-full bg-gradient-to-br from-cyan-600 to-blue-600 flex items-center justify-center font-semibold">
                  M
                </div>
                <div className="text-left">
                  <div className="font-medium text-sm">Mouad Aoughane</div>
                  <div className="text-zinc-500 text-xs flex items-center gap-1.5">
                    <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                    </svg>
                    LinkedIn
                  </div>
                </div>
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-4 border-t border-white/5">
        <div className="max-w-4xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4 text-center sm:text-left">
          <div className="text-zinc-600 text-sm">
            IPSSI MIA4 - 2025
          </div>
          <div className="text-zinc-600 text-xs">
            {lang === 'en' ? 'Client-side processing' : 'Traitement cote client'} · {lang === 'en' ? 'No data collection' : 'Aucune collecte'}
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
