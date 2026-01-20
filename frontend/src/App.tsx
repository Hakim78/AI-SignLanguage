import { useState, useRef, useCallback } from 'react';
import { Header } from './components/Header';
import { VideoFeed } from './components/VideoFeed';
import { PredictionPanel } from './components/PredictionPanel';
import { translations } from './i18n/translations';
import { useSignRecognition } from './hooks/useSignRecognition';

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
        return 'Error loading model';
      default:
        return t('loading');
    }
  }, [status, loadingProgress, t]);

  const steps = lang === 'en' ? [
    { num: '1', title: 'Enable Camera', desc: 'Click Start Camera to activate your webcam. All processing stays on your device.' },
    { num: '2', title: 'Show Sign', desc: 'Position your hand 30-50cm from camera with good lighting and plain background.' },
    { num: '3', title: 'Get Result', desc: 'See real-time ASL recognition with confidence scores and latency metrics.' }
  ] : [
    { num: '1', title: 'Activer la Caméra', desc: 'Cliquez sur Démarrer pour activer votre webcam. Tout le traitement reste sur votre appareil.' },
    { num: '2', title: 'Montrer le Signe', desc: 'Positionnez votre main à 30-50cm de la caméra avec un bon éclairage.' },
    { num: '3', title: 'Obtenir le Résultat', desc: 'Voyez la reconnaissance ASL en temps réel avec les scores de confiance.' }
  ];

  const features = lang === 'en' ? [
    { icon: '🔒', title: '100% Private', desc: 'All processing happens locally in your browser. No data is sent anywhere.' },
    { icon: '⚡', title: 'Real-time', desc: 'Instant recognition with ~15ms latency using WebGL acceleration.' },
    { icon: '🎯', title: '31 Signs', desc: 'Recognizes A-Z letters plus common words like HELLO, YES, NO.' },
    { icon: '🧠', title: 'Tiny Model', desc: 'Only 79K parameters (0.3MB) - efficient and fast on any device.' }
  ] : [
    { icon: '🔒', title: '100% Privé', desc: 'Tout le traitement se fait localement dans votre navigateur.' },
    { icon: '⚡', title: 'Temps Réel', desc: 'Reconnaissance instantanée avec ~15ms de latence via WebGL.' },
    { icon: '🎯', title: '31 Signes', desc: 'Reconnaît les lettres A-Z plus des mots comme HELLO, YES, NO.' },
    { icon: '🧠', title: 'Modèle Léger', desc: 'Seulement 79K paramètres (0.3MB) - efficace sur tout appareil.' }
  ];

  return (
    <div className="min-h-screen bg-gradient-main">
      <Header lang={lang} onToggleLang={handleToggleLang} t={t} />

      {/* Hero Section */}
      <section className="pt-32 pb-16 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <div className="badge mb-6">
            <span className="text-violet-400">✨</span>
            {lang === 'en' ? 'AI-Powered Demo' : 'Démo IA Gratuite'}
          </div>

          <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold mb-6 leading-tight">
            {lang === 'en' ? 'Recognize Sign Language' : 'Reconnaissance de la'}
            <br />
            <span className="gradient-text">
              {lang === 'en' ? 'in Real-Time' : 'Langue des Signes'}
            </span>
          </h1>

          <p className="text-gray-400 text-lg sm:text-xl mb-8 max-w-2xl mx-auto">
            {lang === 'en'
              ? 'Experience real-time ASL recognition powered by S-TRM architecture. 100% browser-based, no server required.'
              : 'Découvrez la reconnaissance ASL en temps réel propulsée par l\'architecture S-TRM. 100% dans le navigateur.'}
          </p>

          <a href="#demo" className="btn-primary text-lg">
            {lang === 'en' ? 'Try Demo' : 'Essayer la Démo'}
            <svg viewBox="0 0 24 24" className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 5v14M5 12l7 7 7-7"/>
            </svg>
          </a>
        </div>
      </section>

      {/* How it works */}
      <section id="features" className="py-16 px-4">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl sm:text-4xl font-bold text-center mb-4">
            {lang === 'en' ? 'How it' : 'Comment ça'}
            <span className="gradient-text"> {lang === 'en' ? 'works?' : 'marche ?'}</span>
          </h2>
          <p className="text-gray-400 text-center mb-12 max-w-xl mx-auto">
            {lang === 'en'
              ? 'No complex setup. No account needed. Just results.'
              : 'Pas d\'inscription complexe. Pas de compte requis. Juste des résultats.'}
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {steps.map((step, i) => (
              <div key={i} className="text-center relative">
                <div className="step-number">{step.num}</div>
                <div className="icon-box mx-auto -mt-8 mb-4 relative z-10">
                  <svg viewBox="0 0 24 24" className="w-6 h-6 text-violet-400" fill="currentColor">
                    {i === 0 && <path d="M15 8v8H5V8h10m1-2H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4V7c0-.55-.45-1-1-1z"/>}
                    {i === 1 && <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>}
                    {i === 2 && <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>}
                  </svg>
                </div>
                <h3 className="text-xl font-semibold mb-2">{step.title}</h3>
                <p className="text-gray-400 text-sm">{step.desc}</p>

                {i < 2 && (
                  <div className="hidden md:block absolute top-16 right-0 translate-x-1/2 text-gray-600">
                    - - - - →
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Demo Section */}
      <section id="demo" className="py-16 px-4">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-3xl sm:text-4xl font-bold text-center mb-12">
            {lang === 'en' ? 'Live' : 'Démo'}
            <span className="gradient-text"> Demo</span>
          </h2>

          <div className="grid grid-cols-1 lg:grid-cols-[1.4fr_1fr] gap-6">
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
      </section>

      {/* Features Grid */}
      <section className="py-16 px-4">
        <div className="max-w-5xl mx-auto">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {features.map((feat, i) => (
              <div key={i} className="feature-card text-center">
                <div className="text-3xl mb-3">{feat.icon}</div>
                <h3 className="font-semibold mb-2">{feat.title}</h3>
                <p className="text-gray-400 text-sm">{feat.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* About / Team */}
      <section id="about" className="py-16 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">
            {lang === 'en' ? 'About the' : 'À propos du'}
            <span className="gradient-text"> {lang === 'en' ? 'Project' : 'Projet'}</span>
          </h2>
          <p className="text-gray-400 mb-8 max-w-2xl mx-auto">
            {lang === 'en'
              ? 'This project was developed as part of IPSSI MIA4 program. S-TRM (Stateful Tiny Recursive Model) is a lightweight architecture designed for real-time sign language recognition.'
              : 'Ce projet a été développé dans le cadre du programme IPSSI MIA4. S-TRM (Stateful Tiny Recursive Model) est une architecture légère conçue pour la reconnaissance de la langue des signes en temps réel.'}
          </p>

          {/* Model Card */}
          <div className="glass-card p-6 mb-8">
            <h3 className="text-lg font-semibold mb-4 gradient-text">Model Card</h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 text-sm">
              {[
                { label: 'Architecture', value: 'S-TRM' },
                { label: 'Parameters', value: '~79K' },
                { label: 'Size', value: '0.3 MB' },
                { label: 'Classes', value: '31' },
                { label: 'Runtime', value: 'ONNX' },
                { label: 'Detection', value: 'MediaPipe' }
              ].map((spec, i) => (
                <div key={i} className="bg-white/5 rounded-lg p-3">
                  <div className="text-gray-500 text-xs uppercase mb-1">{spec.label}</div>
                  <div className="font-semibold">{spec.value}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Team */}
          <h3 className="text-xl font-semibold mb-6">
            {lang === 'en' ? 'Development Team' : 'Équipe de Développement'}
          </h3>
          <div className="flex flex-wrap justify-center gap-4">
            <a
              href="https://www.linkedin.com/in/hakim-djaalal78000/"
              target="_blank"
              rel="noopener noreferrer"
              className="feature-card flex items-center gap-4 px-6 py-4"
            >
              <div className="w-12 h-12 rounded-full bg-gradient-to-br from-violet-500 to-pink-500 flex items-center justify-center text-xl font-bold">
                H
              </div>
              <div className="text-left">
                <div className="font-semibold">Hakim Djaalal</div>
                <div className="text-gray-400 text-sm flex items-center gap-1">
                  <svg viewBox="0 0 24 24" className="w-4 h-4" fill="currentColor">
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
              className="feature-card flex items-center gap-4 px-6 py-4"
            >
              <div className="w-12 h-12 rounded-full bg-gradient-to-br from-cyan-500 to-violet-500 flex items-center justify-center text-xl font-bold">
                M
              </div>
              <div className="text-left">
                <div className="font-semibold">Mouad Aoughane</div>
                <div className="text-gray-400 text-sm flex items-center gap-1">
                  <svg viewBox="0 0 24 24" className="w-4 h-4" fill="currentColor">
                    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                  </svg>
                  LinkedIn
                </div>
              </div>
            </a>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-4 border-t border-white/5">
        <div className="max-w-4xl mx-auto text-center text-gray-500 text-sm">
          <p>IPSSI MIA4 - 2025 | S-TRM: Stateful Tiny Recursive Model</p>
          <p className="mt-2">
            {lang === 'en'
              ? '100% client-side processing • No data collection • Open source'
              : 'Traitement 100% côté client • Aucune collecte de données • Open source'}
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
