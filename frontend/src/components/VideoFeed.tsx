import { useRef, forwardRef, useImperativeHandle, useState, useEffect } from 'react';
import { DotLottieReact } from '@lottiefiles/dotlottie-react';
import aslAlphabetImg from '../assets/image_signLanguage.png';

interface VideoFeedProps {
  isRunning: boolean;
  currentMode: 'SPELLING' | 'WORD';
  onModeChange: (mode: 'SPELLING' | 'WORD') => void;
  onStartStop: () => void;
  isModelReady: boolean;
  handDetected: boolean;
  t: (key: string) => string;
  language: 'en' | 'fr';
}

export interface VideoFeedRef {
  video: HTMLVideoElement | null;
  canvas: HTMLCanvasElement | null;
}

export const VideoFeed = forwardRef<VideoFeedRef, VideoFeedProps>(({
  isRunning,
  currentMode,
  onModeChange,
  onStartStop,
  isModelReady,
  handDetected,
  t,
  language
}, ref) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [showGuide, setShowGuide] = useState(false);

  useImperativeHandle(ref, () => ({
    video: videoRef.current,
    canvas: canvasRef.current
  }));

  // Close modal on ESC key
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && showGuide) {
        setShowGuide(false);
      }
    };
    window.addEventListener('keydown', handleEsc);
    return () => window.removeEventListener('keydown', handleEsc);
  }, [showGuide]);

  return (
    <div className="bento-item h-full flex flex-col">
      {/* Video Container */}
      <div className="relative bg-zinc-950 aspect-[4/3] flex-shrink-0">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover scale-x-[-1]"
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full scale-x-[-1]"
        />

        {/* Minimal Overlays */}
        <div className="absolute top-3 left-3 right-3 flex justify-between items-start">
          {/* Status Badge */}
          <div className="flex items-center gap-2">
            {isRunning && (
              <div className="flex items-center gap-1.5 bg-zinc-950/80 backdrop-blur-sm px-2 py-1 rounded-md">
                <span className="relative flex h-1.5 w-1.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-red-500"></span>
                </span>
                <span className="text-[10px] font-mono text-zinc-400">REC</span>
              </div>
            )}
          </div>

          <div className="flex items-center gap-2">
            {/* ASL Guide Button */}
            <button
              onClick={() => setShowGuide(true)}
              className="flex items-center gap-1.5 bg-indigo-500/20 hover:bg-indigo-500/30 backdrop-blur-sm px-2.5 py-1 rounded-md text-[10px] font-mono text-indigo-300 transition-all border border-indigo-500/30 hover:border-indigo-500/50"
              title={language === 'en' ? 'ASL Alphabet Guide' : 'Guide Alphabet ASL'}
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span className="hidden sm:inline">{language === 'en' ? 'Guide' : 'Guide'}</span>
            </button>

            {/* Hand Detection Status */}
            <div className={`flex items-center gap-1.5 px-2 py-1 rounded-md text-[10px] font-mono transition-all ${
              !isRunning
                ? 'bg-zinc-950/80 text-zinc-600'
                : handDetected
                  ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                  : 'bg-zinc-950/80 text-zinc-500'
            }`}>
              <div className={`w-1.5 h-1.5 rounded-full ${
                !isRunning ? 'bg-zinc-600' : handDetected ? 'bg-green-400' : 'bg-zinc-500'
              }`} />
              {handDetected ? 'DETECTED' : 'NO_HAND'}
            </div>
          </div>
        </div>

        {/* Hand Position Guide - Animated Frame */}
        {isRunning && !handDetected && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="relative w-48 h-48 sm:w-56 sm:h-56">
              {/* Animated scan line */}
              <div className="absolute inset-0 overflow-hidden rounded-xl">
                <div className="absolute inset-x-0 h-1 bg-gradient-to-r from-transparent via-indigo-500/50 to-transparent animate-scan-vertical" />
              </div>
              {/* Frame border */}
              <div className="absolute inset-0 border-2 border-dashed border-indigo-500/30 rounded-xl animate-pulse" />
              {/* Corner markers */}
              <div className="absolute top-0 left-0 w-6 h-6 border-t-2 border-l-2 border-indigo-400 rounded-tl-lg" />
              <div className="absolute top-0 right-0 w-6 h-6 border-t-2 border-r-2 border-indigo-400 rounded-tr-lg" />
              <div className="absolute bottom-0 left-0 w-6 h-6 border-b-2 border-l-2 border-indigo-400 rounded-bl-lg" />
              <div className="absolute bottom-0 right-0 w-6 h-6 border-b-2 border-r-2 border-indigo-400 rounded-br-lg" />
              {/* Center hand icon */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="relative">
                  <svg className="w-16 h-16 text-indigo-500/50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M7 11.5V14m0-2.5v-6a1.5 1.5 0 113 0m-3 6a1.5 1.5 0 00-3 0v2a7.5 7.5 0 0015 0v-5a1.5 1.5 0 00-3 0m-6-3V11m0-5.5v-1a1.5 1.5 0 013 0v1m0 0V11m0-5.5a1.5 1.5 0 013 0v3m0 0V11" />
                  </svg>
                  {/* Pulse ring */}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-20 h-20 rounded-full border border-indigo-500/30 animate-ping" />
                  </div>
                </div>
              </div>
              {/* Instruction text */}
              <div className="absolute -bottom-8 left-0 right-0 text-center">
                <span className="text-xs font-mono text-indigo-400/70">Position hand in frame</span>
              </div>
            </div>
          </div>
        )}

        {/* Active Detection Indicator */}
        {isRunning && handDetected && (
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 pointer-events-none">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-green-500/20 border border-green-500/30 backdrop-blur-sm">
              <div className="relative w-2 h-2">
                <div className="absolute inset-0 rounded-full bg-green-400 animate-ping" />
                <div className="relative rounded-full w-2 h-2 bg-green-400" />
              </div>
              <span className="text-xs font-mono text-green-400">Processing</span>
            </div>
          </div>
        )}

        {/* Idle State */}
        {!isRunning && (
          <div className="absolute inset-0 flex items-center justify-center bg-zinc-950/90">
            <div className="text-center">
              <div className="w-20 h-20 mx-auto mb-4 rounded-2xl bg-zinc-900/80 border border-white/10 flex items-center justify-center relative overflow-hidden">
                {/* Animated gradient */}
                <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/10 via-transparent to-purple-500/10 animate-gradient" />
                {/* Camera Lottie Animation */}
                <div className="w-14 h-14 relative z-10">
                  <DotLottieReact
                    src="https://lottie.host/b9193d59-942f-4c9d-ad71-2d5d09038363/hSp0icdzZB.lottie"
                    loop
                    autoplay
                  />
                </div>
              </div>
              <p className="text-xs font-mono text-zinc-500 mb-1">{t('waitingCam')}</p>
              <p className="text-[10px] text-zinc-600">Click Start to begin</p>
            </div>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="p-4 space-y-4 flex-1 flex flex-col justify-end">
        {/* Segmented Mode Control */}
        <div className="segmented-control w-full">
          <button
            onClick={() => onModeChange('SPELLING')}
            className={`segmented-btn flex-1 ${currentMode === 'SPELLING' ? 'active' : ''}`}
          >
            <span className="flex items-center justify-center gap-1.5">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" />
              </svg>
              A-Z
            </span>
          </button>
          <button
            onClick={() => onModeChange('WORD')}
            className={`segmented-btn flex-1 ${currentMode === 'WORD' ? 'active' : ''}`}
          >
            <span className="flex items-center justify-center gap-1.5">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
              Words
            </span>
          </button>
        </div>

        {/* Start/Stop Button */}
        <button
          onClick={onStartStop}
          disabled={!isModelReady}
          className={`w-full py-3 rounded-lg font-medium text-sm transition-all flex items-center justify-center gap-2 ${
            !isModelReady
              ? 'bg-zinc-900 text-zinc-600 cursor-not-allowed border border-white/5'
              : isRunning
                ? 'bg-red-500/10 text-red-400 border border-red-500/30 hover:bg-red-500/20'
                : 'btn-primary'
          }`}
        >
          {!isModelReady ? (
            <>
              <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
              </svg>
              <span className="font-mono text-xs">loading_model...</span>
            </>
          ) : isRunning ? (
            <>
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <rect x="6" y="6" width="12" height="12" rx="1" />
              </svg>
              {t('stop')}
            </>
          ) : (
            <>
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z" />
              </svg>
              {t('start')}
            </>
          )}
        </button>
      </div>

      {/* ASL Alphabet Guide Modal */}
      {showGuide && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm"
          onClick={() => setShowGuide(false)}
        >
          <div
            className="relative max-w-2xl w-full bg-zinc-900 rounded-2xl border border-white/10 shadow-2xl overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-white/10 bg-zinc-900/50">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center">
                  <svg className="w-5 h-5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 11.5V14m0-2.5v-6a1.5 1.5 0 113 0m-3 6a1.5 1.5 0 00-3 0v2a7.5 7.5 0 0015 0v-5a1.5 1.5 0 00-3 0m-6-3V11m0-5.5v-1a1.5 1.5 0 013 0v1m0 0V11m0-5.5a1.5 1.5 0 013 0v3m0 0V11" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-lg font-bold text-white">
                    {language === 'en' ? 'ASL Alphabet Reference' : 'Référence Alphabet ASL'}
                  </h3>
                  <p className="text-xs text-zinc-500">
                    {language === 'en' ? 'Practice these hand signs' : 'Pratiquez ces signes de la main'}
                  </p>
                </div>
              </div>
              <button
                onClick={() => setShowGuide(false)}
                className="p-2 rounded-lg hover:bg-white/10 transition-colors text-zinc-400 hover:text-white"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Modal Content */}
            <div className="p-6">
              <div className="relative rounded-xl overflow-hidden bg-white">
                <img
                  src={aslAlphabetImg}
                  alt="ASL Alphabet Guide"
                  className="w-full h-auto"
                />
              </div>

              {/* Tips */}
              <div className="mt-4 p-4 rounded-xl bg-zinc-800/50 border border-white/5">
                <h4 className="text-sm font-semibold text-white mb-2 flex items-center gap-2">
                  <svg className="w-4 h-4 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  {language === 'en' ? 'Tips for better recognition' : 'Conseils pour une meilleure reconnaissance'}
                </h4>
                <ul className="text-xs text-zinc-400 space-y-1">
                  <li>• {language === 'en' ? 'Position your hand 30-50cm from the camera' : 'Positionnez votre main à 30-50cm de la caméra'}</li>
                  <li>• {language === 'en' ? 'Use good lighting (natural light works best)' : 'Utilisez un bon éclairage (lumière naturelle)'}</li>
                  <li>• {language === 'en' ? 'Keep a neutral background if possible' : 'Gardez un fond neutre si possible'}</li>
                  <li>• {language === 'en' ? 'Hold each sign steady for accurate detection' : 'Maintenez chaque signe stable pour une détection précise'}</li>
                </ul>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="px-6 py-4 border-t border-white/10 bg-zinc-900/50 flex justify-between items-center">
              <span className="text-xs text-zinc-600 font-mono">ESC {language === 'en' ? 'to close' : 'pour fermer'}</span>
              <button
                onClick={() => setShowGuide(false)}
                className="px-4 py-2 rounded-lg bg-indigo-500/20 text-indigo-300 text-sm font-medium hover:bg-indigo-500/30 transition-colors border border-indigo-500/30"
              >
                {language === 'en' ? 'Got it!' : 'Compris !'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
});

VideoFeed.displayName = 'VideoFeed';
