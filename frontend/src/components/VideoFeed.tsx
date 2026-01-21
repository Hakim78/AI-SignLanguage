import { useRef, forwardRef, useImperativeHandle } from 'react';
import { DotLottieReact } from '@lottiefiles/dotlottie-react';

interface VideoFeedProps {
  isRunning: boolean;
  currentMode: 'SPELLING' | 'WORD';
  onModeChange: (mode: 'SPELLING' | 'WORD') => void;
  onStartStop: () => void;
  isModelReady: boolean;
  handDetected: boolean;
  t: (key: string) => string;
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
  t
}, ref) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useImperativeHandle(ref, () => ({
    video: videoRef.current,
    canvas: canvasRef.current
  }));

  return (
    <div className="card overflow-hidden h-full">
      {/* Video */}
      <div className="relative bg-zinc-900 aspect-[4/3]">
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

        {/* Overlays */}
        <div className="absolute top-3 left-3 right-3 flex justify-between items-start">
          <div className="flex items-center gap-2">
            {isRunning && (
              <div className="flex items-center gap-2 bg-zinc-900/80 backdrop-blur-sm px-2.5 py-1.5 rounded-md">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
                </span>
                <span className="text-xs font-medium">LIVE</span>
              </div>
            )}
            <div className="bg-zinc-900/80 backdrop-blur-sm px-2.5 py-1.5 rounded-md text-xs font-medium">
              {currentMode === 'SPELLING' ? 'A-Z' : 'WORDS'}
            </div>
          </div>

          <div className={`flex items-center gap-2 px-2.5 py-1.5 rounded-md text-xs font-medium transition-all ${
            !isRunning
              ? 'bg-zinc-900/80 text-zinc-400'
              : handDetected
                ? 'bg-green-500/90 text-white'
                : 'bg-zinc-900/80 text-zinc-400'
          }`}>
            {handDetected ? (
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            ) : (
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            )}
            <span className="hidden sm:inline">
              {!isRunning ? t('waitingCam') : handDetected ? t('handDetected') : t('noHand')}
            </span>
          </div>
        </div>

        {/* Hand position guide overlay */}
        {isRunning && !handDetected && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="relative w-48 h-48 sm:w-56 sm:h-56">
              {/* Animated guide box */}
              <div className="absolute inset-0 border-2 border-dashed border-violet-500/50 rounded-2xl animate-pulse" />
              {/* Corner markers */}
              <div className="absolute top-0 left-0 w-6 h-6 border-t-2 border-l-2 border-violet-400 rounded-tl-lg" />
              <div className="absolute top-0 right-0 w-6 h-6 border-t-2 border-r-2 border-violet-400 rounded-tr-lg" />
              <div className="absolute bottom-0 left-0 w-6 h-6 border-b-2 border-l-2 border-violet-400 rounded-bl-lg" />
              <div className="absolute bottom-0 right-0 w-6 h-6 border-b-2 border-r-2 border-violet-400 rounded-br-lg" />
              {/* Hand scan Lottie */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-28 h-28">
                  <DotLottieReact
                    src="https://lottie.host/6416a8a8-714b-4335-aae1-8b9f6a3d7a82/ZQ68fHVUqK.lottie"
                    loop
                    autoplay
                  />
                </div>
              </div>
              {/* Instruction text */}
              <div className="absolute -bottom-10 left-1/2 -translate-x-1/2 whitespace-nowrap">
                <span className="text-xs text-violet-300 bg-zinc-900/80 px-3 py-1 rounded-full">
                  {t('positionHand')}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Placeholder */}
        {!isRunning && (
          <div className="absolute inset-0 flex items-center justify-center bg-zinc-900/50">
            <div className="text-center">
              <div className="w-24 h-24 mx-auto mb-4">
                <DotLottieReact
                  src="https://lottie.host/b9193d59-942f-4c9d-ad71-2d5d09038363/hSp0icdzZB.lottie"
                  loop
                  autoplay
                />
              </div>
              <p className="text-zinc-500 text-sm">{t('waitingCam')}</p>
            </div>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="p-4 sm:p-5 space-y-4">
        {/* Mode selector */}
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => onModeChange('SPELLING')}
            className={`py-2.5 px-3 rounded-lg text-sm font-medium transition-all flex flex-col items-center justify-center ${
              currentMode === 'SPELLING'
                ? 'bg-violet-600 text-white'
                : 'bg-white/5 text-zinc-400 hover:bg-white/10 hover:text-white'
            }`}
          >
            <div className="w-10 h-10 -mt-1 -mb-1">
              <DotLottieReact
                src="https://lottie.host/991d2e44-8657-4beb-a3fb-db93f543358c/NdsOxVga9Y.lottie"
                loop
                autoplay
              />
            </div>
            <span className="text-xs opacity-70">{t('spelling')}</span>
          </button>
          <button
            onClick={() => onModeChange('WORD')}
            className={`py-2.5 px-3 rounded-lg text-sm font-medium transition-all flex flex-col items-center justify-center ${
              currentMode === 'WORD'
                ? 'bg-violet-600 text-white'
                : 'bg-white/5 text-zinc-400 hover:bg-white/10 hover:text-white'
            }`}
          >
            <span className="text-lg mb-1">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
            </span>
            <span className="text-xs opacity-70">{t('words')}</span>
          </button>
        </div>

        {/* Start button */}
        <button
          onClick={onStartStop}
          disabled={!isModelReady}
          className={`w-full py-3.5 rounded-lg font-medium text-sm transition-all flex items-center justify-center gap-2 ${
            !isModelReady
              ? 'bg-zinc-800 text-zinc-600 cursor-not-allowed'
              : isRunning
                ? 'bg-red-500/90 text-white hover:bg-red-500'
                : 'btn-primary'
          }`}
        >
          {!isModelReady ? (
            <>
              <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              {t('loading')}
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
    </div>
  );
});

VideoFeed.displayName = 'VideoFeed';
