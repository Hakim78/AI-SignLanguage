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

        {/* Hand Position Guide with Lottie */}
        {isRunning && !handDetected && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="relative w-48 h-48 sm:w-56 sm:h-56">
              <DotLottieReact
                src="https://lottie.host/7f8a3c4e-b5d9-4a1f-8e2c-6d9a7b5c3e4f/scan-hand.lottie"
                loop
                autoplay
                style={{ width: '100%', height: '100%', opacity: 0.6 }}
              />
              {/* Frame overlay */}
              <div className="absolute inset-0 border border-dashed border-indigo-500/30 rounded-xl" />
              <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-indigo-400/60 rounded-tl" />
              <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-indigo-400/60 rounded-tr" />
              <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-indigo-400/60 rounded-bl" />
              <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-indigo-400/60 rounded-br" />
              {/* Center hand icon fallback */}
              <div className="absolute inset-0 flex items-center justify-center">
                <svg className="w-12 h-12 text-indigo-500/40 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M7 11.5V14m0-2.5v-6a1.5 1.5 0 113 0m-3 6a1.5 1.5 0 00-3 0v2a7.5 7.5 0 0015 0v-5a1.5 1.5 0 00-3 0m-6-3V11m0-5.5v-1a1.5 1.5 0 013 0v1m0 0V11m0-5.5a1.5 1.5 0 013 0v3m0 0V11" />
                </svg>
              </div>
            </div>
          </div>
        )}

        {/* Active Detection Animation */}
        {isRunning && handDetected && (
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 pointer-events-none">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-green-500/20 border border-green-500/30">
              <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
              <span className="text-xs font-mono text-green-400">Processing</span>
            </div>
          </div>
        )}

        {/* Idle State with Lottie */}
        {!isRunning && (
          <div className="absolute inset-0 flex items-center justify-center bg-zinc-950/90">
            <div className="text-center">
              <div className="w-20 h-20 mx-auto mb-4 rounded-2xl bg-zinc-900/50 border border-white/5 flex items-center justify-center">
                <DotLottieReact
                  src="https://lottie.host/4db68bbd-31f6-4cd8-84eb-189de081159a/camera.lottie"
                  loop
                  autoplay
                  style={{ width: 48, height: 48 }}
                />
              </div>
              <p className="text-xs font-mono text-zinc-500">{t('waitingCam')}</p>
              <p className="text-[10px] text-zinc-700 mt-1">Click Start to begin</p>
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
    </div>
  );
});

VideoFeed.displayName = 'VideoFeed';
