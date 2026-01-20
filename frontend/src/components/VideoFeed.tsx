import { useRef, forwardRef, useImperativeHandle } from 'react';

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

        {/* Placeholder */}
        {!isRunning && (
          <div className="absolute inset-0 flex items-center justify-center bg-zinc-900/50">
            <div className="text-center">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-violet-500/10 border border-violet-500/20 flex items-center justify-center">
                <svg className="w-8 h-8 text-violet-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
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
            className={`py-2.5 px-3 rounded-lg text-sm font-medium transition-all ${
              currentMode === 'SPELLING'
                ? 'bg-violet-600 text-white'
                : 'bg-white/5 text-zinc-400 hover:bg-white/10 hover:text-white'
            }`}
          >
            {t('spelling')}
            <span className="block text-xs opacity-70 mt-0.5">A-Z</span>
          </button>
          <button
            onClick={() => onModeChange('WORD')}
            className={`py-2.5 px-3 rounded-lg text-sm font-medium transition-all ${
              currentMode === 'WORD'
                ? 'bg-violet-600 text-white'
                : 'bg-white/5 text-zinc-400 hover:bg-white/10 hover:text-white'
            }`}
          >
            {t('words')}
            <span className="block text-xs opacity-70 mt-0.5">HELLO, YES...</span>
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
