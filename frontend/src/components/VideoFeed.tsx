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
    <div className="glass-card overflow-hidden">
      {/* Video Container */}
      <div className="relative bg-black/50 aspect-[4/3]">
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
        <div className="absolute top-4 left-4 right-4 flex justify-between items-start">
          <div className="flex gap-2">
            {isRunning && (
              <div className="bg-black/70 backdrop-blur-sm px-3 py-1.5 rounded-full text-xs font-semibold flex items-center gap-2">
                <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                LIVE
              </div>
            )}
            <div className="bg-black/70 backdrop-blur-sm px-3 py-1.5 rounded-full text-xs font-semibold">
              {currentMode}
            </div>
          </div>

          {/* Hand Detection Indicator */}
          <div className={`px-3 py-1.5 rounded-full text-xs font-semibold flex items-center gap-2 transition-all ${
            !isRunning
              ? 'bg-black/70 text-gray-400'
              : handDetected
                ? 'bg-green-500/90 text-white'
                : 'bg-red-500/90 text-white'
          }`}>
            {isRunning && (
              handDetected ? (
                <svg viewBox="0 0 24 24" className="w-3.5 h-3.5" fill="currentColor">
                  <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
                </svg>
              ) : (
                <svg viewBox="0 0 24 24" className="w-3.5 h-3.5" fill="currentColor">
                  <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
                </svg>
              )
            )}
            <span>
              {!isRunning ? t('waitingCam') : handDetected ? t('handDetected') : t('noHand')}
            </span>
          </div>
        </div>

        {/* Center placeholder when not running */}
        {!isRunning && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-violet-500/10 border border-violet-500/30 flex items-center justify-center">
                <svg viewBox="0 0 24 24" className="w-10 h-10 text-violet-400" fill="currentColor">
                  <path d="M15 8v8H5V8h10m1-2H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4V7c0-.55-.45-1-1-1z"/>
                </svg>
              </div>
              <p className="text-gray-400 text-sm">{t('waitingCam')}</p>
            </div>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="p-5">
        <div className="flex gap-3 mb-4">
          <button
            onClick={() => onModeChange('SPELLING')}
            className={`flex-1 py-3 px-4 rounded-xl font-semibold transition-all ${
              currentMode === 'SPELLING'
                ? 'bg-gradient-to-r from-violet-500 to-pink-500 text-white'
                : 'bg-white/5 border border-white/10 text-gray-400 hover:bg-white/10 hover:text-white'
            }`}
          >
            <span>{t('spelling')}</span>
            <small className="block text-xs opacity-80 mt-1">A-Z</small>
          </button>
          <button
            onClick={() => onModeChange('WORD')}
            className={`flex-1 py-3 px-4 rounded-xl font-semibold transition-all ${
              currentMode === 'WORD'
                ? 'bg-gradient-to-r from-violet-500 to-pink-500 text-white'
                : 'bg-white/5 border border-white/10 text-gray-400 hover:bg-white/10 hover:text-white'
            }`}
          >
            <span>{t('words')}</span>
            <small className="block text-xs opacity-80 mt-1">HELLO, YES...</small>
          </button>
        </div>

        <button
          onClick={onStartStop}
          disabled={!isModelReady}
          className={`w-full py-4 rounded-xl font-bold text-lg transition-all flex items-center justify-center gap-2 ${
            !isModelReady
              ? 'bg-gray-700/50 text-gray-500 cursor-not-allowed'
              : isRunning
                ? 'bg-red-500 text-white hover:bg-red-600'
                : 'btn-primary hover:scale-[1.02]'
          }`}
        >
          {isRunning ? (
            <>
              <svg viewBox="0 0 24 24" className="w-5 h-5" fill="currentColor">
                <path d="M6 6h12v12H6z"/>
              </svg>
              {t('stop')}
            </>
          ) : (
            <>
              <svg viewBox="0 0 24 24" className="w-5 h-5" fill="currentColor">
                <path d="M8 5v14l11-7z"/>
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
