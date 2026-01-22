import { useRef, useState, useCallback, useEffect } from 'react';
import * as ort from 'onnxruntime-web';

// Configure ONNX Runtime - use CDN for WASM files
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/';

// MediaPipe types (loaded via script tags in index.html)
declare global {
  interface Window {
    Hands: new (config: { locateFile: (file: string) => string }) => MediaPipeHands;
    Camera: new (video: HTMLVideoElement, config: { onFrame: () => Promise<void>; width: number; height: number }) => MediaPipeCamera;
    drawConnectors: (ctx: CanvasRenderingContext2D, landmarks: Landmark[], connections: [number, number][], style: { color: string; lineWidth: number }) => void;
    drawLandmarks: (ctx: CanvasRenderingContext2D, landmarks: Landmark[], style: { color: string; lineWidth: number; radius: number }) => void;
    HAND_CONNECTIONS: [number, number][];
  }
}

interface Landmark {
  x: number;
  y: number;
  z: number;
}

interface MediaPipeHands {
  setOptions: (options: {
    maxNumHands: number;
    modelComplexity: number;
    minDetectionConfidence: number;
    minTrackingConfidence: number;
  }) => void;
  onResults: (callback: (results: HandResults) => void) => void;
  send: (input: { image: HTMLVideoElement }) => Promise<void>;
}

interface MediaPipeCamera {
  start: () => void;
  stop: () => void;
}

interface HandResults {
  image: HTMLVideoElement;
  multiHandLandmarks?: Landmark[][];
}

interface Config {
  class_names: string[];
  scaler: {
    mean: number[];
    scale: number[];
  };
  dynamic_classes: string[];
}

interface Prediction {
  label: string;
  probability: number;
}

export function useSignRecognition() {
  const [isRunning, setIsRunning] = useState(false);
  const [isModelReady, setIsModelReady] = useState(false);
  const [status, setStatus] = useState<'loading' | 'ready' | 'running' | 'error'>('loading');
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [prediction, setPrediction] = useState('-');
  const [confidence, setConfidence] = useState(0);
  const [topPredictions, setTopPredictions] = useState<Prediction[]>([]);
  const [fps, setFps] = useState(0);
  const [latency, setLatency] = useState(0);
  const [handDetected, setHandDetected] = useState(false);
  const [currentMode, setCurrentMode] = useState<'SPELLING' | 'WORD'>('SPELLING');

  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const configRef = useRef<Config | null>(null);
  const cameraRef = useRef<MediaPipeCamera | null>(null);
  const handsRef = useRef<MediaPipeHands | null>(null);
  const frameCountRef = useRef(0);
  const lastFpsTimeRef = useRef(performance.now());
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const onHandResultsRef = useRef<((results: HandResults) => void) | null>(null);

  // Softmax function
  const softmax = useCallback((logits: number[]): number[] => {
    const maxLogit = Math.max(...logits);
    const exps = logits.map(x => Math.exp(x - maxLogit));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
  }, []);

  // Normalize landmarks
  const normalize = useCallback((landmarks: number[]): Float32Array => {
    const config = configRef.current;
    if (!config) return new Float32Array(63);

    const normalized = new Float32Array(63);
    for (let i = 0; i < 63; i++) {
      normalized[i] = (landmarks[i] - config.scaler.mean[i]) / config.scaler.scale[i];
    }
    return normalized;
  }, []);

  // Apply mode mask
  const applyModeMask = useCallback((probs: number[]): number[] => {
    const config = configRef.current;
    if (!config) return probs;

    const masked = [...probs];
    const dynamicClasses = new Set(config.dynamic_classes);

    if (currentMode === 'SPELLING') {
      config.class_names.forEach((name, idx) => {
        if (dynamicClasses.has(name)) masked[idx] = 0;
      });
    } else {
      config.class_names.forEach((name, idx) => {
        if (!dynamicClasses.has(name)) masked[idx] = 0;
      });
    }

    const sum = masked.reduce((a, b) => a + b, 0);
    if (sum > 0) {
      for (let i = 0; i < masked.length; i++) masked[i] /= sum;
    }
    return masked;
  }, [currentMode]);

  // Handle hand results
  const onHandResults = useCallback(async (results: HandResults) => {
    const startTime = performance.now();
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');

    if (!canvas || !ctx) return;

    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
      setHandDetected(false);
      setPrediction('-');
      setConfidence(0);
      setTopPredictions([]);
      ctx.restore();
      return;
    }

    setHandDetected(true);

    const landmarks = results.multiHandLandmarks[0];

    // Use global MediaPipe drawing functions
    if (window.drawConnectors && window.HAND_CONNECTIONS) {
      window.drawConnectors(ctx, landmarks, window.HAND_CONNECTIONS, { color: '#00d4ff', lineWidth: 2 });
    }
    if (window.drawLandmarks) {
      window.drawLandmarks(ctx, landmarks, { color: '#7b2cbf', lineWidth: 1, radius: 4 });
    }

    const coords: number[] = [];
    for (const lm of landmarks) {
      coords.push(lm.x, lm.y, lm.z);
    }

    const normalized = normalize(coords);
    const session = sessionRef.current;
    const config = configRef.current;

    if (session && config) {
      try {
        const tensor = new ort.Tensor('float32', normalized, [1, 63]);
        const outputs = await session.run({ landmarks: tensor });
        const logits = Array.from(outputs.logits.data as Float32Array);
        const probs = softmax(logits);
        const maskedProbs = applyModeMask(probs);

        const indexed = maskedProbs.map((p, i) => ({ prob: p, idx: i }));
        indexed.sort((a, b) => b.prob - a.prob);
        const top5 = indexed.slice(0, 5);

        const topPred = top5[0];
        const predClass = config.class_names[topPred.idx];
        const predConf = Math.round(topPred.prob * 100);

        setPrediction(predClass);
        setConfidence(predConf);
        setTopPredictions(top5.map(({ prob, idx }) => ({
          label: config.class_names[idx],
          probability: Math.round(prob * 100)
        })));

        // Draw on canvas
        ctx.fillStyle = '#00d4ff';
        ctx.font = 'bold 48px Inter, sans-serif';
        ctx.fillText(predClass, 20, 60);
        ctx.font = '24px Inter, sans-serif';
        ctx.fillText(`${predConf}%`, 20, 95);
      } catch (error) {
        console.error('Inference error:', error);
      }
    }

    ctx.restore();

    // Update stats
    const latencyMs = Math.round(performance.now() - startTime);
    setLatency(latencyMs);

    frameCountRef.current++;
    const now = performance.now();
    if (now - lastFpsTimeRef.current >= 1000) {
      setFps(frameCountRef.current);
      frameCountRef.current = 0;
      lastFpsTimeRef.current = now;
    }
  }, [normalize, softmax, applyModeMask]);

  // Keep ref updated with latest callback (fixes stale closure issue)
  onHandResultsRef.current = onHandResults;

  // Load model
  const loadModel = useCallback(async () => {
    try {
      setStatus('loading');
      setLoadingProgress(20);

      const configResponse = await fetch('/config.json');
      configRef.current = await configResponse.json();

      setLoadingProgress(60);

      sessionRef.current = await ort.InferenceSession.create('/model.onnx', {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });

      setLoadingProgress(100);
      setStatus('ready');
      setIsModelReady(true);
    } catch (error) {
      console.error('Failed to load model:', error);
      setStatus('error');
    }
  }, []);

  // Initialize MediaPipe
  const initMediaPipe = useCallback(async (video: HTMLVideoElement): Promise<MediaPipeHands | null> => {
    // Wait for MediaPipe to be available
    const waitForMediaPipe = (): Promise<void> => {
      return new Promise((resolve) => {
        if (window.Hands) {
          resolve();
        } else {
          const check = setInterval(() => {
            if (window.Hands) {
              clearInterval(check);
              resolve();
            }
          }, 100);
        }
      });
    };

    await waitForMediaPipe();
    console.log('MediaPipe Hands available, initializing...');

    try {
      const hands = new window.Hands({
        locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`
      });

      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

      // Use a wrapper that always calls the latest callback from ref
      // This prevents stale closure issues when mode changes
      hands.onResults((results: HandResults) => {
        if (onHandResultsRef.current) {
          onHandResultsRef.current(results);
        }
      });

      // Initialize by sending the actual video element
      // This triggers WASM/graph loading with proper WebGL context
      console.log('Sending first frame to initialize MediaPipe...');
      await hands.send({ image: video });
      console.log('MediaPipe Hands initialized successfully');

      return hands;
    } catch (error) {
      console.error('MediaPipe initialization error:', error);
      return null;
    }
  }, []);

  // Cleanup function for stopping camera properly
  const cleanupCamera = useCallback(() => {
    console.log('Cleaning up camera resources...');

    // Stop MediaPipe Camera
    if (cameraRef.current) {
      try {
        cameraRef.current.stop();
      } catch (e) {
        console.warn('Error stopping MediaPipe camera:', e);
      }
      cameraRef.current = null;
    }

    // Stop all media stream tracks (releases camera hardware)
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => {
        track.stop();
        console.log('Stopped track:', track.kind);
      });
      streamRef.current = null;
    }

    // Clear video source
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    // Clear canvas
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }

    // Reset MediaPipe Hands instance to force reinitialization
    // This prevents issues with stale WebGL contexts
    handsRef.current = null;

    console.log('Camera cleanup complete');
  }, []);

  // Start/Stop camera
  const toggleCamera = useCallback(async (video: HTMLVideoElement, canvas: HTMLCanvasElement) => {
    videoRef.current = video;
    canvasRef.current = canvas;

    if (isRunning) {
      cleanupCamera();
      setIsRunning(false);
      setStatus('ready');
      setHandDetected(false);
      setPrediction('-');
      setConfidence(0);
      setTopPredictions([]);
      setFps(0);
      setLatency(0);
      return;
    }

    try {
      // Ensure any previous stream is cleaned up first
      cleanupCamera();

      // Small delay to ensure camera hardware is released
      await new Promise(resolve => setTimeout(resolve, 100));

      console.log('Requesting camera access...');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' }
      });

      // Store stream reference for cleanup
      streamRef.current = stream;

      video.srcObject = stream;
      await video.play();

      // Wait for video to have valid dimensions
      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error('Video dimensions timeout'));
        }, 5000);

        const checkVideo = () => {
          if (video.videoWidth > 0 && video.videoHeight > 0) {
            clearTimeout(timeout);
            resolve();
          } else {
            requestAnimationFrame(checkVideo);
          }
        };
        checkVideo();
      });

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      console.log(`Video dimensions: ${video.videoWidth}x${video.videoHeight}`);

      // Always reinitialize MediaPipe for a fresh start
      console.log('Initializing MediaPipe Hands...');
      const hands = await initMediaPipe(video);
      if (!hands) {
        console.error('Failed to initialize MediaPipe');
        cleanupCamera();
        setStatus('error');
        return;
      }
      handsRef.current = hands;

      if (!window.Camera) {
        console.error('MediaPipe Camera not loaded');
        cleanupCamera();
        setStatus('error');
        return;
      }

      const camera = new window.Camera(video, {
        onFrame: async () => {
          if (handsRef.current && streamRef.current) {
            try {
              await handsRef.current.send({ image: video });
            } catch (e) {
              // Ignore errors when camera is being stopped
              if (streamRef.current) {
                console.warn('Frame processing error:', e);
              }
            }
          }
        },
        width: 640,
        height: 480
      });

      camera.start();
      cameraRef.current = camera;
      setIsRunning(true);
      setStatus('running');
      console.log('Camera started successfully');
    } catch (error) {
      console.error('Camera error:', error);
      cleanupCamera();
      setStatus('error');
    }
  }, [isRunning, initMediaPipe, cleanupCamera]);

  // Load model on mount
  useEffect(() => {
    loadModel();
  }, [loadModel]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      console.log('Component unmounting, cleaning up...');
      // Stop MediaPipe Camera
      if (cameraRef.current) {
        try {
          cameraRef.current.stop();
        } catch (e) {
          console.warn('Error stopping camera on unmount:', e);
        }
      }
      // Stop all media stream tracks
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return {
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
  };
}
