import { useRef, useState, useCallback, useEffect } from 'react';
import { Hands, HAND_CONNECTIONS } from '@mediapipe/hands';
import type { Results } from '@mediapipe/hands';
import { Camera } from '@mediapipe/camera_utils';
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';
import * as ort from 'onnxruntime-web';

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
  const cameraRef = useRef<Camera | null>(null);
  const handsRef = useRef<Hands | null>(null);
  const frameCountRef = useRef(0);
  const lastFpsTimeRef = useRef(performance.now());
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

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
  const onHandResults = useCallback(async (results: Results) => {
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
    drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: '#00d4ff', lineWidth: 2 });
    drawLandmarks(ctx, landmarks, { color: '#7b2cbf', lineWidth: 1, radius: 4 });

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

  // Load model
  const loadModel = useCallback(async () => {
    try {
      setStatus('loading');
      setLoadingProgress(20);

      const configResponse = await fetch('/config.json');
      configRef.current = await configResponse.json();

      setLoadingProgress(60);

      sessionRef.current = await ort.InferenceSession.create('/model.onnx');

      setLoadingProgress(100);
      setStatus('ready');
      setIsModelReady(true);
    } catch (error) {
      console.error('Failed to load model:', error);
      setStatus('error');
    }
  }, []);

  // Initialize MediaPipe
  const initMediaPipe = useCallback(() => {
    const hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    hands.onResults(onHandResults);
    handsRef.current = hands;
  }, [onHandResults]);

  // Start/Stop camera
  const toggleCamera = useCallback(async (video: HTMLVideoElement, canvas: HTMLCanvasElement) => {
    videoRef.current = video;
    canvasRef.current = canvas;

    if (isRunning) {
      cameraRef.current?.stop();
      setIsRunning(false);
      setStatus('ready');
      setHandDetected(false);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' }
      });

      video.srcObject = stream;
      await video.play();

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      if (!handsRef.current) {
        initMediaPipe();
      }

      const camera = new Camera(video, {
        onFrame: async () => {
          if (handsRef.current) {
            await handsRef.current.send({ image: video });
          }
        },
        width: 640,
        height: 480
      });

      camera.start();
      cameraRef.current = camera;
      setIsRunning(true);
      setStatus('running');
    } catch (error) {
      console.error('Camera error:', error);
      setStatus('error');
    }
  }, [isRunning, initMediaPipe]);

  // Load model on mount
  useEffect(() => {
    loadModel();
  }, [loadModel]);

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
