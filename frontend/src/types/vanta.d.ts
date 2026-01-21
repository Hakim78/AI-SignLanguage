declare module 'vanta/dist/vanta.fog.min' {
  interface VantaFogOptions {
    el: HTMLElement | null;
    THREE?: unknown;
    mouseControls?: boolean;
    touchControls?: boolean;
    gyroControls?: boolean;
    minHeight?: number;
    minWidth?: number;
    highlightColor?: number;
    midtoneColor?: number;
    lowlightColor?: number;
    baseColor?: number;
    blurFactor?: number;
    speed?: number;
    zoom?: number;
  }

  interface VantaEffect {
    destroy: () => void;
    setOptions: (options: Partial<VantaFogOptions>) => void;
    resize: () => void;
  }

  export default function FOG(options: VantaFogOptions): VantaEffect;
}
