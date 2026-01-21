import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

interface HeroSectionProps {
  language: 'en' | 'fr';
}

export function HeroSection({ language }: HeroSectionProps) {
  const vantaRef = useRef<HTMLDivElement>(null);
  const [vantaEffect, setVantaEffect] = useState<ReturnType<typeof import('vanta/dist/vanta.fog.min').default> | null>(null);

  useEffect(() => {
    if (!vantaEffect && vantaRef.current) {
      import('vanta/dist/vanta.fog.min').then((FOG) => {
        setVantaEffect(
          FOG.default({
            el: vantaRef.current,
            THREE: THREE,
            mouseControls: false,
            touchControls: false,
            gyroControls: false,
            minHeight: 200.0,
            minWidth: 200.0,
            highlightColor: 0x4c1d95,   // Violet tres fonce
            midtoneColor: 0x312e81,     // Indigo fonce
            lowlightColor: 0x0f0f23,    // Bleu nuit
            baseColor: 0x050505,        // Noir de base (--bg-primary)
            blurFactor: 0.7,
            speed: 0.4,                 // Tres lent pour etre subtil
            zoom: 0.6
          })
        );
      });
    }
    return () => {
      if (vantaEffect) vantaEffect.destroy();
    };
  }, [vantaEffect]);

  return (
    <section id="hero" className="relative min-h-[90vh] flex flex-col items-center justify-center overflow-hidden bg-[#050505] pt-20 border-b border-white/5">

      {/* --- VANTA FOG BACKGROUND --- */}
      <div
        ref={vantaRef}
        className="absolute inset-0 w-full h-full opacity-40 pointer-events-none"
      />

      {/* --- BACKGROUND LAYERS --- */}
      <div className="absolute inset-0 w-full h-full pointer-events-none">
        {/* 1. Grille Mathematique */}
        <div className="absolute inset-0 bg-grid-white" />

        {/* 2. Masque Radial (Pour fondre la grille sur les bords) */}
        <div className="absolute inset-0 bg-[#050505] [mask-image:radial-gradient(ellipse_at_center,transparent_10%,black_80%)]" />

        {/* 3. Lumiere "Optique" Centrale */}
        <div className="absolute top-[-10%] left-1/2 -translate-x-1/2 w-[80vw] h-[50vh] spotlight rounded-full opacity-60" />
      </div>

      <div className="relative z-10 max-w-5xl mx-auto px-4 sm:px-6 text-center">

        {/* --- TITRE PRINCIPAL --- */}
        <h1 className="text-5xl sm:text-7xl md:text-8xl font-bold tracking-tight mb-8 text-white leading-[1.1]">
          {language === 'en' ? 'Bridging the' : 'Traduire le'}<br />
          {/* Texte avec effet metallique */}
          <span className="animate-shine font-extrabold tracking-tighter">
            {language === 'en' ? 'Silence Gap.' : 'Silence.'}
          </span>
        </h1>

        {/* --- DESCRIPTION TECHNIQUE --- */}
        <p className="text-lg sm:text-xl text-zinc-400 max-w-2xl mx-auto mb-12 font-light leading-relaxed antialiased">
          {language === 'en'
            ? 'Real-time Sign Language recognition powered by Edge AI. Privacy-first, zero-latency inference running directly in your browser.'
            : 'Reconnaissance de la langue des signes temps-reel par Edge AI. Confidentialite totale, zero latence, executee directement dans votre navigateur.'}
        </p>

        {/* --- BOUTONS D'ACTION --- */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-5 mb-24">
          <a
            href="#demo"
            className="group relative h-14 px-8 rounded-full bg-white text-black font-semibold text-sm tracking-wide flex items-center justify-center gap-3 hover:bg-zinc-200 transition-all shadow-[0_0_20px_rgba(255,255,255,0.1)]"
          >
            <span>{language === 'en' ? 'LAUNCH DEMO' : 'LANCER LA DEMO'}</span>
            <svg className="w-4 h-4 transition-transform group-hover:translate-x-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
            </svg>
          </a>

          <a
            href="#specs"
            className="h-14 px-8 rounded-full border border-zinc-800 bg-black/50 text-zinc-300 font-medium text-sm tracking-wide flex items-center justify-center hover:bg-zinc-900 hover:text-white hover:border-zinc-700 transition-all backdrop-blur-md"
          >
            {language === 'en' ? 'VIEW SPECS' : 'VOIR SPECS'}
          </a>
        </div>

        {/* --- BANDEAU DE DONNEES (DATA STRIP) --- */}
        <div className="w-full border-t border-white/10 bg-white/[0.01]">
          <div className="grid grid-cols-2 md:grid-cols-4 divide-x divide-white/10">
            {[
              { label: 'LATENCY', value: '< 15 ms' },
              { label: 'MODEL SIZE', value: '79 KB' },
              { label: 'ACCURACY', value: '94.2 %' },
              { label: 'PRIVACY', value: 'LOCAL' },
            ].map((stat, i) => (
              <div key={i} className="py-6 flex flex-col items-center justify-center group hover:bg-white/[0.02] transition-colors cursor-default">
                <span className="text-[10px] font-mono text-zinc-600 uppercase tracking-widest mb-1 group-hover:text-zinc-500 transition-colors">
                  {stat.label}
                </span>
                <span className="text-sm md:text-base font-mono text-zinc-300 font-medium group-hover:text-white transition-colors">
                  {stat.value}
                </span>
              </div>
            ))}
          </div>
        </div>

      </div>
    </section>
  );
}
