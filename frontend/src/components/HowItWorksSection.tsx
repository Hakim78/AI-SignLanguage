import { useState, useRef } from 'react';
import type { ReactNode } from 'react';

// Import step images
import step1Img from '../assets/step1.png';
import step2Img from '../assets/step2.png';
import step3Img from '../assets/step3.png';

// --- TYPES & DATA ---
interface Step {
  id: number;
  titleEn: string;
  titleFr: string;
  descriptionEn: string;
  descriptionFr: string;
  icon: ReactNode;
  color: string;      // Couleur principale (ex: texte)
  glowColor: string;  // Couleur de la lueur (rgba)
  image: string;      // Background image
}

const STEPS: Step[] = [
  {
    id: 1,
    titleEn: 'Position Hand',
    titleFr: 'Positionnez',
    descriptionEn: 'Place your hand 30-50cm from camera. Ensure good lighting for optimal detection.',
    descriptionFr: 'Placez votre main à 30-50cm. Assurez un bon éclairage pour une détection optimale.',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 11.5V14m0-2.5v-6a1.5 1.5 0 113 0m-3 6a1.5 1.5 0 00-3 0v2a7.5 7.5 0 0015 0v-5a1.5 1.5 0 00-3 0m-6-3V11m0-5.5v-1a1.5 1.5 0 013 0v1m0 0V11m0-5.5a1.5 1.5 0 013 0v3m0 0V11" />
      </svg>
    ),
    color: "#60a5fa", // Blue-400
    glowColor: "rgba(96, 165, 250, 0.4)",
    image: step1Img
  },
  {
    id: 2,
    titleEn: 'AI Detection',
    titleFr: 'Détection IA',
    descriptionEn: 'MediaPipe extracts 21 3D landmarks. Data is normalized locally in real-time.',
    descriptionFr: 'MediaPipe extrait 21 points 3D. Les données sont normalisées localement en temps réel.',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
      </svg>
    ),
    color: "#c084fc", // Purple-400
    glowColor: "rgba(192, 132, 252, 0.4)",
    image: step2Img
  },
  {
    id: 3,
    titleEn: 'S-TRM Inference',
    titleFr: 'Inférence S-TRM',
    descriptionEn: 'Our recursive model analyzes the sequence and predicts the sign instantly.',
    descriptionFr: 'Notre modèle récursif analyse la séquence et prédit le signe instantanément.',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
    color: "#f472b6", // Pink-400
    glowColor: "rgba(244, 114, 182, 0.4)",
    image: step3Img
  },
];

// --- COMPOSANTS INTERNES ---

// 1. Carte avec effet Spotlight (Lumiere qui suit la souris)
interface SpotlightCardProps {
  children: ReactNode;
  className?: string;
  borderColor?: string;
  backgroundImage?: string;
}

const SpotlightCard = ({ children, className = "", borderColor = "rgba(255,255,255,0.1)", backgroundImage }: SpotlightCardProps) => {
  const divRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [opacity, setOpacity] = useState(0);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!divRef.current) return;
    const rect = divRef.current.getBoundingClientRect();
    setPosition({ x: e.clientX - rect.left, y: e.clientY - rect.top });
  };

  return (
    <div
      ref={divRef}
      onMouseMove={handleMouseMove}
      onMouseEnter={() => setOpacity(1)}
      onMouseLeave={() => setOpacity(0)}
      className={`relative rounded-2xl border border-white/5 bg-black/40 overflow-hidden ${className}`}
    >
      {/* Background Image */}
      {backgroundImage && (
        <div
          className="absolute inset-0 bg-cover bg-center opacity-10 group-hover:opacity-20 transition-opacity duration-500"
          style={{ backgroundImage: `url(${backgroundImage})` }}
        />
      )}
      {/* Spotlight Gradient */}
      <div
        className="pointer-events-none absolute -inset-px transition duration-300"
        style={{
          opacity,
          background: `radial-gradient(600px circle at ${position.x}px ${position.y}px, ${borderColor}, transparent 40%)`,
        }}
      />
      <div className="relative h-full">{children}</div>
    </div>
  );
};

// 2. Carte d'Étape individuelle
const StepCard = ({ step, index, total, language }: { step: Step, index: number, total: number, language: 'en'|'fr' }) => {
  return (
    <div className="relative group h-full">
      {/* Connector Line (Desktop) */}
      {index < total - 1 && (
        <div className="hidden md:block absolute top-12 left-[calc(50%+4rem)] w-[calc(100%-8rem)] h-[2px] z-0">
          {/* Ligne pointillée statique */}
          <div className="absolute inset-0 border-t-2 border-dashed border-white/10" />
          {/* Ligne animée (Gradient Flow) */}
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent w-1/2 animate-[shimmer_2s_infinite]" />
        </div>
      )}

      <SpotlightCard className="p-8 h-full" borderColor={step.glowColor} backgroundImage={step.image}>
        <div className="flex flex-col items-center text-center">
          
          {/* ICONE AVEC GLOW */}
          <div className="relative mb-6 group-hover:scale-110 transition-transform duration-500">
            {/* Lueur d'arrière-plan */}
            <div 
              className="absolute inset-0 rounded-full blur-2xl opacity-20 group-hover:opacity-60 transition-opacity duration-500"
              style={{ backgroundColor: step.color }}
            />
            
            {/* Cercle Icône */}
            <div className="relative w-16 h-16 rounded-2xl bg-[#0a0a0c] border border-white/10 flex items-center justify-center shadow-2xl">
              <div 
                className="transition-colors duration-300"
                style={{ color: step.color }}
              >
                {step.icon}
              </div>
              
              {/* Badge Numéro */}
              <div className="absolute -top-3 -right-3 w-7 h-7 rounded-lg bg-[#1a1a1e] border border-white/10 flex items-center justify-center text-xs font-bold text-zinc-400 group-hover:text-white transition-colors">
                {step.id}
              </div>
            </div>
          </div>

          {/* TEXTE */}
          <h3 className="text-xl font-bold text-white mb-3 group-hover:text-transparent group-hover:bg-clip-text group-hover:bg-gradient-to-r group-hover:from-white group-hover:to-zinc-400 transition-all">
            {language === 'en' ? step.titleEn : step.titleFr}
          </h3>
          
          <p className="text-zinc-400 text-sm leading-relaxed">
            {language === 'en' ? step.descriptionEn : step.descriptionFr}
          </p>
        </div>
      </SpotlightCard>

      {/* Mobile Connector */}
      {index < total - 1 && (
        <div className="md:hidden flex justify-center py-6">
          <div className="w-[1px] h-12 border-l-2 border-dashed border-white/10" />
        </div>
      )}
    </div>
  );
};

// --- COMPOSANT PRINCIPAL ---
interface HowItWorksSectionProps {
  language: 'en' | 'fr';
}

export default function HowItWorksSection({ language }: HowItWorksSectionProps) {
  return (
    <section className="relative py-24 sm:py-32 px-4 sm:px-6 overflow-hidden bg-[#050505]">
      
      {/* 1. LIQUID BACKGROUND (Alternative légère à Vanta.js) */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-600/10 rounded-full blur-[100px] animate-pulse" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-violet-600/10 rounded-full blur-[100px] animate-pulse delay-1000" />
      </div>

      {/* Divider Top */}
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent opacity-50" />

      <div className="relative max-w-6xl mx-auto z-10">
        
        {/* HEADER */}
        <div className="text-center mb-20">
          <h2 className="text-4xl sm:text-5xl font-bold text-white mb-6 tracking-tight">
            {language === 'en' ? (
              <>How it <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-violet-400">works</span></>
            ) : (
              <>Comment ça <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-violet-400">marche</span></>
            )}
          </h2>
          <p className="text-zinc-400 text-lg max-w-2xl mx-auto">
            {language === 'en'
              ? 'Zero server latency. Privacy by design. Just allow camera access.'
              : 'Zéro latence serveur. Privé par défaut. Autorisez simplement la caméra.'}
          </p>
        </div>

        {/* STEPS GRID */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 md:gap-6">
          {STEPS.map((step, index) => (
            <StepCard
              key={step.id}
              step={step}
              index={index}
              total={STEPS.length}
              language={language}
            />
          ))}
        </div>

      </div>

      {/* Divider Bottom */}
      <div className="absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent opacity-50" />
    </section>
  );
}