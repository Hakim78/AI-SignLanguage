import { useState } from 'react';
import type { ReactNode } from 'react';
import step1Bg from '../assets/step1.png';
import step2Bg from '../assets/step2.png';
import step3Bg from '../assets/step3.png';

const STEP_BACKGROUNDS = [step1Bg, step2Bg, step3Bg];

interface Step {
  id: number;
  titleEn: string;
  titleFr: string;
  descriptionEn: string;
  descriptionFr: string;
  icon: ReactNode;
  hex: string;
  glowColor: string;
  borderColor: string;
}

const STEPS: Step[] = [
  {
    id: 1,
    titleEn: 'Position Hand',
    titleFr: 'Positionnez',
    descriptionEn: 'Place your hand 30-50cm from the camera with good lighting.',
    descriptionFr: 'Placez votre main a 30-50cm de la camera avec un bon eclairage.',
    icon: (
      <svg className="w-5 h-5 sm:w-6 sm:h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 11.5V14m0-2.5v-6a1.5 1.5 0 113 0m-3 6a1.5 1.5 0 00-3 0v2a7.5 7.5 0 0015 0v-5a1.5 1.5 0 00-3 0m-6-3V11m0-5.5v-1a1.5 1.5 0 013 0v1m0 0V11m0-5.5a1.5 1.5 0 013 0v3m0 0V11" />
      </svg>
    ),
    hex: "#60a5fa",
    glowColor: "rgba(96, 165, 250, 0.25)",
    borderColor: "rgb(59, 130, 246)"
  },
  {
    id: 2,
    titleEn: 'AI Detection',
    titleFr: 'Detection IA',
    descriptionEn: 'MediaPipe detects 21 hand landmarks in real-time with high precision.',
    descriptionFr: 'MediaPipe detecte 21 points de la main en temps reel avec haute precision.',
    icon: (
      <svg className="w-5 h-5 sm:w-6 sm:h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    ),
    hex: "#a78bfa",
    glowColor: "rgba(167, 139, 250, 0.25)",
    borderColor: "rgb(139, 92, 246)"
  },
  {
    id: 3,
    titleEn: 'Recognition',
    titleFr: 'Reconnaissance',
    descriptionEn: 'S-TRM neural network predicts ASL signs with 79K parameters.',
    descriptionFr: 'Le reseau S-TRM predit les signes ASL avec 79K parametres.',
    icon: (
      <svg className="w-5 h-5 sm:w-6 sm:h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
    hex: "#f472b6",
    glowColor: "rgba(244, 114, 182, 0.25)",
    borderColor: "rgb(236, 72, 153)"
  },
];

// Animated arrow between steps (desktop)
const AnimatedArrow = ({ flip }: { flip?: boolean }) => (
  <div className={`hidden md:block absolute top-16 -right-[50%] w-full h-24 pointer-events-none z-0 ${flip ? 'transform scale-y-[-1] top-20' : ''}`}>
    <svg width="100%" height="100%" viewBox="0 0 200 60" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path
        d="M20 40 C 70 10, 130 10, 180 40"
        stroke="url(#gradient-flow)"
        strokeWidth="3"
        strokeLinecap="round"
        strokeDasharray="10 10"
        className="animate-dash"
      />
      <path d="M175 35 L180 40 L175 45" stroke="rgba(255,255,255,0.3)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
      <defs>
        <linearGradient id="gradient-flow" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="rgba(255,255,255,0.05)" />
          <stop offset="50%" stopColor="rgba(255,255,255,0.3)" />
          <stop offset="100%" stopColor="rgba(255,255,255,0.05)" />
        </linearGradient>
      </defs>
    </svg>
  </div>
);

// Mobile arrow (vertical)
const MobileArrow = ({ color }: { color: string }) => (
  <div className="md:hidden flex justify-center py-6">
    <div className="flex flex-col items-center gap-1">
      <div className="w-0.5 h-8 bg-gradient-to-b from-transparent via-white/20 to-white/10 rounded-full" />
      <svg className="w-5 h-5 animate-bounce" style={{ color }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
      </svg>
    </div>
  </div>
);

interface StepItemProps {
  step: Step;
  index: number;
  total: number;
  hoveredIndex: number | null;
  setHoveredIndex: (index: number | null) => void;
  language: 'en' | 'fr';
  backgroundImage: string;
}

const StepItem = ({ step, index, total, hoveredIndex, setHoveredIndex, language, backgroundImage }: StepItemProps) => {
  const isHovered = hoveredIndex === index;
  const isDimmed = hoveredIndex !== null && hoveredIndex !== index;

  return (
    <>
      <div
        className={`relative flex flex-col items-center text-center group transition-all duration-500 ${isDimmed ? 'opacity-40 blur-[1px]' : 'opacity-100 blur-0'}`}
        onMouseEnter={() => setHoveredIndex(index)}
        onMouseLeave={() => setHoveredIndex(null)}
      >
        {/* Background image */}
        <div
          className="absolute inset-0 rounded-3xl opacity-10 bg-cover bg-center transition-opacity duration-500 pointer-events-none"
          style={{
            backgroundImage: `url(${backgroundImage})`,
            opacity: isHovered ? 0.2 : 0.08
          }}
        />

        {/* Desktop arrow */}
        {index < total - 1 && <AnimatedArrow flip={index % 2 !== 0} />}

        {/* Big 3D number */}
        <div className="relative mb-6 sm:mb-10">
          {/* Glow behind */}
          <div
            className="absolute inset-0 blur-3xl rounded-full transition-all duration-500"
            style={{
              backgroundColor: step.glowColor,
              transform: isHovered ? 'scale(1.5)' : 'scale(1)',
              opacity: isHovered ? 1 : 0
            }}
          />

          <div
            className="relative z-10 transition-all duration-300 ease-out"
            style={{
              transform: isHovered ? 'translateY(-10px) scale(1.05)' : 'translateY(0) scale(1)',
            }}
          >
            <span
              className="text-7xl sm:text-8xl md:text-9xl font-black text-white select-none block"
              style={{
                textShadow: isHovered
                  ? `8px 8px 0px ${step.hex}, 16px 16px 20px rgba(0,0,0,0.5)`
                  : `4px 4px 0px ${step.hex}, 8px 8px 0px rgba(0,0,0,0.5)`,
                WebkitTextStroke: '1px rgba(255,255,255,0.1)',
                transition: 'text-shadow 0.3s ease-out'
              }}
            >
              {step.id}
            </span>

            {/* Floating icon badge */}
            <div
              className="absolute -bottom-0 -right-2 sm:-right-4 bg-zinc-900 border p-2 sm:p-3 rounded-xl sm:rounded-2xl shadow-xl flex items-center justify-center transition-all duration-300"
              style={{
                borderColor: isHovered ? step.borderColor : 'rgb(63, 63, 70)',
                transform: isHovered ? 'rotate(12deg) scale(1.1)' : 'rotate(0) scale(1)'
              }}
            >
              <div style={{ color: isHovered ? step.hex : 'rgb(161, 161, 170)' }} className="transition-colors duration-300">
                {step.icon}
              </div>
            </div>
          </div>
        </div>

        {/* Text */}
        <div className="relative z-10 max-w-xs sm:max-w-sm px-2">
          <h3
            className="text-xl sm:text-2xl font-bold text-white mb-3 sm:mb-4 transition-colors duration-300"
            style={{ color: isHovered ? step.hex : 'white' }}
          >
            {language === 'en' ? step.titleEn : step.titleFr}
          </h3>
          <p className="text-zinc-400 leading-relaxed text-base sm:text-lg">
            {language === 'en' ? step.descriptionEn : step.descriptionFr}
          </p>
        </div>
      </div>

      {/* Mobile arrow */}
      {index < total - 1 && <MobileArrow color={step.hex} />}
    </>
  );
};

interface HowItWorksSectionProps {
  language: 'en' | 'fr';
}

export default function HowItWorksSection({ language }: HowItWorksSectionProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  return (
    <section className="relative py-16 sm:py-24 md:py-32 px-4 overflow-hidden bg-zinc-950 min-h-[600px] md:min-h-[700px] flex items-center">
      {/* Gradient overlays */}
      <div className="absolute inset-0 bg-gradient-to-b from-zinc-900 via-transparent to-zinc-900 pointer-events-none" />
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_20%,rgba(139,92,246,0.08),transparent_50%)]" />
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_70%_80%,rgba(96,165,250,0.06),transparent_50%)]" />

      <div className="max-w-7xl mx-auto w-full relative z-10">
        {/* Header */}
        <div className="text-center mb-12 sm:mb-20 md:mb-28">
          <h2 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold text-white mb-4 sm:mb-6 tracking-tight">
            {language === 'en' ? (
              <>How it <span className="text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-purple-500">works?</span></>
            ) : (
              <>Comment ca <span className="text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-purple-500">marche ?</span></>
            )}
          </h2>
          <p className="text-zinc-400 text-base sm:text-lg md:text-xl max-w-2xl mx-auto px-4">
            {language === 'en'
              ? 'No signup required. No server upload. 100% privacy-first.'
              : 'Pas d\'inscription. Aucun envoi serveur. 100% prive.'}
          </p>
        </div>

        {/* Steps Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-8 lg:gap-12">
          {STEPS.map((step, index) => (
            <StepItem
              key={step.id}
              step={step}
              index={index}
              total={STEPS.length}
              hoveredIndex={hoveredIndex}
              setHoveredIndex={setHoveredIndex}
              language={language}
              backgroundImage={STEP_BACKGROUNDS[index]}
            />
          ))}
        </div>
      </div>
    </section>
  );
}
