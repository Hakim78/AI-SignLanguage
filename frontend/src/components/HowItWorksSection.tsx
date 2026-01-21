import { useState } from 'react';
import type { ReactNode } from 'react';

interface Step {
  id: number;
  titleEn: string;
  titleFr: string;
  descriptionEn: string;
  descriptionFr: string;
  icon: ReactNode;
  accentColor: string;
}

const STEPS: Step[] = [
  {
    id: 1,
    titleEn: 'Position Hand',
    titleFr: 'Positionnez',
    descriptionEn: 'Place your hand 30-50cm from the camera with good lighting.',
    descriptionFr: 'Placez votre main a 30-50cm de la camera avec un bon eclairage.',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 11.5V14m0-2.5v-6a1.5 1.5 0 113 0m-3 6a1.5 1.5 0 00-3 0v2a7.5 7.5 0 0015 0v-5a1.5 1.5 0 00-3 0m-6-3V11m0-5.5v-1a1.5 1.5 0 013 0v1m0 0V11m0-5.5a1.5 1.5 0 013 0v3m0 0V11" />
      </svg>
    ),
    accentColor: "#60a5fa"
  },
  {
    id: 2,
    titleEn: 'AI Detection',
    titleFr: 'Detection IA',
    descriptionEn: 'MediaPipe detects 21 hand landmarks in real-time.',
    descriptionFr: 'MediaPipe detecte 21 points de la main en temps reel.',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    ),
    accentColor: "#a78bfa"
  },
  {
    id: 3,
    titleEn: 'Recognition',
    titleFr: 'Reconnaissance',
    descriptionEn: 'S-TRM neural network predicts ASL signs instantly.',
    descriptionFr: 'Le reseau S-TRM predit les signes ASL instantanement.',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
    accentColor: "#f472b6"
  },
];

interface StepCardProps {
  step: Step;
  index: number;
  total: number;
  isHovered: boolean;
  onHover: (index: number | null) => void;
  language: 'en' | 'fr';
}

const StepCard = ({ step, index, total, isHovered, onHover, language }: StepCardProps) => {
  return (
    <div className="relative">
      {/* Connector line - desktop only */}
      {index < total - 1 && (
        <div className="hidden md:block absolute top-10 left-[calc(50%+3rem)] w-[calc(100%-3rem)] h-px">
          <div className="w-full h-full bg-gradient-to-r from-white/10 via-white/5 to-transparent" />
        </div>
      )}

      <div
        className="relative group cursor-pointer"
        onMouseEnter={() => onHover(index)}
        onMouseLeave={() => onHover(null)}
      >
        {/* Card */}
        <div
          className={`relative p-6 sm:p-8 rounded-2xl border transition-all duration-300 ${
            isHovered
              ? 'bg-white/[0.04] border-white/10 -translate-y-1'
              : 'bg-white/[0.02] border-white/[0.04]'
          }`}
        >
          {/* Step number & icon row */}
          <div className="flex items-center gap-4 mb-5">
            {/* Number badge */}
            <div
              className="w-12 h-12 rounded-xl flex items-center justify-center text-lg font-bold transition-all duration-300"
              style={{
                backgroundColor: isHovered ? `${step.accentColor}20` : 'rgba(255,255,255,0.03)',
                color: isHovered ? step.accentColor : '#71717a',
                border: `1px solid ${isHovered ? `${step.accentColor}40` : 'rgba(255,255,255,0.06)'}`
              }}
            >
              {step.id}
            </div>

            {/* Icon */}
            <div
              className="transition-colors duration-300"
              style={{ color: isHovered ? step.accentColor : '#71717a' }}
            >
              {step.icon}
            </div>
          </div>

          {/* Title */}
          <h3
            className="text-lg sm:text-xl font-semibold mb-2 transition-colors duration-300"
            style={{ color: isHovered ? '#fafafa' : '#d4d4d8' }}
          >
            {language === 'en' ? step.titleEn : step.titleFr}
          </h3>

          {/* Description */}
          <p className="text-zinc-500 text-sm sm:text-base leading-relaxed">
            {language === 'en' ? step.descriptionEn : step.descriptionFr}
          </p>

          {/* Subtle glow on hover */}
          {isHovered && (
            <div
              className="absolute inset-0 rounded-2xl opacity-20 blur-xl -z-10 transition-opacity duration-300"
              style={{ backgroundColor: step.accentColor }}
            />
          )}
        </div>
      </div>

      {/* Mobile connector */}
      {index < total - 1 && (
        <div className="md:hidden flex justify-center py-4">
          <div className="w-px h-8 bg-gradient-to-b from-white/10 to-transparent" />
        </div>
      )}
    </div>
  );
};

interface HowItWorksSectionProps {
  language: 'en' | 'fr';
}

export default function HowItWorksSection({ language }: HowItWorksSectionProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  return (
    <section className="relative py-16 sm:py-20 lg:py-24 px-4 sm:px-6">
      {/* Subtle section divider */}
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/[0.06] to-transparent" />

      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12 sm:mb-16">
          <h2 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-white mb-3 sm:mb-4">
            {language === 'en' ? (
              <>How it <span className="text-gradient">works</span></>
            ) : (
              <>Comment ca <span className="text-gradient">marche</span></>
            )}
          </h2>
          <p className="text-zinc-500 text-sm sm:text-base max-w-lg mx-auto">
            {language === 'en'
              ? 'No signup. No server. 100% private.'
              : 'Pas d\'inscription. Pas de serveur. 100% prive.'}
          </p>
        </div>

        {/* Steps Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 sm:gap-6">
          {STEPS.map((step, index) => (
            <StepCard
              key={step.id}
              step={step}
              index={index}
              total={STEPS.length}
              isHovered={hoveredIndex === index}
              onHover={setHoveredIndex}
              language={language}
            />
          ))}
        </div>
      </div>

      {/* Subtle section divider */}
      <div className="absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-white/[0.06] to-transparent" />
    </section>
  );
}
