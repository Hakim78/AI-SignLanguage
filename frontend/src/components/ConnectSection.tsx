import { useState, useRef } from 'react';
import { DotLottieReact } from '@lottiefiles/dotlottie-react';

// --- COMPOSANT SPOTLIGHT CARD (Réutilisable) ---
interface SpotlightCardProps {
  children: React.ReactNode;
  className?: string;
  spotlightColor?: string;
}

const SpotlightCard = ({ children, className = "", spotlightColor = "rgba(255,255,255,0.15)" }: SpotlightCardProps) => {
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
      className={`relative rounded-2xl border border-white/5 bg-zinc-900/40 overflow-hidden ${className}`}
    >
      <div
        className="pointer-events-none absolute -inset-px transition duration-300"
        style={{
          opacity,
          background: `radial-gradient(600px circle at ${position.x}px ${position.y}px, ${spotlightColor}, transparent 40%)`,
        }}
      />
      <div className="relative h-full">{children}</div>
    </div>
  );
};

interface ConnectSectionProps {
  language: 'en' | 'fr';
}

export function ConnectSection({ language }: ConnectSectionProps) {
  return (
    <section id="connect" className="py-24 sm:py-32 px-4 sm:px-6 relative overflow-hidden">
      {/* Background Glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-3xl h-full opacity-20 pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-indigo-500/10 rounded-full blur-[120px]" />
      </div>

      <div className="section-divider mb-16 opacity-50" />

      <div className="max-w-5xl mx-auto">

        {/* Header with Lottie */}
        <div className="text-center mb-16">
          <div className="w-24 h-24 mx-auto mb-6 relative">
            {/* Glow behind Lottie */}
            <div className="absolute inset-0 bg-blue-500/20 blur-2xl rounded-full scale-75" />
            <DotLottieReact
              src="https://lottie.host/f5b74eb2-0985-4c6c-8f67-6a1000b9bf0f/R6PHcntLFr.lottie"
              loop
              autoplay
            />
          </div>

          <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4 tracking-tight">
            {language === 'en' ? (
              <>Connect & <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-cyan-400">Inspect</span></>
            ) : (
              <>Connectez & <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-cyan-400">Inspectez</span></>
            )}
          </h2>

          <p className="text-zinc-400 text-lg max-w-xl mx-auto">
            {language === 'en'
              ? 'Review the implementation, read the technical report, or get in touch.'
              : 'Examinez l\'implémentation, lisez le rapport technique, ou contactez-nous.'}
          </p>
        </div>

        {/* Links Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">

          {/* GitHub */}
          <a href="https://github.com/Hakim78/AI-SignLanguage" target="_blank" rel="noopener noreferrer" className="group h-full">
            <SpotlightCard className="p-6 h-full transition-transform duration-300 group-hover:-translate-y-1" spotlightColor="rgba(255,255,255,0.1)">
              <div className="flex flex-col items-center text-center h-full">
                <div className="w-12 h-12 rounded-xl bg-zinc-900 border border-white/10 flex items-center justify-center mb-4 text-zinc-400 group-hover:text-white group-hover:border-white/20 transition-all duration-300 shadow-lg">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                    <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.17 6.839 9.49.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.604-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.464-1.11-1.464-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836c.85.004 1.705.114 2.504.336 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.167 22 16.418 22 12c0-5.523-4.477-10-10-10z" />
                  </svg>
                </div>
                <h3 className="font-semibold text-white mb-1">GitHub Repo</h3>
                <p className="text-xs text-zinc-500 group-hover:text-zinc-400 transition-colors">
                  {language === 'en' ? 'View Source Code' : 'Voir le Code Source'}
                </p>
              </div>
            </SpotlightCard>
          </a>

          {/* LinkedIn - Hakim */}
          <a href="https://www.linkedin.com/in/hakim-djaalal78000/" target="_blank" rel="noopener noreferrer" className="group h-full">
            <SpotlightCard className="p-6 h-full transition-transform duration-300 group-hover:-translate-y-1" spotlightColor="rgba(10, 102, 194, 0.2)">
              <div className="flex flex-col items-center text-center h-full">
                <div className="w-12 h-12 rounded-xl bg-zinc-900 border border-white/10 flex items-center justify-center mb-4 text-zinc-400 group-hover:text-[#0A66C2] group-hover:border-[#0A66C2]/30 transition-all duration-300 shadow-lg">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                  </svg>
                </div>
                <h3 className="font-semibold text-white mb-1">Hakim Djaalal</h3>
                <p className="text-xs text-zinc-500 group-hover:text-zinc-400 transition-colors">Student</p>
              </div>
            </SpotlightCard>
          </a>

          {/* LinkedIn - Mouad */}
          <a href="https://www.linkedin.com/in/mouad-aoughane-2943b6208/" target="_blank" rel="noopener noreferrer" className="group h-full">
            <SpotlightCard className="p-6 h-full transition-transform duration-300 group-hover:-translate-y-1" spotlightColor="rgba(10, 102, 194, 0.2)">
              <div className="flex flex-col items-center text-center h-full">
                <div className="w-12 h-12 rounded-xl bg-zinc-900 border border-white/10 flex items-center justify-center mb-4 text-zinc-400 group-hover:text-[#0A66C2] group-hover:border-[#0A66C2]/30 transition-all duration-300 shadow-lg">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                  </svg>
                </div>
                <h3 className="font-semibold text-white mb-1">Mouad Aoughane</h3>
                <p className="text-xs text-zinc-500 group-hover:text-zinc-400 transition-colors">Student</p>
              </div>
            </SpotlightCard>
          </a>

          {/* Technical Report */}
          <div className="group h-full cursor-not-allowed opacity-75">
            <SpotlightCard className="p-6 h-full transition-transform duration-300" spotlightColor="rgba(168, 85, 247, 0.15)">
              <div className="flex flex-col items-center text-center h-full">
                <div className="w-12 h-12 rounded-xl bg-zinc-900 border border-white/10 flex items-center justify-center mb-4 text-zinc-500 group-hover:text-purple-400 group-hover:border-purple-500/30 transition-all duration-300 shadow-lg">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                <h3 className="font-semibold text-white mb-1">Technical Report</h3>
                <div className="flex items-center gap-1.5 mt-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-yellow-500 animate-pulse" />
                  <p className="text-xs text-yellow-500/80">
                    {language === 'en' ? 'Coming soon' : 'Bientôt disponible'}
                  </p>
                </div>
              </div>
            </SpotlightCard>
          </div>

        </div>

        {/* Footer Info */}
        <div className="mt-16 pt-8 border-t border-white/5 text-center">
          <p className="text-xs text-zinc-500 font-mono tracking-wide uppercase mb-3">
            {language === 'en' ? 'Developed' : 'Développé'} {language === 'en' ? 'by' : 'par'}
          </p>
          <div className="flex items-center justify-center gap-6 text-sm text-zinc-300">
            <span>Hakim Djaalal</span>
            <span className="w-1 h-1 rounded-full bg-zinc-700" />
            <span>Mouad Aoughane</span>
          </div>
        </div>

      </div>
    </section>
  );
}
