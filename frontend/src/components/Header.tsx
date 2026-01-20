interface HeaderProps {
  lang: 'en' | 'fr';
  onToggleLang: () => void;
}

export function Header({ lang, onToggleLang }: HeaderProps) {
  return (
    <header className="fixed top-0 left-0 right-0 z-50">
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        <nav className="navbar flex items-center justify-between h-16 mt-4 px-4 sm:px-6 rounded-xl">
          {/* Logo */}
          <a href="#" className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-600 to-purple-600 flex items-center justify-center">
              <svg viewBox="0 0 24 24" className="w-4.5 h-4.5 text-white" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
            </div>
            <span className="font-semibold text-sm sm:text-base">S-TRM</span>
          </a>

          {/* Navigation */}
          <div className="hidden md:flex items-center gap-1">
            <a href="#demo" className="px-4 py-2 text-sm text-zinc-400 hover:text-white transition-colors">
              Demo
            </a>
            <a href="#how-it-works" className="px-4 py-2 text-sm text-zinc-400 hover:text-white transition-colors">
              {lang === 'en' ? 'How it works' : 'Fonctionnement'}
            </a>
            <a href="#about" className="px-4 py-2 text-sm text-zinc-400 hover:text-white transition-colors">
              {lang === 'en' ? 'About' : 'A propos'}
            </a>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-2">
            <button
              onClick={onToggleLang}
              className="px-3 py-1.5 text-sm text-zinc-400 hover:text-white transition-colors rounded-md hover:bg-white/5"
            >
              {lang === 'en' ? 'FR' : 'EN'}
            </button>
            <a href="#demo" className="btn-primary text-sm py-2 px-4">
              {lang === 'en' ? 'Try it' : 'Essayer'}
            </a>
          </div>
        </nav>
      </div>
    </header>
  );
}
