interface HeaderProps {
  lang: 'en' | 'fr';
  onToggleLang: () => void;
  t: (key: string) => string;
}

export function Header({ lang, onToggleLang, t }: HeaderProps) {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 flex justify-center pt-4 px-4">
      <nav className="navbar-floating flex items-center gap-4 px-2 py-2">
        {/* Logo */}
        <div className="flex items-center gap-2 pl-4">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-pink-500 flex items-center justify-center">
            <svg viewBox="0 0 24 24" className="w-5 h-5 text-white" fill="currentColor">
              <path d="M12.89 1.45l8 4A2 2 0 0 1 22 7.24v9.53a2 2 0 0 1-1.11 1.79l-8 4a2 2 0 0 1-1.79 0l-8-4a2 2 0 0 1-1.1-1.8V7.24a2 2 0 0 1 1.11-1.79l8-4a2 2 0 0 1 1.78 0z"/>
            </svg>
          </div>
          <span className="font-bold text-lg">S-TRM</span>
        </div>

        {/* Nav Links */}
        <div className="hidden sm:flex items-center gap-1 bg-white/5 rounded-full px-1 py-1">
          <a href="#demo" className="px-4 py-2 rounded-full text-sm text-gray-300 hover:text-white hover:bg-white/10 transition-all">
            Demo
          </a>
          <a href="#features" className="px-4 py-2 rounded-full text-sm text-gray-300 hover:text-white hover:bg-white/10 transition-all">
            {lang === 'en' ? 'Features' : 'Fonctionnalités'}
          </a>
          <a href="#about" className="px-4 py-2 rounded-full text-sm text-gray-300 hover:text-white hover:bg-white/10 transition-all">
            {lang === 'en' ? 'About' : 'À propos'}
          </a>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2 pr-2">
          <button
            onClick={onToggleLang}
            className="px-3 py-2 rounded-full text-sm text-gray-400 hover:text-white hover:bg-white/10 transition-all"
          >
            {lang.toUpperCase()}
          </button>
          <a
            href="#demo"
            className="btn-primary text-sm"
          >
            {t('contact')}
            <svg viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M5 12h14M12 5l7 7-7 7"/>
            </svg>
          </a>
        </div>
      </nav>
    </header>
  );
}
