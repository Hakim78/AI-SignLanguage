// Import de l'image réelle générée
import confusionMatrixImg from '../assets/confusion_matrix_pro.png';

interface ConfusionMatrixPreviewProps {
  language: 'en' | 'fr';
}

export function ConfusionMatrixPreview({ language }: ConfusionMatrixPreviewProps) {
  return (
    <div className="card p-5 h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-zinc-200">
          {language === 'en' ? 'Confusion Matrix' : 'Matrice de Confusion'}
        </h3>
        <span className="text-xs px-2 py-1 rounded-full bg-violet-500/10 text-violet-400 border border-violet-500/20 font-mono">
          31 Classes
        </span>
      </div>

      {/* Container de l'image */}
      <div className="relative flex-1 bg-black/20 rounded-lg overflow-hidden border border-white/5 group">
        {/* L'image s'adapte au conteneur */}
        <div className="absolute inset-0 flex items-center justify-center p-2">
          <img 
            src={confusionMatrixImg} 
            alt="S-TRM Confusion Matrix" 
            className="max-w-full max-h-full object-contain transition-transform duration-500 group-hover:scale-105"
          />
        </div>
        
        {/* Overlay au survol pour montrer que c'est une vraie donnée */}
        <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex flex-col items-center justify-center pointer-events-none">
          <span className="text-white font-medium text-sm mb-1">
            {language === 'en' ? 'Normalized Performance' : 'Performance Normalisée'}
          </span>
          <span className="text-zinc-400 text-xs">
            {language === 'en' ? 'Generated from Test Set' : 'Générée depuis le Test Set'}
          </span>
        </div>
      </div>

      {/* Légende / Stats rapides en bas */}
      <div className="mt-4 pt-4 border-t border-white/5">
        <div className="flex justify-between items-center text-xs">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-indigo-500" />
            <span className="text-zinc-400">
              {language === 'en' ? 'High Accuracy' : 'Haute Précision'}
            </span>
          </div>
          <span className="text-zinc-500 font-mono">N=3,439 samples</span>
        </div>
        
        <p className="mt-3 text-xs text-zinc-500 leading-relaxed">
          {language === 'en' 
            ? "Strong diagonal indicates robust classification across ASL alphabet. Minor confusion clusters observed between M/N due to landmark similarity."
            : "La diagonale marquée indique une classification robuste. Des confusions mineures sont observées entre M/N dues à la similarité des points clés."}
        </p>
      </div>
    </div>
  );
}