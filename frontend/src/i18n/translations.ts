export const translations = {
  en: {
    title: "Sign Language AI",
    subtitle: "Real-time browser inference via",
    architecture: "S-TRM Architecture",
    contact: "Contact",
    showInfo: "How to use & Info",
    howToUse: "How to Use",
    tips: [
      "Position your hand 30-50cm from camera",
      "Use good lighting (natural light preferred)",
      "Keep your hand fully visible in frame",
      "Use a plain, contrasting background",
      "Hold signs steady for best accuracy"
    ],
    privacy: "Privacy",
    privacyPoints: [
      "100% client-side processing",
      "No video sent to any server",
      "No data collection or storage",
      "Camera access only when enabled"
    ],
    limitations: "Known Limitations",
    limitationPoints: [
      "J and Z require motion (limited support)",
      "Single hand detection only",
      "Performance varies with lighting",
      "Not a full ASL translator"
    ],
    modelCard: "Model Card",
    specs: {
      architecture: "Architecture",
      parameters: "Parameters",
      input: "Input",
      classes: "Classes",
      runtime: "Runtime",
      detection: "Hand Detection"
    },
    spelling: "SPELLING",
    words: "WORDS",
    start: "Start Camera",
    stop: "Stop Camera",
    loading: "Loading model...",
    loadingConfig: "Loading configuration...",
    loadingOnnx: "Loading ONNX model...",
    ready: "Ready! Click Start Camera",
    running: "Running...",
    waitingCam: "Waiting for camera",
    handDetected: "Hand detected",
    noHand: "No hand detected",
    confidence: "confidence",
    topPredictions: "Top Predictions",
    latency: "Latency (ms)",
    params: "Params",
    footer: "Developed for IPSSI MIA4 - 2025 | S-TRM: Stateful Tiny Recursive Model"
  },
  fr: {
    title: "IA Langue des Signes",
    subtitle: "Inference temps-reel dans le navigateur via",
    architecture: "Architecture S-TRM",
    contact: "Contact",
    showInfo: "Mode d'emploi & Infos",
    howToUse: "Mode d'emploi",
    tips: [
      "Positionnez votre main a 30-50cm de la camera",
      "Utilisez un bon eclairage (lumiere naturelle)",
      "Gardez votre main entierement visible",
      "Utilisez un arriere-plan uni et contrastant",
      "Maintenez les signes stables pour plus de precision"
    ],
    privacy: "Confidentialite",
    privacyPoints: [
      "Traitement 100% cote client",
      "Aucune video envoyee a un serveur",
      "Aucune collecte ni stockage de donnees",
      "Acces camera uniquement si active"
    ],
    limitations: "Limitations connues",
    limitationPoints: [
      "J et Z necessitent un mouvement (support limite)",
      "Detection d'une seule main",
      "Performances variables selon l'eclairage",
      "Pas un traducteur ASL complet"
    ],
    modelCard: "Fiche Modele",
    specs: {
      architecture: "Architecture",
      parameters: "Parametres",
      input: "Entree",
      classes: "Classes",
      runtime: "Runtime",
      detection: "Detection Main"
    },
    spelling: "LETTRES",
    words: "MOTS",
    start: "Demarrer Camera",
    stop: "Arreter Camera",
    loading: "Chargement du modele...",
    loadingConfig: "Chargement de la configuration...",
    loadingOnnx: "Chargement du modele ONNX...",
    ready: "Pret ! Cliquez sur Demarrer",
    running: "En cours...",
    waitingCam: "En attente de la camera",
    handDetected: "Main detectee",
    noHand: "Aucune main detectee",
    confidence: "confiance",
    topPredictions: "Top Predictions",
    latency: "Latence (ms)",
    params: "Params",
    footer: "Developpe pour IPSSI MIA4 - 2025 | S-TRM: Stateful Tiny Recursive Model"
  }
} as const;

export type Language = keyof typeof translations;
export type TranslationKey = keyof typeof translations.en;
