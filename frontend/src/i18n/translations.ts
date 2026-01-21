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
    positionHand: "Position your hand here",
    confidence: "confidence",
    topPredictions: "Top Predictions",
    latency: "Latency (ms)",
    params: "Params",
    footer: "Developed for IPSSI MIA4 - 2025 | S-TRM: Stateful Tiny Recursive Model"
  },
  fr: {
    title: "IA Langue des Signes",
    subtitle: "Inférence temps-réel dans le navigateur via",
    architecture: "Architecture S-TRM",
    contact: "Contact",
    showInfo: "Mode d'emploi & Infos",
    howToUse: "Mode d'emploi",
    tips: [
      "Positionnez votre main à 30-50cm de la caméra",
      "Utilisez un bon éclairage (lumière naturelle)",
      "Gardez votre main entièrement visible",
      "Utilisez un arrière-plan uni et contrastant",
      "Maintenez les signes stables pour plus de précision"
    ],
    privacy: "Confidentialité",
    privacyPoints: [
      "Traitement 100% côté client",
      "Aucune vidéo envoyée à un serveur",
      "Aucune collecte ni stockage de données",
      "Accès caméra uniquement si activé"
    ],
    limitations: "Limitations connues",
    limitationPoints: [
      "J et Z nécessitent un mouvement (support limité)",
      "Détection d'une seule main",
      "Performances variables selon l'éclairage",
      "Pas un traducteur ASL complet"
    ],
    modelCard: "Fiche Modèle",
    specs: {
      architecture: "Architecture",
      parameters: "Paramètres",
      input: "Entrée",
      classes: "Classes",
      runtime: "Runtime",
      detection: "Détection Main"
    },
    spelling: "LETTRES",
    words: "MOTS",
    start: "Démarrer Caméra",
    stop: "Arrêter Caméra",
    loading: "Chargement du modèle...",
    loadingConfig: "Chargement de la configuration...",
    loadingOnnx: "Chargement du modèle ONNX...",
    ready: "Prêt ! Cliquez sur Démarrer",
    running: "En cours...",
    waitingCam: "En attente de la caméra",
    handDetected: "Main détectée",
    noHand: "Aucune main détectée",
    positionHand: "Positionnez votre main ici",
    confidence: "confiance",
    topPredictions: "Top Prédictions",
    latency: "Latence (ms)",
    params: "Params",
    footer: "Développé pour IPSSI MIA4 - 2025 | S-TRM: Stateful Tiny Recursive Model"
  }
} as const;

export type Language = keyof typeof translations;
export type TranslationKey = keyof typeof translations.en;
