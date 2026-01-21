/**
 * Analytics wrapper for Plausible/Umami
 * Supports easy switching between providers via VITE_ANALYTICS_PROVIDER env
 */

type AnalyticsProvider = 'plausible' | 'umami' | 'none';

interface TrackEventProps {
  [key: string]: string | number | boolean;
}

interface PlausibleWindow extends Window {
  plausible?: (eventName: string, options?: { props?: TrackEventProps }) => void;
}

interface UmamiWindow extends Window {
  umami?: {
    track: (eventName: string, props?: TrackEventProps) => void;
  };
}

// Configuration
const ANALYTICS_PROVIDER: AnalyticsProvider =
  (import.meta.env.VITE_ANALYTICS_PROVIDER as AnalyticsProvider) || 'plausible';
const ANALYTICS_DEBUG = import.meta.env.VITE_ANALYTICS_DEBUG === 'true';

/**
 * Track a custom event
 * @param eventName - Name of the event (e.g., 'section_enter', 'section_exit')
 * @param props - Event properties (e.g., { section: 'hero', duration_s: 15 })
 */
export function track(eventName: string, props?: TrackEventProps): void {
  // Skip if SSR
  if (typeof window === 'undefined') return;

  // Debug logging
  if (ANALYTICS_DEBUG) {
    console.log(`[Analytics] ${eventName}`, props);
  }

  // Skip tracking in development unless debug is enabled
  if (import.meta.env.DEV && !ANALYTICS_DEBUG) {
    return;
  }

  try {
    switch (ANALYTICS_PROVIDER) {
      case 'plausible': {
        const win = window as PlausibleWindow;
        if (win.plausible) {
          win.plausible(eventName, { props });
        }
        break;
      }
      case 'umami': {
        const win = window as UmamiWindow;
        if (win.umami?.track) {
          win.umami.track(eventName, props);
        }
        break;
      }
      case 'none':
      default:
        // No-op
        break;
    }
  } catch (error) {
    if (ANALYTICS_DEBUG) {
      console.error('[Analytics] Error tracking event:', error);
    }
  }
}

/**
 * Track page view (called automatically by analytics scripts, but can be used for SPAs)
 */
export function trackPageView(url?: string): void {
  if (typeof window === 'undefined') return;

  if (ANALYTICS_DEBUG) {
    console.log('[Analytics] Page view:', url || window.location.pathname);
  }

  // Plausible and Umami auto-track page views,
  // but for SPA navigation you might need manual tracking
  // This is handled automatically by the respective scripts
}

/**
 * Check if analytics is loaded and ready
 */
export function isAnalyticsReady(): boolean {
  if (typeof window === 'undefined') return false;

  switch (ANALYTICS_PROVIDER) {
    case 'plausible':
      return !!(window as PlausibleWindow).plausible;
    case 'umami':
      return !!(window as UmamiWindow).umami?.track;
    default:
      return false;
  }
}

export { ANALYTICS_DEBUG, ANALYTICS_PROVIDER };
