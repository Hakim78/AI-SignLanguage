import { useEffect, useRef, useCallback } from 'react';
import { track, ANALYTICS_DEBUG } from './index';

// Tracked section IDs
const TRACKED_SECTIONS = ['hero', 'demo', 'specs', 'connect'] as const;
type SectionId = (typeof TRACKED_SECTIONS)[number];

// Minimum duration (in seconds) before sending section_exit event
const MIN_DURATION_SECONDS = 2;

// Intersection threshold - section must be at least this % visible
const VISIBILITY_THRESHOLD = 0.5;

interface SectionState {
  section: SectionId | null;
  enterTime: number | null;
}

/**
 * Hook to track which section the user is viewing
 * Uses IntersectionObserver to detect the most visible section
 * Sends section_enter and section_exit events with duration
 */
export function useSectionAnalytics(): void {
  const stateRef = useRef<SectionState>({
    section: null,
    enterTime: null,
  });

  const visibilityMap = useRef<Map<SectionId, number>>(new Map());

  // Send section_exit event with duration
  const sendSectionExit = useCallback((section: SectionId, enterTime: number) => {
    const duration = Math.round((Date.now() - enterTime) / 1000);

    // Only send if duration exceeds minimum threshold
    if (duration >= MIN_DURATION_SECONDS) {
      track('section_exit', {
        section,
        duration_s: duration,
      });
    } else if (ANALYTICS_DEBUG) {
      console.log(`[Analytics] Skipped section_exit (duration ${duration}s < ${MIN_DURATION_SECONDS}s)`);
    }
  }, []);

  // Handle section change
  const handleSectionChange = useCallback((newSection: SectionId | null) => {
    const state = stateRef.current;

    // Skip if same section
    if (state.section === newSection) return;

    // Send exit event for previous section
    if (state.section && state.enterTime) {
      sendSectionExit(state.section, state.enterTime);
    }

    // Send enter event for new section
    if (newSection) {
      track('section_enter', { section: newSection });
    }

    // Update state
    stateRef.current = {
      section: newSection,
      enterTime: newSection ? Date.now() : null,
    };
  }, [sendSectionExit]);

  // Determine the most visible section
  const determineMostVisibleSection = useCallback((): SectionId | null => {
    let maxVisibility = 0;
    let mostVisibleSection: SectionId | null = null;

    visibilityMap.current.forEach((visibility, section) => {
      if (visibility > maxVisibility) {
        maxVisibility = visibility;
        mostVisibleSection = section;
      }
    });

    // Only return a section if it meets the visibility threshold
    return maxVisibility >= VISIBILITY_THRESHOLD ? mostVisibleSection : null;
  }, []);

  useEffect(() => {
    // Skip SSR
    if (typeof window === 'undefined') return;

    // Get all tracked section elements
    const sections = TRACKED_SECTIONS.map(id => document.getElementById(id)).filter(
      (el): el is HTMLElement => el !== null
    );

    if (sections.length === 0) {
      if (ANALYTICS_DEBUG) {
        console.warn('[Analytics] No tracked sections found. Expected IDs:', TRACKED_SECTIONS);
      }
      return;
    }

    if (ANALYTICS_DEBUG) {
      console.log('[Analytics] Tracking sections:', sections.map(s => s.id));
    }

    // Create intersection observer
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const sectionId = entry.target.id as SectionId;
          visibilityMap.current.set(sectionId, entry.intersectionRatio);
        });

        // Determine and handle the most visible section
        const mostVisible = determineMostVisibleSection();
        handleSectionChange(mostVisible);
      },
      {
        // Track visibility from 0% to 100%
        threshold: [0, 0.25, 0.5, 0.75, 1.0],
        // Consider root margin for header offset
        rootMargin: '-64px 0px 0px 0px',
      }
    );

    // Observe all sections
    sections.forEach((section) => observer.observe(section));

    // Cleanup: send final exit event and disconnect observer
    const handleBeforeUnload = () => {
      const state = stateRef.current;
      if (state.section && state.enterTime) {
        sendSectionExit(state.section, state.enterTime);
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      observer.disconnect();
      window.removeEventListener('beforeunload', handleBeforeUnload);

      // Send exit event on unmount
      const state = stateRef.current;
      if (state.section && state.enterTime) {
        sendSectionExit(state.section, state.enterTime);
      }
    };
  }, [determineMostVisibleSection, handleSectionChange, sendSectionExit]);
}

export { TRACKED_SECTIONS };
