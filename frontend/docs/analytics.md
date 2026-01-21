# Analytics Integration

This project uses privacy-friendly analytics to track anonymous visitor statistics and section engagement.

## Overview

- **Provider**: Plausible (default) or Umami
- **Privacy**: No cookies, no PII, GDPR compliant
- **Events**: Page views + custom section tracking

## Configuration

### 1. Choose Your Provider

#### Option A: Plausible (Recommended)

1. Create a free account at [plausible.io](https://plausible.io)
2. Add your domain (e.g., `asl-sign-recognition.vercel.app`)
3. Update `index.html`:

```html
<script
  defer
  data-domain="your-actual-domain.vercel.app"
  src="https://plausible.io/js/script.js"
></script>
```

#### Option B: Umami

1. Self-host Umami or use [Umami Cloud](https://umami.is)
2. Create a website and get your Website ID
3. Replace the script in `index.html`:

```html
<script
  defer
  src="https://your-umami-instance.com/script.js"
  data-website-id="YOUR_WEBSITE_UUID"
></script>
```

4. Update `.env`:

```env
VITE_ANALYTICS_PROVIDER=umami
```

### 2. Environment Variables

Copy `.env.example` to `.env` and configure:

```env
# Analytics provider: 'plausible' | 'umami' | 'none'
VITE_ANALYTICS_PROVIDER=plausible

# Enable debug mode (logs events to console)
VITE_ANALYTICS_DEBUG=false
```

## Tracked Sections

The following sections are tracked via IntersectionObserver:

| Section ID | Description |
|------------|-------------|
| `hero`     | Landing hero section |
| `demo`     | Live demo with video feed |
| `specs`    | Technical specifications |
| `connect`  | Contact & links section |

## Events

### Automatic Events

- **Page views**: Tracked automatically by Plausible/Umami

### Custom Events

| Event Name | Properties | Description |
|------------|-----------|-------------|
| `section_enter` | `{ section: string }` | User scrolls into a section |
| `section_exit` | `{ section: string, duration_s: number }` | User leaves a section (only if duration >= 2s) |

## Testing

### Local Development

1. Enable debug mode:

```env
VITE_ANALYTICS_DEBUG=true
```

2. Run dev server:

```bash
npm run dev
```

3. Open browser console and scroll through sections
4. You should see logs like:

```
[Analytics] section_enter { section: 'hero' }
[Analytics] section_exit { section: 'hero', duration_s: 5 }
[Analytics] section_enter { section: 'demo' }
```

### Production (Vercel)

1. Deploy to Vercel
2. Open Plausible/Umami dashboard
3. Navigate to your site and scroll through sections
4. Check the dashboard for:
   - Real-time visitors
   - Custom events under "Goals" (Plausible) or "Events" (Umami)

### Verify Events in Plausible

1. Go to your Plausible dashboard
2. Navigate to **Goals** > **Add Goal** > **Custom Event**
3. Add goals for `section_enter` and `section_exit`
4. View section engagement data in real-time

## Privacy Notice

Add this to your privacy policy or footer:

> "We use anonymous statistics to understand how visitors interact with our demo. No personal data is collected."

This is already included in the footer: "Anonymous stats"

## Troubleshooting

### Events not showing up

1. Check if the analytics script is loading (Network tab)
2. Ensure `data-domain` matches your exact domain
3. Enable `VITE_ANALYTICS_DEBUG=true` to verify events fire
4. Plausible has a ~5 min delay for non-real-time data

### Section tracking not working

1. Verify sections have correct IDs: `hero`, `demo`, `specs`, `connect`
2. Check browser console for `[Analytics]` logs
3. Ensure IntersectionObserver is supported (all modern browsers)

## Architecture

```
src/lib/analytics/
├── index.ts              # track() wrapper function
└── useSectionAnalytics.ts # React hook for section tracking
```

The `track()` function abstracts the analytics provider, making it easy to switch between Plausible and Umami without changing application code.
