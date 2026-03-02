# LearningXP — Mobile Website Deployment

Operations guide for deploying generated courses as a mobile-friendly website on matthieu-separt.site.

---

## Architecture Overview

LearningXP generates **self-contained static HTML** (inline CSS/JS, no backend required at runtime). Deployment is therefore a static-file serving problem — no application server needed for the course viewer itself.

```
Generation (local, Qube):
  PDF → Python pipeline (LLM) → Static HTML/CSS/JS in output/<slug>/html/

Serving:
  Internet → Bbox Router :443 → Pi (nginx + SSL) → Qube (static files via nginx/caddy)

Runtime (client-side):
  Browser loads HTML → All interactivity is JS → LocalStorage for progress/FSRS state
```

### What Gets Deployed

```
output/<slug>/html/
├── index.html              # Landing page (course overview, knowledge map, mind map)
├── chapter_01.html         # Self-contained chapter (4-13 MB each with embedded images)
├── chapter_02.html
├── ...
├── review.html             # FSRS spaced repetition
├── mixed_review.html       # Cross-chapter practice
└── course_meta.json        # Editable metadata (title, summary)
```

Key characteristics:
- **Fully self-contained**: CSS/JS inlined, images base64-encoded
- **No server-side logic**: everything runs in the browser
- **LocalStorage**: progress, FSRS cards, theme, API keys persist per-browser
- **CDN dependencies**: KaTeX, Mermaid.js, vis-network loaded from jsdelivr

---

## Quick Start: Deploy a Course

### Prerequisites

- A generated course in `output/<slug>/html/`
- SSH access to Qube (`100.71.230.23` via Tailscale)
- SSH access to Pi (`100.127.229.52` via Tailscale) for nginx config

### 1. Generate the Course (if not already done)

```bash
# On Qube, in the learningxp_generator project
cd ~/Documents/1.LEARNING/learningxp_generator

# Ensure .env has OPENROUTER_KEY or OPENAI_KEY
cat .env

# Generate
uv run main.py your-book.pdf

# Or re-render only (no LLM cost)
uv run main.py your-book.pdf --render-only
```

### 2. Copy Output to Serving Directory

```bash
SLUG="your-book-slug"
SERVE_DIR="/opt/learningxp/courses"

# Create serving directory
sudo mkdir -p "$SERVE_DIR/$SLUG"

# Copy generated HTML
cp -r output/$SLUG/html/* "$SERVE_DIR/$SLUG/"

# Verify
ls -la "$SERVE_DIR/$SLUG/index.html"
```

### 3. Configure Nginx on Pi

SSH into the Pi and add a location block:

```bash
ssh ms@100.127.229.52
sudo nano /etc/nginx/sites-available/chainlit
```

Add inside the `server` block for `matthieu-separt.site`:

```nginx
# LearningXP courses → Qube via Tailscale
location /learn/ {
    proxy_pass http://100.71.230.23:8080/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # Large files (chapters can be 5-13 MB with embedded images)
    proxy_buffer_size 16k;
    proxy_buffers 8 32k;
    proxy_busy_buffers_size 64k;

    # Cache static course files aggressively
    proxy_cache_valid 200 7d;
    add_header Cache-Control "public, max-age=604800, immutable";
}
```

```bash
sudo nginx -t && sudo systemctl reload nginx
```

### 4. Serve Files from Qube

Option A — **nginx** (recommended for production):

```bash
# Install nginx on Qube if not present
brew install nginx

# Create config
cat > /opt/homebrew/etc/nginx/servers/learningxp.conf << 'EOF'
server {
    listen 8080;
    server_name localhost;
    root /opt/learningxp/courses;

    # Serve index.html for directory requests
    index index.html;

    # Enable gzip for the large HTML files
    gzip on;
    gzip_types text/html application/json;
    gzip_min_length 1000;

    # Each course is a subdirectory
    location / {
        try_files $uri $uri/ =404;
    }
}
EOF

# Start/reload
brew services start nginx
# or: nginx -s reload
```

Option B — **Caddy** (if already running on Qube):

```Caddyfile
:8080 {
    root * /opt/learningxp/courses
    file_server {
        precompressed gzip
    }
    encode gzip
}
```

```bash
caddy reload --config /opt/homebrew/etc/Caddyfile
```

### 5. Verify

```bash
# Direct from Qube
curl -o /dev/null -w "%{http_code} %{size_download}" http://localhost:8080/your-book-slug/index.html

# Through Pi
curl -o /dev/null -w "%{http_code}" https://matthieu-separt.site/learn/your-book-slug/

# Open in browser
open "https://matthieu-separt.site/learn/your-book-slug/"
```

---

## Course Index Page (Optional)

If you deploy multiple courses, add a simple course listing at `/learn/`:

```bash
cat > /opt/learningxp/courses/index.html << 'HTMLEOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Learning Courses</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               background: #0f172a; color: #f1f5f9; padding: 2rem; }
        h1 { margin-bottom: 1.5rem; }
        .courses { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; }
        .course-card { background: #1e293b; border: 1px solid #475569; border-radius: 12px;
                       padding: 1.25rem; text-decoration: none; color: inherit; transition: border-color 0.2s; }
        .course-card:hover { border-color: #3b82f6; }
        .course-card h2 { font-size: 1.1rem; margin-bottom: 0.5rem; }
        .course-card p { font-size: 0.9rem; color: #94a3b8; }
    </style>
</head>
<body>
    <h1>Learning Courses</h1>
    <div class="courses" id="courseList"></div>
    <script>
        // Auto-populate from course_meta.json in each subdirectory
        // Or manually list courses:
        const courses = [
            { slug: 'your-book-slug', title: 'Course Title', desc: 'Short description' },
        ];
        const list = document.getElementById('courseList');
        courses.forEach(c => {
            list.innerHTML += `<a class="course-card" href="${c.slug}/">
                <h2>${c.title}</h2><p>${c.desc}</p></a>`;
        });
    </script>
</body>
</html>
HTMLEOF
```

---

## Mobile Improvements Needed

The current templates have basic responsiveness (single `@media (max-width: 900px)` breakpoint) but need work for a good phone experience.

### Current State

| Aspect | Status | Notes |
|--------|--------|-------|
| Viewport meta tag | Done | `width=device-width, initial-scale=1.0` |
| Sidebar collapse | Done | Hamburger menu at < 900px |
| Flexbox layouts | Done | Content wraps on narrow screens |
| Dark/light theme | Done | Toggle in header |
| Phone breakpoint (480px) | Missing | No phone-specific adjustments |
| Tablet breakpoint (768px) | Missing | Shares desktop or mobile layout |
| Touch targets (44x44px) | Partial | Some buttons too small (chapter nav: 0.2rem padding) |
| `100vh` on mobile | Broken | Address bar causes viewport shift |
| Swipe navigation | Missing | No gesture support |
| PWA / Add to Home Screen | Missing | No manifest or service worker |
| Offline support | Missing | CDN deps (KaTeX, Mermaid) require internet |

### Priority Fixes (styles.css)

**1. Fix viewport height on mobile browsers:**
```css
/* Replace all 100vh with 100dvh (dynamic viewport height) */
body { height: 100dvh; }
.course-container { height: calc(100dvh - 41px); }
.sidebar { height: calc(100dvh - 41px); }
```

**2. Add phone breakpoint:**
```css
@media (max-width: 480px) {
    .slide-viewport { padding: 1rem 0.75rem; padding-top: 3rem; }
    .slide h1 { font-size: 1.25rem; }
    .slide h2 { font-size: 1.1rem; }
    .nav-btn { min-width: 60px; padding: 0.4rem 0.75rem; font-size: 0.8rem; }
    .chapter-nav-btn { padding: 0.4rem 0.6rem; font-size: 0.8rem; min-height: 44px; }
    .quiz-option { padding: 0.75rem; min-height: 44px; }
    .confidence-dot { width: 44px; height: 44px; }
    .review-btn { min-width: 60px; padding: 0.75rem; font-size: 0.85rem; min-height: 44px; }
    .course-header { font-size: 0.75rem; }
    .breadcrumb-current { max-width: 100px; }
}
```

**3. Enlarge touch targets (WCAG 2.5.5 — 44x44px minimum):**
```css
@media (pointer: coarse) {
    .chapter-nav-btn, .nav-btn, .quiz-option,
    .review-btn, .confidence-dot, .sidebar-toggle {
        min-height: 44px;
        min-width: 44px;
    }
}
```

**4. Handle Mermaid/diagram overflow:**
```css
.mermaid-container, .concept-map-container {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
}
```

### PWA Support (Future)

To enable "Add to Home Screen" on mobile, add to the generated output:

**manifest.json:**
```json
{
    "name": "Course Title",
    "short_name": "Course",
    "start_url": "index.html",
    "display": "standalone",
    "background_color": "#0f172a",
    "theme_color": "#3b82f6",
    "icons": [{ "src": "icon-192.png", "sizes": "192x192", "type": "image/png" }]
}
```

**Service worker** (cache KaTeX, Mermaid, vis-network for offline use):
```js
const CACHE_NAME = 'lxp-v1';
const PRECACHE = [
    './', './index.html',
    'https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.js',
    'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js',
    'https://cdn.jsdelivr.net/npm/vis-network@9/standalone/umd/vis-network.min.js',
];

self.addEventListener('install', e => e.waitUntil(
    caches.open(CACHE_NAME).then(c => c.addAll(PRECACHE))
));

self.addEventListener('fetch', e => e.respondWith(
    caches.match(e.request).then(r => r || fetch(e.request))
));
```

---

## Infrastructure Reference

Reuses the existing matthieu-separt.site infrastructure.

### Network Path

```
Internet
    │
DNS: matthieu-separt.site → 176.147.232.153
    │
Bbox Router (192.168.1.254)
    │ Port forwarding: 80 → Pi, 443 → Pi
    │
Raspberry Pi "ms-server" (Tailscale: 100.127.229.52)
    │ nginx + Let's Encrypt SSL (certbot)
    │ /learn/*  → Qube:8080  (LearningXP courses)
    │ /IsoCrates* → Qube:3001 (IsoCrates)
    │ /api/*     → Qube:8001  (IsoCrates API)
    │ /          → localhost:3000 (SilverBullet)
    │
Qube (macOS, Tailscale: 100.71.230.23)
    ├── nginx/caddy on :8080 → /opt/learningxp/courses/
    ├── IsoCrates frontend :3001
    ├── IsoCrates backend :8001
    └── PostgreSQL :5432
```

### Port Reference

| Port | Machine | Service | Purpose |
|------|---------|---------|---------|
| 80 | Pi | nginx | HTTP → HTTPS redirect + ACME |
| 443 | Pi | nginx | HTTPS reverse proxy |
| 8080 | Qube | nginx/caddy | LearningXP static file server |
| 3001 | Qube | Next.js | IsoCrates frontend |
| 8001 | Qube | uvicorn | IsoCrates backend |

### DNS (no changes needed)

| Type | Host | Value |
|------|------|-------|
| A | @ | 176.147.232.153 |
| CNAME | www | matthieu-separt.site |

---

## Deploying a New Course (Runbook)

Step-by-step checklist for adding a new course:

```bash
# 1. Generate (on Qube, in learningxp_generator project)
uv run main.py /path/to/book.pdf

# 2. Note the slug from output
SLUG="book-name-slug"
ls output/$SLUG/html/

# 3. Copy to serving directory
sudo cp -r output/$SLUG/html/ /opt/learningxp/courses/$SLUG/

# 4. Test locally
curl -I http://localhost:8080/$SLUG/index.html

# 5. Test through Pi
curl -I https://matthieu-separt.site/learn/$SLUG/

# 6. Test on phone
# Open https://matthieu-separt.site/learn/$SLUG/ on mobile browser
```

### Updating an Existing Course

```bash
SLUG="existing-slug"

# Re-render (no LLM cost — just re-generates HTML from existing JSON)
uv run main.py /path/to/book.pdf --render-only

# Overwrite served files
sudo rsync -av --delete output/$SLUG/html/ /opt/learningxp/courses/$SLUG/
```

---

## File Size Considerations

Generated chapters are large because images are base64-embedded:

| File | Typical Size | Notes |
|------|-------------|-------|
| `index.html` | 200-300 KB | Knowledge map, mind map |
| `chapter_XX.html` | 700 KB - 13 MB | Depends on embedded images |
| `review.html` | 130 KB | FSRS review engine |
| `mixed_review.html` | 130 KB | Cross-chapter practice |

For mobile on cellular networks, this can be slow. Mitigations:

1. **gzip compression** (configured in nginx/caddy above) — reduces ~60-70%
2. **Set `EMBED_IMAGES=false`** in `.env` before generating — images stay as separate files, HTML drops to ~100-300 KB per chapter. Requires serving the `json/` directory alongside `html/`.
3. **Cache headers** — `Cache-Control: public, max-age=604800, immutable` so chapters load once and stay cached.

---

## launchd Service (Optional)

If you want the LearningXP file server to auto-start on Qube boot:

```bash
cat > ~/Library/LaunchAgents/com.learningxp.fileserver.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.learningxp.fileserver</string>
    <key>ProgramArguments</key>
    <array>
        <string>/opt/homebrew/bin/nginx</string>
        <string>-g</string>
        <string>daemon off;</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/ms/Library/Logs/learningxp-server.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/ms/Library/Logs/learningxp-server.error.log</string>
</dict>
</plist>
EOF

launchctl load ~/Library/LaunchAgents/com.learningxp.fileserver.plist
```

---

## Troubleshooting

### Course not loading

```bash
# Check file server is running on Qube
curl -I http://localhost:8080/

# Check Pi can reach Qube
ssh ms@100.127.229.52 "curl -I http://100.71.230.23:8080/"

# Check nginx config on Pi
ssh ms@100.127.229.52 "sudo nginx -t"
```

### Blank page / JS errors on mobile

```bash
# Check CDN dependencies are reachable
curl -I https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.js
curl -I https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js

# If CDN is blocked, vendor the files locally (see "Vendoring CDN Dependencies" below)
```

### Large files loading slowly on mobile

```bash
# Verify gzip is working
curl -H "Accept-Encoding: gzip" -sI http://localhost:8080/slug/chapter_01.html | grep Content-Encoding
# Should show: Content-Encoding: gzip

# If not, check nginx gzip config
```

### Public IP changed (site unreachable from internet)

```bash
# Check current public IP
curl -s https://api.ipify.org

# Compare with DNS
dig +short matthieu-separt.site

# If different: update A record at domain registrar
# CG-NAT warning: if port forwarding stops working, call Bouygues for dedicated IPv4
```

### SSL certificate issues

```bash
ssh ms@100.127.229.52
sudo certbot renew --dry-run
sudo systemctl reload nginx
```

---

## Vendoring CDN Dependencies (Offline Support)

To make courses work without internet (important for mobile with spotty connectivity):

```bash
# Download dependencies locally
mkdir -p /opt/learningxp/vendor
cd /opt/learningxp/vendor

# KaTeX
curl -L -o katex.min.js "https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.js"
curl -L -o katex.min.css "https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.css"
curl -L -o auto-render.min.js "https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/contrib/auto-render.min.js"
# Also need katex fonts directory

# Mermaid
curl -L -o mermaid.min.js "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"

# vis-network
curl -L -o vis-network.min.js "https://cdn.jsdelivr.net/npm/vis-network@9/standalone/umd/vis-network.min.js"
```

Then update `src/rendering/templates/base.html` and `index.html` to reference `/learn/vendor/` instead of CDN URLs. This requires a code change in the HTML generator — a `--vendor-cdn` flag would be the cleanest approach.

---

## Future: Full Web App (Upload PDFs Online)

The current deployment serves **pre-generated** courses. If you eventually want users to upload PDFs and generate courses through the browser:

### What's Needed

| Component | Technology | Effort |
|-----------|-----------|--------|
| Upload API | FastAPI on Qube | ~1 week |
| Job queue | Celery + Redis (or just background threads) | ~1 week |
| Progress UI | WebSocket or polling | ~3 days |
| Auth | JWT (reuse IsoCrates pattern) | ~2 days |
| Cost control | Rate limiting, usage quotas | ~2 days |

### Architecture (Future)

```
Pi (nginx)
    /learn/*          → Qube:8080  (static courses)
    /learn/api/*      → Qube:8090  (FastAPI upload service)

Qube
    :8080  nginx      → /opt/learningxp/courses/
    :8090  FastAPI     → accepts PDF uploads, runs pipeline, serves ZIP or adds to courses/
```

### Challenges

- **LLM cost**: a full course generation costs $2-10 in API calls
- **Processing time**: 20-60 minutes per book depending on length
- **Memory**: PDF parsing + concurrent LLM calls needs ~2-4 GB
- **Security**: must validate uploaded PDFs (size limits, file type checks)

This is a much larger project and is **not needed** for simply serving your generated courses on mobile.
