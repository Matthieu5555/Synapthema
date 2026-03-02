# Synapthema — Current Deployment State

Last updated: 2026-02-18

This document describes how Synapthema courses are currently deployed and served on `matthieu-separt.site`. Read this before making any changes.

---

## What Was Done

### 1. Serving Directory

Course HTML files are served from:

```
/Users/ms/learningxp/courses/
├── index.html              ← Course picker page (lists all courses)
├── courses.json            ← Manifest of deployed course slugs
└── cfa-program2026l2v6/    ← First deployed course
    ├── index.html
    ├── chapter_01.html … chapter_05.html
    ├── review.html
    ├── mixed_review.html
    └── course_meta.json
```

This directory is **separate** from the Synapthema source/output. Files are copied here from `output/<slug>/html/` after generation.

### 2. Caddy Configuration

Caddy (homebrew service) serves the site. Config at `/opt/homebrew/etc/Caddyfile`:

```caddyfile
# Synapthema courses — plain HTTP for Pi proxy
:8080 {
    root * /Users/ms/learningxp/courses
    file_server
    encode gzip
    header Cache-Control "public, max-age=604800"
}

matthieu-separt.site {
    # ... existing IsoCrates routes ...

    # Synapthema courses — static files (also here for direct Caddy TLS access)
    handle_path /learn/* {
        root * /Users/ms/learningxp/courses
        file_server
        encode gzip
        header Cache-Control "public, max-age=604800"
    }

    # ... rest of config ...
}
```

Two listeners exist:
- **`:8080`** — plain HTTP, for the Pi's nginx to proxy to (production path)
- **`handle_path /learn/*`** inside the `matthieu-separt.site` block — for direct access if Caddy handles TLS

### 3. Pi Nginx (DONE)

The Pi (`100.127.229.52`) has a `/learn/` location block in its nginx config at `/etc/nginx/sites-available/chainlit`:

```nginx
# Synapthema courses → Qube via Tailscale
location /learn/ {
    proxy_pass http://100.71.230.23:8080/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_buffer_size 16k;
    proxy_buffers 8 32k;
    proxy_busy_buffers_size 64k;
    proxy_cache_valid 200 7d;
    add_header Cache-Control "public, max-age=604800, immutable";
}
```

To modify: `ssh ms@100.127.229.52`, edit the file, then `sudo nginx -t && sudo systemctl reload nginx`

### 4. Course Picker (index.html)

The root index page at `/Users/ms/learningxp/courses/index.html` is a static HTML page that:
- Reads `courses.json` for the list of deployed course slugs
- Fetches `<slug>/course_meta.json` for each course to get title, summary, journey
- Renders clickable cards that link to `<slug>/`

To add a new course to the picker, add its slug to `courses.json`.

### 5. Course Metadata

Each course has a `course_meta.json` in its directory. This is what the index page reads. You can edit it to change the displayed title/summary without re-rendering:

```json
{
  "course_title": "CFA Fixed Income",
  "course_summary": "...",
  "learner_journey": "Term Structure Foundations → ...",
  "subtitle": "Interactive Training Course — 5 Chapters"
}
```

---

## How to Deploy a New Course

### Step 1: Generate (if not already done)

```bash
cd /Users/ms/Documents/Synapthema
uv run main.py /path/to/book.pdf

# Or re-render HTML only (no LLM cost):
uv run main.py /path/to/book.pdf --render-only
```

### Step 2: Copy to serving directory

```bash
SLUG="your-course-slug"
mkdir -p /Users/ms/learningxp/courses/$SLUG
cp /Users/ms/Documents/Synapthema/output/$SLUG/html/* /Users/ms/learningxp/courses/$SLUG/
```

### Step 3: (Optional) Edit course_meta.json

```bash
# Change the displayed title if needed
vim /Users/ms/learningxp/courses/$SLUG/course_meta.json
```

### Step 4: Register in the manifest

Edit `/Users/ms/learningxp/courses/courses.json` and add the new slug:

```json
[
  { "slug": "cfa-program2026l2v6" },
  { "slug": "your-course-slug" }
]
```

### Step 5: Verify

```bash
# Local test
curl -s http://localhost:8080/$SLUG/index.html | head -5

# Course picker
curl -s http://localhost:8080/ | head -5

# Public (after Pi nginx is configured)
# https://matthieu-separt.site/learn/$SLUG/
```

No Caddy reload needed — it's static file serving.

---

## How to Update an Existing Course

```bash
SLUG="cfa-program2026l2v6"

# Re-render from existing JSON (free, no LLM calls)
cd /Users/ms/Documents/Synapthema
uv run main.py input/cfa-program2026L2V6.PDF --render-only

# Overwrite served files
cp /Users/ms/Documents/Synapthema/output/$SLUG/html/* /Users/ms/learningxp/courses/$SLUG/
```

---

## Network Architecture

```
Internet
  │
  DNS: matthieu-separt.site → Bbox Router (176.147.232.153)
  │    Port 443 forwarded to Pi
  │
Raspberry Pi (Tailscale: 100.127.229.52)
  │  nginx + Let's Encrypt SSL
  │  /learn/*     → Qube:8080  (Synapthema courses)  ← NEEDS SETUP
  │  /IsoCrates*  → Qube:3001  (IsoCrates frontend)
  │  /api/*       → Qube:8001  (IsoCrates backend)
  │
Qube — this machine (Tailscale: 100.71.230.23)
  ├── Caddy :8080  → /Users/ms/learningxp/courses/  (static files)
  ├── Next.js :3001  (IsoCrates frontend)
  ├── uvicorn :8001  (IsoCrates backend)
  └── PostgreSQL :5432
```

---

## Key File Locations

| What | Path |
|------|------|
| Caddy config | `/opt/homebrew/etc/Caddyfile` |
| Serving root | `/Users/ms/learningxp/courses/` |
| Course manifest | `/Users/ms/learningxp/courses/courses.json` |
| Course picker HTML | `/Users/ms/learningxp/courses/index.html` |
| Synapthema source | `/Users/ms/Documents/Synapthema/` |
| Synapthema output | `/Users/ms/Documents/Synapthema/output/<slug>/html/` |
| Pi nginx config | `/etc/nginx/sites-available/chainlit` (on Pi) |

---

## Troubleshooting

**502 on `/learn/`**: Check that Caddy is running on Qube (`brew services list | grep caddy`) and that port 8080 is serving (`curl http://localhost:8080/`).

**Course not showing in picker**: Check that the slug is in `courses.json` and that `<slug>/course_meta.json` exists.

**Caddy not running**: `brew services restart caddy`

**Changed Caddyfile**: `caddy validate --config /opt/homebrew/etc/Caddyfile && caddy reload --config /opt/homebrew/etc/Caddyfile`
