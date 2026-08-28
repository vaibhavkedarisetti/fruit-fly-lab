# Deploying

The site is the `web/` directory. It is a plain static site: no build step, no
server, no serverless functions. The whole simulation runs in the visitor's
browser in a Web Worker.

## The one setting that matters

**Vercel → Project → Settings → Build and Deployment → Root Directory → `web`**

Then redeploy (Deployments → ⋯ → Redeploy).

If the Root Directory is anything else, Vercel serves that folder instead and
every route 404s. A quick way to tell what it is currently serving:

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://YOUR-DEPLOYMENT.vercel.app/
```

`404` on `/` but `200` on some other path means the Root Directory is pointing
at the wrong folder.

## Two working configurations

| Root Directory | Config file used | Notes |
|---|---|---|
| `web` | `web/vercel.json` | Recommended. Unambiguous. |
| `.` (repo root) | `vercel.json` (has `outputDirectory: "web"`) | Also works. |

Both files are committed, so either setting is correct. What does *not* work is
pointing the Root Directory at any other subdirectory.

## Fresh import (if the existing project is misconfigured)

1. Delete the old project, or just fix its Root Directory as above.
2. <https://vercel.com/new> → import `fruit-fly-lab`.
3. **Root Directory: `web`** — set this on the import screen, before deploying.
4. Framework Preset: **Other**. Leave Build Command and Install Command empty.
5. Deploy.

## From the CLI

Run it from the repository root, not from a subdirectory. Running `npx vercel`
inside `requirements/` is what creates a project named "requirements" that
serves two text files.

```bash
npx vercel --cwd web --prod
```

## What visitors download

| File | Raw | Served (Brotli/gzip) |
|---|---:|---:|
| `data/connectome.bin` | 22.9 MB | ~11 MB |
| `data/neurons.bin` | 3.5 MB | ~2 MB |
| `data/meta.json` | 0.7 MB | ~0.2 MB |

Cached immutably after the first visit. Requires a browser with module Web
Workers: Chrome/Edge 91+, Firefox 114+, Safari 15+, and roughly 200 MB of tab
memory.

## Verify before deploying

```bash
python -m tools.verify_web_engine
```

All three scenarios must report `MATCH` — identical spike counts across all
139,255 neurons between the browser engine and the Python engine.
