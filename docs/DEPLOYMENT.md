# Deploying DAG Brain for other people

DAG Brain is built as a single-user tool. For a lab, a supervisor or a collaborator you can run one shared instance, with a sign-in, the settings screen locked, and the container hardened. This guide covers a small, invitation-only deployment, not a public service.

## What the hosted mode does

| Setting | Effect |
|:--|:--|
| `DAGBRAIN_PASSWORD` | Every page and API call needs a signed session cookie from the sign-in page. Wrong guesses are slowed, five failures from one address lock it out for 15 minutes, and a global brake limits floods. |
| `DAGBRAIN_SECRET` | Signs the session cookie. Keep it stable so a restart does not sign everyone out. |
| `DAGBRAIN_HOSTED=1` | Locks the settings screen. Model keys, endpoints and the privacy option can only come from the server's environment, so a visitor cannot redirect the server's requests or spend your key. |
| `DAGBRAIN_ORIGINS` | Cross-site API access is off unless you list origins here. |
| `DAGBRAIN_MAX_JOBS` | How many heavy analyses (CSV tests, estimation, PDF parsing) run at once. They run off the web thread, so one analysis does not freeze the app. |

The shared passphrase is a gate for invited people, not per-user accounts. Everyone who signs in sees and can change every study. Use it for demos and small trusted groups; keep sensitive data out of it.

## Run it with Docker

On the server:

```bash
git clone https://github.com/SlightlyHappy/DAG_Brain.git && cd DAG_Brain/deploy
cp env.hosted.example .env && chmod 600 .env
# put a passphrase and a secret in .env:   openssl rand -base64 18   and   openssl rand -hex 32
mkdir -p data
docker compose -f docker-compose.hosted.yml up -d --build
curl http://127.0.0.1:8010/api/health
```

The compose file publishes the app on `127.0.0.1:8010` only, runs it as an unprivileged user with a read-only filesystem, all capabilities dropped, and limits of 2 GB and 2 CPUs. Studies, uploads and settings live in `deploy/data/`. Back that folder up.

## Put it on the internet

Terminate HTTPS in front of the container with a reverse proxy (Caddy, nginx) or a Cloudflare Tunnel that forwards a hostname to `http://127.0.0.1:8010`. The app marks its cookie `Secure` when the proxy sends `X-Forwarded-Proto: https`, and uses `CF-Connecting-IP` for the lockout when Cloudflare is in front.

For stronger control, also put the hostname behind your proxy's own access control, for example Cloudflare Access restricted to invited email addresses. The built-in sign-in then becomes a second layer.

## Before you share it

- **Data.** Anything someone uploads sits on your server. Use the built-in examples or synthetic data for demonstrations.
- **A model key is optional.** Without one the rule-based assistant answers and all calculations still work. If you add one, use a key with a spending limit.
- **Literature search** needs outbound access. Set `OFFLINE_MODE=true` to block it, together with cloud models.
- **Updates.** `git pull && docker compose -f docker-compose.hosted.yml up -d --build`. Review changes before pulling.
- **If you modify it** and let others use it, the AGPL requires you to offer them your source and keep the attribution (see [NOTICE](../NOTICE)).
