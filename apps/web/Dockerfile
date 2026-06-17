FROM node:20-alpine AS base
RUN npm install -g pnpm

WORKDIR /app

# ── Dev stage (used by docker-compose for hot-reload) ────────────────────────
FROM base AS dev
COPY pnpm-workspace.yaml package.json pnpm-lock.yaml ./
COPY apps/web/package.json apps/web/
RUN pnpm install --frozen-lockfile
COPY apps/web apps/web
WORKDIR /app/apps/web
EXPOSE 5173
CMD ["pnpm", "dev", "--host", "0.0.0.0"]

# ── Build stage ───────────────────────────────────────────────────────────────
FROM base AS build
COPY pnpm-workspace.yaml package.json pnpm-lock.yaml ./
COPY apps/web/package.json apps/web/
RUN pnpm install --frozen-lockfile
COPY apps/web apps/web
WORKDIR /app/apps/web
RUN pnpm build

# ── Production: serve with nginx ─────────────────────────────────────────────
FROM nginx:alpine AS prod
COPY --from=build /app/apps/web/dist /usr/share/nginx/html
COPY apps/web/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
