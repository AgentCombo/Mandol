# Mandol Website

This directory contains the Docusaurus static front page for Mandol. It is not
the Sphinx documentation source; the maintained docs live in `../docs`.

## Install

```bash
npm ci
```

## Develop

```bash
npm run start
```

## Typecheck

```bash
npm run typecheck
```

## Build

```bash
npm run build
```

The deployment workflow builds the maintained Sphinx documentation and publishes
it alongside the Docusaurus front page at `/Mandol/docs/`.
