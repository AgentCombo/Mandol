# Security Policy

## Reporting a Vulnerability

Please report vulnerabilities through GitHub's private vulnerability-reporting
form:

https://github.com/AgentCombo/Mandol/security/advisories/new

Do not include vulnerability details in a public issue. If private reporting is
unavailable, open a public issue requesting a private contact channel without
disclosing the sensitive details.

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if available)

### Response Expectations

Mandol is a research artifact maintained on a best-effort basis. The
maintainers will triage reports as availability permits and will coordinate
disclosure for confirmed issues.

## Supported Versions

| Version or branch | Support status |
| --- | --- |
| `0.1.0` / `main` | Best-effort fixes for the current public release and active development branch |
| `0.1.0a1` / `paper-repro` | Best-effort fixes that preserve artifact behavior |
| Earlier snapshots | Unsupported |

## Security Best Practices

- Never commit API keys, passwords, or other secrets to the repository.
- Use `.env` or process environment variables for sensitive configuration.
- Review model files and benchmark datasets before using untrusted copies.
- Keep dependencies current within the compatibility constraints documented by
  the artifact.
