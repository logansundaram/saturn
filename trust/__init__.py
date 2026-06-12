"""The trust stack — the product. One module per concern: the gate policy object (policy), the
network chokepoint + egress ledger/chain (egress), cloud-boundary secret stripping (redaction),
prompt-injection quarantine + taint tracking (quarantine), ed25519 artifact signing + the
canonical digest scheme (signing), the per-answer trust receipt (receipt), the signed posture
attestation (trust_report), and answer-level provenance — the Glass Box (glassbox).
"""
