"""Preserved physical identity for existing planner refinement adapters."""
from hashlib import sha256
import json

def signature(days):
    """Physical prescriptions, including ranges/structure, not only target miles."""
    return [{'date': d.date.isoformat(), 'role': d.day_role,
             'prescription': d.recommendation.model_dump(mode='json', exclude={'generated_at'})
             if d.recommendation else None} for d in days]


def fingerprint(days):
    return sha256(json.dumps(signature(days), sort_keys=True).encode()).hexdigest()
