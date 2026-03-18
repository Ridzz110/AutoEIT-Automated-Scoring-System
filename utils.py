import re

def clean_stimulus(stimulus: str) -> str:
    """Remove score number from stimulus e.g. 'Quiero cortarme el pelo (7)' -> 'Quiero cortarme el pelo'"""
    return re.sub(r'\s*\(\d+\)', '', stimulus).strip()