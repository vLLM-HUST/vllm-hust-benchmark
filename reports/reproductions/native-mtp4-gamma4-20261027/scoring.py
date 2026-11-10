"""Official answer extraction order; invalid answers count wrong, never random."""
import re

def extract_answer(text):
    text = text.replace('**', '')
    for pattern, flags in [(r'answer is \(?([A-J])\)?', 0),
                           (r'.*[aA]nswer:\s*([A-J])', 0),
                           (r'\b([A-J])\b(?!.*\b[A-J]\b)', re.DOTALL)]:
        match = re.search(pattern, text, flags)
        if match:
            return match.group(1)
    return None
