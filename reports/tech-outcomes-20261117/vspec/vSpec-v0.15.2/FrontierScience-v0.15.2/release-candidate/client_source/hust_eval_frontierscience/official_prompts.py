"""FrontierScience Appendix B, arXiv:2601.21165v1, pp.13-14.
Source: https://arxiv.org/pdf/2601.21165v1 (CC BY 4.0).
Transcription normalizes PDF line wrapping, soft hyphenation and typography;
retains the original 'attemped' spelling. See docs/OFFICIAL_PROTOCOL.md.
"""
SOURCE = {
    "paper": "https://arxiv.org/abs/2601.21165v1",
    "pdf_sha256": "a11077ec358612db1098b715c7631fd249e4b909afda5716fb573ffcc0663cfe",
    "pages": [13, 14],
    "license": "CC BY 4.0",
    "message_role_assumption": "single user message; paper does not specify API message roles",
}
OLYMPIAD = """You are grading an attempted answer to a science olympiad problem. You will be given the problem, attempted answer, and reference answer. Evaluate the solution against the provided reference solution, ensuring it is complete and matches the reference solution. Pay close attention to detail and grade it strictly, but fairly.

The reference answer is either a single number or expression in latex formatting, a chemical formula, a compound name, or a phrase referring to a specific name, entity, or method.

Mark the attempted answer as correct if it fully matches the reference answer or is otherwise equivalent (e.g., an equivalent algebraic expression, a numerical number within 1 decimal place rounding of the reference answer (e.g., 6.69 ≈ 6.7), an equivalent name for a compound/formula, equivalent when accounting for units, etc.). Mark it as incorrect if it is not equivalent to the reference answer.
***
The problem: {problem}
***
The reference answer: {reference answer}
***
The attempted answer: {answer}
***
First, think step-by-step about whether the attempted answer matches the reference answer. If the attempted answer is correct, write "VERDICT: CORRECT" in the last line of your response, with no other text or formatting. If it is incorrect, write "VERDICT: INCORRECT"."""
RESEARCH = """You are grading a science exam.

You will be given the problem, attempted answer, and a rubric to grade the answer. The rubric will total up to 10 points.

Evaluate the attemped answer against the provided rubric. Pay close attention to detail and grade it strictly, but fairly. Only evaluate against the rubric, as you yourself should not make any judgements (e.g., even if you think the answer is correct but rubric is wrong, you should treat the rubric as the gold standard). Return the absolute total number of points earned (it can be a decimal based on the rubric). ***
The problem: {problem}
***
The rubric: {rubric}
***
The attempted answer: {answer}
***
First, think step-by-step about each rubric item. Explain your reasoning for each rubric item. Then, tally the points up and write VERDICT: <total_points> in the last line of your response, no other text. For example, VERDICT: 2.5 or VERDICT: 8."""
PROMPTS = {"olympiad": OLYMPIAD, "research": RESEARCH}
