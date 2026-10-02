import pytest

from cise.shared.prompting import TASKS, build_prompt
from cise.cci.interval_bridge import annotate_prompt


@pytest.mark.parametrize('task', ['wbg', 'sse', 'pv'])
def test_shared_prompt_and_interval_feedback_contract(task):
    parents = [{'parent_id': 'example-parent', 'path': 'private-path-sentinel',
                'cif_path': 'private-cif-sentinel', 'sites': []}]
    prompt = build_prompt(task, parents, {}, 1)
    assert TASKS[task]['target'] in prompt
    assert 'private-path-sentinel' not in prompt
    assert 'private-cif-sentinel' not in prompt
    assert 'uncorrected surrogate predictions' in prompt
    interval_prompt = annotate_prompt(prompt, task)
    assert 'Gibbs test-point-corrected conformal bounds' in interval_prompt
    assert 'uncorrected surrogate predictions' not in interval_prompt
    assert 'Raw estimates are excluded from feedback' in interval_prompt
    assert TASKS[task]['target'] in interval_prompt
