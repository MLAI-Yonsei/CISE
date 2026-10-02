from types import SimpleNamespace

import pytest

from cise.cci import interval_bridge
from cise.shared import islands
from cise.shared.inference import score_values


def candidate(candidate_id, score, **flags):
    return dict(candidate_id=candidate_id, formula=candidate_id, score=score, **flags)


def ids(rows):
    return {row['candidate_id'] for row in rows}


def assert_partition(archive, successes, failures):
    prompt_success, prompt_failure = archive.get_prompt_memories()
    assert ids(prompt_success) == set(successes)
    assert ids(prompt_failure) == set(failures)
    for rank in (None, lambda rows: sorted(rows, key=lambda row: -row['score'])):
        unique = archive.get_unique_examples(rank_candidates=rank)
        assert ids(unique['success']) == set(successes)
        assert ids(unique['failure']) == set(failures)


def corrected_wbg(monkeypatch, *, cutoff=.5, status='ok'):

    properties = {'band_gap': {'lower': {'q': .5}},
                  'formation_energy': {'upper': {'q': .5}}}
    monkeypatch.setattr(interval_bridge, 'calibrations',
                        lambda: {'wbg': {'properties': properties}})
    monkeypatch.setattr(interval_bridge, 'scales',
                        lambda *args: {'band_gap': 1., 'formation_energy': 1.})
    monkeypatch.setattr(interval_bridge, 'structure_feature_values', lambda path: {})
    weights = SimpleNamespace(
        cutoff=lambda *args: SimpleNamespace(cutoff=cutoff, status=status, diagnostics={}),
        ratio=lambda *args: 1., iteration=1, frozen_hash='offline-feedback-test')
    row = dict(candidate_id='interval_boundary', formula='MgO', task='wbg',
               property_values={'band_gap': 3., 'formation_energy': -1.5},
               species=['Mg', 'O'], cif_path='mocked-structure-path')
    return interval_bridge.correct(row, score_values, weights)


@pytest.mark.parametrize('flags,score,successful', [
    ({'proxy_pass': True}, 0., True),
    ({'proxy_pass': True}, -1., True),
    ({'proxy_pass': False}, 10., False),
    ({'interval_pass': True, 'proxy_pass': False}, 0., True),
    ({'interval_pass': False, 'proxy_pass': True}, 10., False),
    ({'interval_pass': False}, 10., False),
    ({}, 10., False),
])
def test_feedback_partition_uses_qualification_not_score(flags, score, successful):
    archive = islands.Island(4, 1., 10)
    row = candidate('example', score, **flags)
    archive.register_candidate(row, {'total': score}, iteration=1)
    assert_partition(archive, ['example'] if successful else [],
                     [] if successful else ['example'])


def test_point_threshold_pass_with_zero_reward_is_success():
    score, failures = score_values('wbg', {'band_gap': 2.5, 'formation_energy': -1.},
                                   ['Mg', 'O'])
    assert score == 0. and failures == []
    archive = islands.Island(4, 1., 10)
    row = candidate('point_boundary', score, proxy_pass=not failures)
    archive.register_candidate(row, {'total': score}, iteration=1)
    assert_partition(archive, ['point_boundary'], [])


def test_interval_threshold_pass_with_zero_reward_is_success(monkeypatch):
    row = corrected_wbg(monkeypatch)
    assert row['conformal_bounds'] == {
        'band_gap': {'lower': 2.5}, 'formation_energy': {'upper': -1.}}
    assert row['score'] == 0. and row['interval_pass'] is True
    assert row['proxy_pass'] is True and row['abstained'] is False
    archive = islands.Island(4, 1., 10)
    archive.register_candidate(row, {'total': row['score']}, iteration=1)
    assert_partition(archive, ['interval_boundary'], [])


def test_scores_still_rank_within_flag_based_partitions():
    archive = islands.Island(4, 1., 10)
    rows = [candidate('pass_low', 0., proxy_pass=True),
            candidate('fail_low', 1., proxy_pass=False),
            candidate('pass_high', 3., proxy_pass=True),
            candidate('fail_high', 5., proxy_pass=False)]
    for row in rows:
        archive.register_candidate(row, {'total': row['score']}, iteration=1)
    success, failure = archive.get_prompt_memories()
    assert [row['candidate_id'] for row in success] == ['pass_high', 'pass_low']
    assert [row['candidate_id'] for row in failure] == ['fail_high', 'fail_low']
    unique = archive.get_unique_examples()
    assert [row['candidate_id'] for row in unique['success']] == ['pass_high', 'pass_low']
    assert [row['candidate_id'] for row in unique['failure']] == ['fail_high', 'fail_low']


@pytest.mark.parametrize('flags', [
    {'proxy_pass': False, 'interval_pass': False},
    {'proxy_pass': True, 'interval_pass': True},
    {},
])
def test_direct_island_registration_excludes_abstentions(flags):
    archive = islands.Island(4, 1., 10)
    row = candidate('abstention', 1000., abstained=True, **flags)
    archive.register_candidate(row, {'total': row['score']}, iteration=1)
    assert_partition(archive, [], [])
    assert archive._num_items == 0
    assert archive._clusters == {}
    assert archive._iteration_history == []
    assert archive._unique_candidates == {}
    assert not archive.is_duplicate(row)


def test_real_interval_abstention_never_enters_buffer(monkeypatch):
    row = corrected_wbg(monkeypatch, cutoff=float('inf'), status='unbounded_interval')
    assert row['abstained'] is True and row['interval_pass'] is False
    buffer = islands.ExperienceBuffer(2, reset_period_seconds=10**9)
    buffer.register(0, row, {'total': row['score']}, iteration=1)
    assert buffer._best_candidate_per_island == [None, None]
    assert buffer._best_score_per_island == [float('-inf'), float('-inf')]
    assert buffer._best_scores_per_test_per_island == [None, None]
    for archive in buffer._islands:
        assert_partition(archive, [], [])
        assert archive._iteration_history == [] and archive._num_items == 0
    monkeypatch.setattr(islands.np.random, 'randint', lambda _: 0)
    prompt = buffer.get_prompt(iteration=1)
    assert prompt.memory_success == [] and prompt.memory_failure == []
    assert buffer.get_unique_examples_for_evolution(0) == {'success': [], 'failure': []}


def test_seed_all_and_reset_founders_exclude_high_reward_abstentions(monkeypatch):
    buffer = islands.ExperienceBuffer(4, reset_period_seconds=100)
    valid = candidate('valid', 0., proxy_pass=True, abstained=False)
    abstained = candidate('abstained', 1000., proxy_pass=True,
                          interval_pass=True, abstained=True)

    abstained['formula'] = valid['formula']
    buffer.seed_all([valid, abstained])
    for index, archive in enumerate(buffer._islands):
        assert_partition(archive, ['valid'], [])
        assert ids(archive._iteration_history) == {'valid'}
        assert buffer._best_candidate_per_island[index]['candidate_id'] == 'valid'
        assert buffer._best_score_per_island[index] == 0.
    clock = buffer._last_reset_time + 101
    monkeypatch.setattr(islands.time, 'time', lambda: clock)
    buffer._maybe_reset()
    for index, archive in enumerate(buffer._islands):
        assert_partition(archive, ['valid'], [])
        assert ids(archive._iteration_history) == {'valid'}
        assert buffer._best_candidate_per_island[index]['candidate_id'] == 'valid'
        assert buffer._best_scores_per_test_per_island[index] == {'total': 0.}
    monkeypatch.setattr(islands.np.random, 'randint', lambda _: 0)
    prompt = buffer.get_prompt(iteration=1)
    assert ids(prompt.memory_success) == {'valid'} and prompt.memory_failure == []
