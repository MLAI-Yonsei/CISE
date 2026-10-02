import copy
import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock
import pytest
import requests
from cise import providers
from cise.runtime import load, save


@pytest.fixture
def provider_env(tmp_path, monkeypatch):
    config = copy.deepcopy(providers.CONFIG)
    config['demo'] = False
    monkeypatch.setattr(providers, 'CONFIG', config)
    monkeypatch.setattr(providers, 'ROOT', tmp_path)
    monkeypatch.setattr(providers, 'MODEL', 'example/model')
    monkeypatch.setattr(providers, 'key', lambda: 'test-secret')
    return tmp_path


def args():
    return 'wbg', 1, 0, [{'role': 'user', 'content': 'Generate two candidates'}], {}


def response(http=200, payload=None):
    result = Mock(status_code=http)
    result.json.return_value = payload or dict(id='generation-id', choices=[dict(message=dict(content='{"candidates":[]}'))])
    return result


def test_structured_request_and_concurrent_cache_issue_one_post(provider_env, monkeypatch):
    post = Mock(return_value=response(payload=dict(id='id', choices=[dict(message=dict(content='answer'))],
        usage=dict(prompt_tokens=3, completion_tokens=2, total_tokens=5, cost=99))))
    monkeypatch.setattr(providers.requests, 'post', post)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: providers.request(*args()), range(8)))
    assert all(content == 'answer' for content, _ in results)
    assert post.call_count == 1
    body = post.call_args.kwargs['json']
    assert body['model'] == 'example/model'
    assert body['response_format']['json_schema']['strict']
    assert results[0][1]['usage'] == dict(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    stored = (provider_env / 'wbg/api/0001_0.json').read_text()
    assert 'test-secret' not in stored
    assert 'cost' not in stored


@pytest.mark.parametrize('changed', ['messages', 'schema', 'model', 'tokens', 'demo'])
def test_cache_refuses_changed_request(provider_env, monkeypatch, changed):
    post = Mock(return_value=response())
    monkeypatch.setattr(providers.requests, 'post', post)
    request_args = list(args())
    providers.request(*request_args)
    if changed == 'messages': request_args[3] = [{'role': 'user', 'content': 'different'}]
    if changed == 'schema': request_args[4] = {'type': 'object'}
    if changed == 'model': monkeypatch.setattr(providers, 'MODEL', 'other/model')
    if changed == 'tokens': providers.CONFIG['llm']['max_output_tokens'] += 1
    if changed == 'demo': providers.CONFIG['demo'] = True
    with pytest.raises(RuntimeError, match='CONTEXT_MISMATCH'):
        providers.request(*request_args)
    assert post.call_count == 1


@pytest.mark.parametrize('http,expected', [(401, 'AUTH_OR_CREDIT'), (402, 'AUTH_OR_CREDIT'),
    (403, 'AUTH_OR_CREDIT'), (400, 'CONFIGURATION_REJECTED'), (404, 'CONFIGURATION_REJECTED'),
    (405, 'CONFIGURATION_REJECTED'), (422, 'CONFIGURATION_REJECTED')])
def test_deterministic_rejection_is_cached(provider_env, monkeypatch, http, expected):
    post = Mock(return_value=response(http=http))
    monkeypatch.setattr(providers.requests, 'post', post)
    for _ in range(2):
        with pytest.raises(RuntimeError, match=expected): providers.request(*args())
    assert post.call_count == 1


def test_transient_http_error_cached_as_known_rejection(provider_env, monkeypatch):
    post = Mock(return_value=response(http=503))
    monkeypatch.setattr(providers.requests, 'post', post)
    assert providers.request(*args())[0] == ''
    assert providers.request(*args())[1]['status'] == 'http_error'
    assert post.call_count == 1


@pytest.mark.parametrize('failure', ['timeout', 'decode', 'shape'])
def test_ambiguous_failure_never_reposted(provider_env, monkeypatch, failure):
    reply = response()
    post = Mock(return_value=reply)
    if failure == 'timeout': post.side_effect = requests.Timeout('test-secret')
    if failure == 'decode': reply.json.side_effect = ValueError('test-secret')
    if failure == 'shape': reply.json.return_value = {'choices': []}
    monkeypatch.setattr(providers.requests, 'post', post)
    for _ in range(2):
        with pytest.raises(RuntimeError, match='OUTCOME_UNKNOWN'): providers.request(*args())
    assert post.call_count == 1
    assert 'test-secret' not in (provider_env / 'wbg/api/0001_0.json').read_text()


def test_process_crash_after_intent_never_reposted(provider_env, monkeypatch):
    post = Mock(side_effect=KeyboardInterrupt)
    monkeypatch.setattr(providers.requests, 'post', post)
    with pytest.raises(KeyboardInterrupt): providers.request(*args())
    assert load(provider_env / 'wbg/api/0001_0.json')['status'] == 'request_started'
    with pytest.raises(RuntimeError, match='OUTCOME_UNKNOWN'): providers.request(*args())
    assert post.call_count == 1


def test_demo_is_offline_and_cached_without_credentials(provider_env, monkeypatch):
    providers.CONFIG['demo'] = True
    save(provider_env / 'wbg/api/0001_context.json', dict(parents=[dict(parent_id='seed')]))
    monkeypatch.setattr(providers, 'key', Mock(side_effect=AssertionError('credential access')))
    monkeypatch.setattr(providers.requests, 'post', Mock(side_effect=AssertionError('network access')))
    content, record = providers.request(*args())
    assert len(json.loads(content)['candidates']) == 2
    assert record['demo'] and record['status'] == 'responded'
    assert providers.request(*args())[0] == content


def test_missing_credentials_fail_before_durable_intent(provider_env, monkeypatch):
    monkeypatch.setattr(providers, 'key', Mock(side_effect=RuntimeError('Missing API key')))
    with pytest.raises(RuntimeError, match='Missing API key'): providers.request(*args())
    assert not (provider_env / 'wbg/api/0001_0.json').exists()
