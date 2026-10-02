import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'release_checks', Path(__file__).resolve().parents[1] / 'scripts/check_release.py'
)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


@pytest.mark.parametrize('name,data', [
    ('assets/model.json', b'{}'),
    ('AK.md', b'credential'),
    ('src/sample.parquet', b'dataset'),
    ('src/example.py', b'sk-' + b'a' * 40),
    ('src/example.py', b'AKIA' + b'A' * 16),
    ('docs/example.md', b'/' + b'home/example/private/file'),
])
def test_prohibited_release_content_is_rejected(name, data):
    with pytest.raises(ValueError):
        release.check_item(name, data)


def test_json_dataset_cannot_hide_inside_source(tmp_path, monkeypatch):
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src/data.json').write_text('{"rows": []}')
    monkeypatch.setattr(release, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='Unexpected public asset'):
        release.check_source()


@pytest.mark.parametrize('wheel', [True, False])
def test_stale_package_rejected_against_source_manifest(tmp_path, monkeypatch, wheel):
    (tmp_path / 'provenance').mkdir()
    (tmp_path / 'provenance/sources.json').write_text(json.dumps({'sources': [
        {'destination': 'src/cise/kernel.py',
         'distributed_sha256': hashlib.sha256(b'current').hexdigest()}
    ]}))
    monkeypatch.setattr(release, 'ROOT', tmp_path)
    name = 'cise/kernel.py' if wheel else 'package/src/cise/kernel.py'
    release.check_archive_provenance([(name, b'current')], wheel=wheel)
    with pytest.raises(ValueError, match='Archive provenance mismatch'):
        release.check_archive_provenance([(name, b'old')], wheel=wheel)


def test_new_integration_module_is_also_checked(tmp_path, monkeypatch):
    (tmp_path / 'provenance').mkdir()
    (tmp_path / 'provenance/sources.json').write_text('{"sources": []}')
    (tmp_path / 'src/cise').mkdir(parents=True)
    (tmp_path / 'src/cise/providers.py').write_bytes(b'new integration')
    monkeypatch.setattr(release, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='Archive provenance mismatch'):
        release.check_archive_provenance([('cise/providers.py', b'old integration')], wheel=True)
