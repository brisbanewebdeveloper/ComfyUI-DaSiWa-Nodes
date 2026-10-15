"""Session listing exposes safe preview handles and missing-export state."""
from test_h3_segment_backend import core, latent
import folder_paths


def test_listing_reports_safe_export_handle_and_missing_export(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, 'get_output_directory', lambda: str(tmp_path))
    store = core.ClipStore(tmp_path / 'checkpoints')
    path = tmp_path / 'takes' / 'root.mp4'
    path.parent.mkdir()
    path.write_bytes(b'export availability only; format is validated by Publish')
    ticket = store.stage(latent(124), dict(session='s',run_id='root',mode='REF2VA',resolved_prompt='root'))
    store.publish(ticket, path)
    row = store.list_clips('s')['clips'][0]
    assert row['export_available'] is True
    assert row['export']['filename'] == 'root.mp4'
    assert row['export']['subfolder'] == 'takes'
    assert 'output_path' not in row
    assert str(tmp_path) not in str(row)
    path.unlink()
    row = store.list_clips('s')['clips'][0]
    assert row['export_available'] is False
    assert 'filename' not in row.get('export', {})


def test_listing_accepts_published_temp_export_and_legacy_temp_path(tmp_path, monkeypatch):
    output, temp = tmp_path / 'output', tmp_path / 'temp'
    output.mkdir()
    (temp / 'nested').mkdir(parents=True)
    path = temp / 'nested' / 'take.mp4'
    path.write_bytes(b'temporary exported video')
    monkeypatch.setattr(folder_paths, 'get_output_directory', lambda: str(output))
    monkeypatch.setattr(folder_paths, 'get_temp_directory', lambda: str(temp))
    store = core.ClipStore(output / 'checkpoints')
    for name, export in [('new', {'kind':'full','frames':124,'start_frame':0,'filename':'take.mp4','subfolder':'nested','type':'temp'}),('legacy',None)]:
        ticket = store.stage(latent(124), dict(session='s',run_id=name,mode='REF2VA',resolved_prompt='root'))
        store.publish(ticket, path, export=export)
    rows = store.list_clips('s')['clips']
    assert len(rows) == 2
    for row in rows:
        assert row['export_available'] is True
        assert row['export']['filename'] == 'take.mp4'
        assert row['export']['subfolder'] == 'nested'
        assert row['export']['type'] == 'temp'
        assert str(tmp_path) not in str(row)


def test_listing_enables_existing_imported_prefix_without_exposing_input_path(tmp_path, monkeypatch):
    source_dir = tmp_path / 'input'
    output = tmp_path / 'output'
    source_dir.mkdir()
    output.mkdir()
    source = source_dir / 'source.mp4'
    source.write_bytes(b'original imported media')
    monkeypatch.setattr(folder_paths, 'get_input_directory', lambda: str(source_dir))
    monkeypatch.setattr(folder_paths, 'get_output_directory', lambda: str(output))
    store = core.ClipStore(output / 'checkpoints')
    ticket = store.stage(latent(124), dict(session='s',run_id='root',mode='REF2VA',resolved_prompt='root',
        provenance={'kind':'imported_video','filename':'source.mp4'}))
    store.publish(ticket, source)
    row = store.list_clips('s')['clips'][0]
    assert row['export_available'] is True
    assert row['export']['filename'] == 'source.mp4'
    assert row['export']['type'] == 'input'
    assert row['export']['source_prefix'] is True
    assert str(tmp_path) not in str(row)


def test_listing_does_not_expose_or_enable_external_output(tmp_path, monkeypatch):
    output = tmp_path / 'output'
    output.mkdir()
    monkeypatch.setattr(folder_paths, 'get_output_directory', lambda: str(output))
    external = tmp_path / 'external.mp4'
    external.write_bytes(b'not under output')
    store = core.ClipStore(output / 'checkpoints')
    ticket = store.stage(latent(124), dict(session='s',run_id='root',mode='REF2VA',resolved_prompt='root'))
    store.publish(ticket, external)
    row = store.list_clips('s')['clips'][0]
    assert row['export_available'] is False
    assert 'filename' not in row.get('export', {})
    assert str(tmp_path) not in str(row)
