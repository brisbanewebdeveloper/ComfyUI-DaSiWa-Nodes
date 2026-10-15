"""Editing saved archives preserves original files and selected tensor values."""
import asyncio
import importlib
import json
import types

import pytest
import torch
from safetensors.torch import save_file
from test_h3_refmod_export import exporter, package


@pytest.fixture
def archive(tmp_path, monkeypatch):
    m = exporter()
    fmt = importlib.import_module(package.__name__ + '.helper_refmod_format')
    library = importlib.import_module(package.__name__ + '.refmod_library')
    monkeypatch.setattr(m, 'refmods_roots', lambda: [str(tmp_path)])
    monkeypatch.setattr(fmt, 'refmods_roots', lambda: [str(tmp_path)])
    return m, fmt, library


@pytest.mark.parametrize('legacy', [False, True])
def test_edit_saved_members_without_encoding(tmp_path, archive, legacy):
    m, fmt, library = archive
    latent = torch.arange(96, dtype=torch.float32).reshape(1, 24, 1, 2, 2)
    if legacy:
        save_file({'latent': latent}, tmp_path / 'source.safetensors', metadata={
            'refmod_meta': json.dumps({'kind': 'image', '_format_version': 4,
                'description': 'Original', 'concept_type': 'identity'})})
    else:
        m.dasiwa_save_bundle([(torch.ones(1, 32, 2, 7), {'kind': 'audio', 'description': 'Drop'}),
                             (latent, {'kind': 'image', 'description': 'Original'})], [],
                            name='source', description='Bundle', concept_type='identity')
    original = (tmp_path / 'source.safetensors').read_bytes()
    detail = library.refmod_details('source')
    assert detail['concept_type'] == 'identity'
    assert detail['members'][-1] == {'kind': 'image', 'description': 'Original'}
    selected = {'name': 'source', 'members': [{'index': 0 if legacy else 1, 'description': ''}]}
    result = m.DaSiWaH3RefModExport().create_refmod(
        json.dumps({'items': [], 'refmods': [selected]}), 'edit', name='copy',
        description='Edited bundle', concept_type='style')
    assert result['ui']['dasiwa_refmod_export'][0]['members'] == 1
    copied = fmt.load_refmods('copy')
    assert torch.equal(copied[0][0], latent)
    assert copied[0][1]['description'] == ''
    assert library.refmod_details('copy')['description'] == 'Edited bundle'
    assert library.refmod_details('copy')['concept_type'] == 'style'
    assert (tmp_path / 'source.safetensors').read_bytes() == original


@pytest.mark.parametrize('members', [[], None, 'bad', [{}], [{'index': True, 'description': ''}],
    [{'index': -1, 'description': ''}], [{'index': 2, 'description': ''}],
    [{'index': 0, 'description': 3}], [{'index': 0, 'description': ''}, {'index': 0, 'description': ''}]])
def test_reject_invalid_member_selection(monkeypatch, members):
    m = exporter()
    monkeypatch.setattr(m, 'load_refmods', lambda _: [(torch.zeros(1, 24, 1, 2, 2), {'kind': 'image'})] * 2)
    with pytest.raises(ValueError, match='member'):
        m.dasiwa_export_refs(json.dumps({'refmods': [{'name': 'source', 'members': members}]}))


def test_library_exposes_member_notes_without_replacing_bundle_description(tmp_path, archive, monkeypatch):
    m, fmt, library = archive
    monkeypatch.setattr(library, 'refmods_roots', lambda: [str(tmp_path)])
    monkeypatch.setattr(library, '_entries_cache', {'sig': None, 'entries': []})
    m.dasiwa_save_bundle([
        (torch.zeros(1, 24, 1, 2, 2), {'kind': 'image', 'description': 'First picture'}),
        (torch.ones(1, 24, 1, 2, 2), {'kind': 'image', 'description': 'Second picture'}),
        (torch.ones(1, 32, 2, 7), {'kind': 'audio', 'description': ''}),
    ], [], name='source', description='Bundle overview')
    entry, = library.library_entries()
    assert entry['description'] == 'Bundle overview'
    assert entry['kinds'] == ['image', 'image', 'audio']
    assert entry['members'] == [
        {'kind': 'image', 'description': 'First picture'},
        {'kind': 'image', 'description': 'Second picture'},
        {'kind': 'audio', 'description': ''},
    ]


def test_refmod_detail_http_metadata_only(tmp_path, archive):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    m, fmt, library = archive
    m.dasiwa_save_bundle([(torch.zeros(1, 24, 1, 2, 2),
                          {'kind': 'image', 'description': 'Note', 'source': '/private/file.png'})], [], name='source')
    async def check():
        routes = web.RouteTableDef()
        library.register_routes(types.SimpleNamespace(routes=routes))
        app = web.Application()
        app.add_routes(routes)
        async with TestClient(TestServer(app)) as client:
            response = await client.get('/dasiwa/refmods/detail', params={'name': 'source'})
            assert response.status == 200
            detail = await response.json()
            assert detail['members'] == [{'kind': 'image', 'description': 'Note'}]
            for name in ('../escape', '/absolute', 'missing', ''):
                response = await client.get('/dasiwa/refmods/detail', params={'name': name})
                assert response.status == 400
    asyncio.run(check())
