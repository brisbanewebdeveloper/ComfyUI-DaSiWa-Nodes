"""Training-free Director archives, executed in the normal ComfyUI queue."""
import json
import os
import tempfile
import uuid
from pathlib import Path

import torch
from safetensors.torch import save_file

from .helper_refmod_format import _validate_refmod_latent, load_refmods, refmods_roots


def dasiwa_interrupted():
    from comfy.model_management import throw_exception_if_processing_interrupted
    throw_exception_if_processing_interrupted()


def dasiwa_validate_name(name):
    if not isinstance(name, str):
        raise ValueError('RefMod name must be a plain basename.')
    if not name:
        return
    stem = name.split('.')[0].rstrip(' ').upper()
    reserved = {'CON', 'PRN', 'AUX', 'NUL', 'CONIN$', 'CONOUT$'}
    reserved.update(f'{prefix}{n}' for prefix in ('COM', 'LPT') for n in '123456789¹²³')
    if (name.startswith('.') or name.endswith(('.', ' ')) or name != name.strip() or
            any(c in '/\\:<>"|?*%' or ord(c) < 32 or 127 <= ord(c) <= 159 for c in name) or
            stem in reserved or len(name.encode('utf-8')) > 200):
        raise ValueError('RefMod name must be a safe plain basename (no paths or reserved names).')


def dasiwa_save_bundle(refs, warnings, name='', description='', concept_type='generic'):
    dasiwa_validate_name(name)
    if not 1 <= len(refs) <= 256:
        raise ValueError('RefMod export requires 1–256 members.')
    if concept_type not in ('generic', 'identity', 'style', 'motion', 'audio'):
        raise ValueError('Invalid RefMod concept_type.')
    tensors, members = {}, []
    for index, (latent, metadata) in enumerate(refs):
        dasiwa_interrupted()
        meta = dict(metadata)
        if meta.get('kind') not in {'image', 'video', 'audio'}:
            raise ValueError('Invalid RefMod member kind.')
        _validate_refmod_latent(latent, meta, str(index + 1))
        if not torch.isfinite(latent).all().item():
            raise ValueError('RefMod export contains a non-finite latent.')
        meta.update(_format_version=4, mode=meta.get('mode', 'encode'))
        meta['latent_t'] = int(latent.shape[-1] if meta['kind'] == 'audio' else latent.shape[2])
        if meta['kind'] == 'audio':
            meta['sample_rate'] = 32000
        else:
            meta.update(latent_h=int(latent.shape[3]), latent_w=int(latent.shape[4]))
        tensors[f'ref_{index}'] = latent.detach().cpu().contiguous().clone()
        members.append(meta)
    name = name or 'director-' + uuid.uuid4().hex
    root = Path(refmods_roots()[0])
    root.mkdir(parents=True, exist_ok=True)
    metadata = {'_format_version': 5, 'kind': 'bundle', 'name': name,
                'members': members, 'description': description, 'concept_type': concept_type,
                'dasiwa_warnings': list(warnings)}
    fd, temporary = tempfile.mkstemp(prefix='.', suffix='.tmp', dir=root)
    os.close(fd)
    try:
        save_file(tensors, temporary, metadata={'refmod_meta': json.dumps(metadata)})
        with open(temporary, 'rb+') as handle:
            os.fsync(handle.fileno())
        dasiwa_interrupted()
        # Same-directory link publishes atomically without replacing existing archives.
        os.link(temporary, root / (name + '.safetensors'))
    finally:
        Path(temporary).unlink(missing_ok=True)
    return name


def dasiwa_input_root():
    import folder_paths
    return folder_paths.get_input_directory()


class _DaSiWaCaptureClip:
    """Skip only unused text conditioning; Core still performs real VAE encoding."""
    def tokenize(self, text, **kwargs):
        return None

    def encode_from_tokens_scheduled(self, tokens):
        return [[None, {}]]


def dasiwa_capture(kwargs):
    from .nodes_minimax_h3_director_guide import _native_node
    positive, _ = _native_node('MiniMaxH3ReferenceToVideo').execute(
        clip=_DaSiWaCaptureClip(), prompt='', width=32, height=32, ref_image_size='max', **kwargs)
    blocks = positive[0][1].get('minimax_refs', [])
    if not blocks:
        raise ValueError('Core returned no encoded RefMod references.')
    return blocks


def dasiwa_media_crop(item, path, stream_kind):
    import math
    from .h3_continuity.pyav_media import open_media, check_dimensions
    with open_media(path) as container:
        streams = getattr(container.streams, stream_kind)
        if not streams:
            raise ValueError('RefMod source has no requested media stream.')
        stream = streams[0]
        duration = (float(stream.duration * stream.time_base)
                    if stream.duration is not None and stream.time_base is not None
                    else (float(container.duration or 0) - float(container.start_time or 0)) / 1_000_000)
        try:
            start = float(item.get('trim_start', 0))
            end = duration if item.get('trim_end') is None else float(item['trim_end'])
        except (TypeError, ValueError) as exc:
            raise ValueError('RefMod crop must be finite numbers.') from exc
        if (not all(math.isfinite(v) for v in (start, end, duration)) or duration <= 0 or
                start < 0 or end <= start or end > duration + .05 or end - start > 15):
            raise ValueError('Choose a nonempty RefMod crop of at most 15 seconds inside the source.')
        if stream_kind == 'video':
            check_dimensions(stream.width, stream.height)
            rate = float(stream.average_rate or stream.guessed_rate or 0)
            if not math.isfinite(rate) or not 0 < rate <= 240:
                raise ValueError('RefMod source needs a known frame rate up to 240 fps.')
            estimate = (math.ceil((end - start) * max(24, rate)) + 2) * stream.width * stream.height * 36
        else:
            rate, channels = stream.rate, stream.codec_context.channels
            if not rate or not 0 < rate <= 384000 or not 0 < channels <= 32:
                raise ValueError('RefMod source has unsupported audio rate/channel count.')
            estimate = (end - start) * rate * channels * 12
        if estimate > 1024 ** 3:
            raise ValueError('RefMod source exceeds the 1 GiB decode estimate; use a shorter/downscaled source.')
    return start, min(end, duration)


def dasiwa_decode_media(path, stream_kind, start, end):
    from fractions import Fraction
    from .h3_continuity.pyav_media import open_media, decoded, guard
    from .helper_minimax_h3_director import decode_audio_frame
    deadline = guard(timeout=60)
    def check():
        dasiwa_interrupted()
        deadline()
    # Existing Director loaders reopen filenames unrestricted and cannot interrupt their decode loops.
    with open_media(path) as container:
        check()
        stream = getattr(container.streams, stream_kind)[0]
        stream.codec_context.thread_count = 2
        origin = Fraction(container.start_time or 0, 1_000_000)
        if start:
            container.seek(int((origin + Fraction(str(start))) / stream.time_base), stream=stream, backward=True)
        chunks, times = [], []
        sample_count = 0
        prior = None
        for frame in decoded(container, stream, check):
            if frame.pts is None or frame.time_base is None:
                raise ValueError('RefMod source needs timestamped frames.')
            timestamp = float(frame.pts * frame.time_base - origin)
            if timestamp >= end:
                break
            if stream_kind == 'video':
                if timestamp < start:
                    prior = (frame, timestamp)
                    continue
                if len(chunks) >= 3602 or (len(chunks) + 1) * frame.width * frame.height * 36 > 1024 ** 3:
                    raise ValueError('RefMod source exceeded its bounded decode estimate.')
                chunks.append(torch.from_numpy(frame.to_rgb().to_ndarray()).float() / 255.)
                times.append(timestamp)
            else:
                rate = int(stream.rate)
                if timestamp + frame.samples / rate <= start:
                    continue
                samples = decode_audio_frame(frame)
                left = max(0, round((start - timestamp) * rate))
                right = min(samples.shape[-1], round((end - timestamp) * rate))
                if right > left:
                    sample_count += right - left
                    if sample_count > (end - start + .05) * rate:
                        raise ValueError('RefMod source exceeded its bounded audio decode estimate.')
                    chunks.append(torch.from_numpy(samples[:, left:right]).float())
        if stream_kind == 'audio':
            if not chunks:
                raise ValueError('RefMod audio crop produced no samples.')
            waveform = torch.cat(chunks, dim=-1).unsqueeze(0)
            if not torch.isfinite(waveform).all() or not torch.any(waveform != 0):
                raise ValueError('RefMod audio crop is silent or non-finite.')
            return {'waveform': waveform, 'sample_rate': int(stream.rate)}
        if prior is not None:
            chunks.insert(0, torch.from_numpy(prior[0].to_rgb().to_ndarray()).float() / 255.)
            times.insert(0, prior[1])
        if not chunks:
            raise ValueError('RefMod video crop produced no frames.')
        source = torch.stack(chunks)
        ticks = torch.arange(start, end, 1 / 24, dtype=torch.float64)
        times = torch.tensor(times, dtype=torch.float64)
        right = torch.searchsorted(times, ticks).clamp(max=len(times) - 1)
        left = (right - 1).clamp(min=0)
        nearest = torch.where((ticks - times[left]).abs() <= (times[right] - ticks).abs(), left, right)
        return source[nearest]


def dasiwa_export_refs(timeline_data, vae=None, audio_vae=None):
    state = json.loads(timeline_data)
    if not isinstance(state, dict):
        raise ValueError('RefMod snapshot must be an object.')
    items, rows = state.get('items', []), state.get('refmods', [])
    if not isinstance(items, list) or not isinstance(rows, list) or len(items) > 256 or len(rows) > 256:
        raise ValueError('RefMod snapshot must contain bounded items/refmods arrays.')
    refs, warnings = [], []
    for item in items:
        dasiwa_interrupted()
        if not isinstance(item, dict) or item.get('type') not in {'image', 'video', 'audio'}:
            raise ValueError('RefMod snapshot contains unsupported media.')
        value = item.get('value')
        if value in (None, ''):
            continue
        from .helper_minimax_h3_director import resolve_input_path, load_image
        from .nodes_minimax_h3_director_guide import _h3_vae_kind
        from .h3_continuity.pyav_media import check_dimensions
        from PIL import Image
        root = dasiwa_input_root()
        path = resolve_input_path(value, root)
        kind = item['type']
        mode = item.get('media_mode', 'video')
        if kind == 'video' and mode not in {'video', 'audio', 'video_audio'}:
            raise ValueError('Invalid RefMod video stream choice.')
        attached = item.get('audio')
        has_attached = attached not in (None, '')
        visual = kind == 'image' or (kind == 'video' and mode != 'audio')
        audio = kind == 'audio' or (kind == 'video' and (mode != 'video' or has_attached))
        if visual and _h3_vae_kind(vae) != 'video':
            raise ValueError('Connect the MiniMax H3 video VAE for RefMod export.')
        if audio and _h3_vae_kind(audio_vae) != 'audio':
            raise ValueError('Connect the MiniMax H3 audio VAE for RefMod export.')
        kwargs = {'vae': vae if visual else None, 'audio_vae': audio_vae if audio else None, 'length': 5}
        start, end = 0., None
        if kind == 'image':
            with Image.open(path) as image:
                check_dimensions(*image.size)
                if image.width * image.height * 36 > 1024 ** 3:
                    raise ValueError('RefMod image exceeds the decode estimate; use a downscaled source.')
            kwargs['ref_images'] = {'ref_image_1': load_image(value, root)}
        else:
            start, end = dasiwa_media_crop(item, path, 'video' if kind == 'video' and (visual or has_attached) else 'audio')
            crop = {'trim_start': start, 'trim_end': end}
            audio_path = resolve_input_path(attached, root) if has_attached else path
            if audio:
                dasiwa_media_crop(crop, audio_path, 'audio')
            if visual:
                frames = dasiwa_decode_media(path, 'video', start, end)
                count = int(frames.shape[0])
                if count < 5:
                    raise ValueError('RefMod video needs at least five decoded frames.')
                retained = count - (count - 5) % 17
                if retained != count:
                    warnings.append(f'Item {len(refs) + 1}: H3 frame grid kept {retained}/{count} frames.')
                kwargs.update(length=count + (5 - count) % 17, ref_videos={'ref_video_1': frames})
                del frames
            if audio:
                sound = dasiwa_decode_media(audio_path, 'audio', start, end)
                key, slot = ('ref_video_audios', 'ref_video_audio_1') if visual else ('ref_audios', 'ref_audio_1')
                kwargs[key] = {slot: sound}
                del sound
        blocks = dasiwa_capture(kwargs)
        provenance = {'source': value, 'source_item_id': str(item.get('id', len(refs))),
                      'source_enabled': item.get('enabled', True), 'description': str(item.get('description', item.get('prompt', ''))),
                      'source_media_mode': mode, 'trim_start': start, 'trim_end': end,
                      'source_role': item.get('forge_role'), 'source_audio': attached if has_attached else None}
        if kind == 'video' and visual:
            provenance.update(source_frame_rate=24, source_frame_count=count, encoded_frame_count=retained)
        for block in blocks:
            for key, member_kind in [('latent', 'image' if kind == 'image' else 'video'), ('audio_latent', 'audio')]:
                if block.get(key) is not None:
                    dasiwa_interrupted()
                    refs.append((block[key].detach().cpu().contiguous().clone(), {**provenance, 'kind': member_kind}))
        del kwargs, blocks
        if len(refs) > 256:
            raise ValueError('RefMod export exceeds 256 members.')
    for row in rows:
        dasiwa_interrupted()
        if not isinstance(row, dict):
            raise ValueError('Saved RefMod row must be an object.')
        if not row.get('name'):
            continue
        loaded = load_refmods(row['name'])
        selection = row.get('members')
        if 'members' in row:
            if not isinstance(selection, list) or not 1 <= len(selection) <= len(loaded):
                raise ValueError('Select a nonempty bounded RefMod members array.')
            indices = []
            for member in selection:
                if (not isinstance(member, dict) or type(member.get('index')) is not int or
                        not 0 <= member['index'] < len(loaded) or member['index'] in indices or
                        not isinstance(member.get('description'), str)):
                    raise ValueError('Invalid RefMod member selection or description.')
                indices.append(member['index'])
        else:
            selection = [{'index': index} for index in range(len(loaded))]
        for member in selection:
            dasiwa_interrupted()
            latent, metadata = loaded[member['index']]
            exported_metadata = {**metadata, 'source_refmod': row['name']}
            if 'description' in member:
                exported_metadata['description'] = member['description']
            elif 'description' in row:
                exported_metadata['description'] = str(row['description'])
            refs.append((latent, exported_metadata))
            if len(refs) > 256:
                raise ValueError('RefMod export exceeds 256 members.')
    if not refs:
        raise ValueError('RefMod export requires nonempty references.')
    return refs, warnings


class DaSiWaH3RefModExport:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'timeline_data': ('STRING', {'default': '{}', 'tooltip': 'Selected Director snapshot, including disabled/hidden references.'}),
            'request_id': ('STRING', {'default': '', 'tooltip': 'Unique export request ID to prevent cached button results.'}),
            'name': ('STRING', {'default': '', 'tooltip': 'Safe basename without paths; empty generates a unique director name. Never overwrites.'}),
            'description': ('STRING', {'default': '', 'tooltip': 'Descriptive bundle metadata, not a training instruction.'}),
            'concept_type': (['generic', 'identity', 'style', 'motion', 'audio'], {'tooltip': 'Metadata label: general, identity, style, motion, or audio; does not alter encoding.'}),
        }, 'optional': {
            'vae': ('VAE', {'tooltip': 'MiniMax H3 video VAE, required for raw images/video.'}),
            'audio_vae': ('VAE', {'tooltip': 'MiniMax H3 audio VAE, required for raw audio or selected video sound.'}),
        }}

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('refmod_name',)
    FUNCTION = 'create_refmod'
    CATEGORY = 'DaSiWa/MiniMax H3'
    OUTPUT_NODE = True

    def create_refmod(self, timeline_data, request_id, name='', description='', concept_type='generic',
                      vae=None, audio_vae=None):
        dasiwa_interrupted()
        if not isinstance(request_id, str) or not request_id:
            raise ValueError('A nonempty RefMod request_id is required.')
        dasiwa_validate_name(name)
        refs, warnings = dasiwa_export_refs(timeline_data, vae, audio_vae)
        name = dasiwa_save_bundle(refs, warnings, name, description, concept_type)
        counts = {kind: sum(meta['kind'] == kind for _, meta in refs) for kind in ('image', 'video', 'audio')}
        message = f'Saved {name}.safetensors ({len(refs)} members); {counts}'
        return {'ui': {'dasiwa_refmod_export': [{'name': name, 'members': len(refs),
                 'counts': counts, 'warnings': warnings, 'message': message}]}, 'result': (name,)}
