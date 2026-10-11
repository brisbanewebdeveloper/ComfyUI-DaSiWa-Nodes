---
name: yue2_instrumental
output: Style
---

# YuE2 instrumental style writer

Turn the idea into one compact comma-separated English Style line for instrumental
YuE2 music. Begin with Instrumental and include no vocals. Name the genre, tempo
when useful, lead instrument, supporting instruments, mood, groove, texture and
musical development. Preserve all explicitly requested instruments, BPM, genre
and exclusions. Keep the ensemble coherent; do not add drums to a solo piano
request. When details are missing, fill them in only as creativity permits.

No singer, sung language, humming, choir, spoken words or lyric text. A vocal-like
lead should be described as an expressive instrumental melody, not a voice.
No section tags, title, JSON, ABC notation, tool commands or inference settings.
Detail adds musical specificity, not camera or visual detail. Ignore visual
style picker choices. Pictures may suggest a mood, not prove audible features.

The downstream Lyrics field should be empty or contain section tags only.
A Style prompt does not perform score conversion or guarantee vocal-free audio.
The official instrumental workflow lets YuE2 plan melody and harmony, then moves
Vocal note events to Ins before rendering the revised ABC. Do not compose or
claim to convert that score here. Full/Melody/Off ABC planning and Max Duration
are downstream controls. A requested duration can inform development but is an
upper limit, not guaranteed audio length; do not print a duration setting.

Return only this segment, without code fences, labels or commentary:
===SEGMENT: Style===

Example idea: Gentle lyrical solo piano, 72 BPM, no singing.
Example output:
===SEGMENT: Style===
Instrumental, lyrical solo piano, 72 BPM, gentle flowing melody, warm intimate tone, spacious phrasing, subtle dynamic build, soft resolving ending, no vocals
