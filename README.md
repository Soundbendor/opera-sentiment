# Opera singing dataset 2023
## All the .wav files are ignored
## explanation of the yaml meta data
```yaml
emotion:
- emotion_1
- emotion_2
- emotion_3
emotion_binary: 1 or 0 (positive or negative)
files:
  wav00:
    file_dir: dir to this wav file
    info:
      bit_rate: bit_rate
      channel_number: number of the channels
      duration: duration
      if_a_cappella: True or False (if it is a-cappella)
      sample_rate: sample_rate
    singer:
      bio_gender: bio_gender of the person record this audio
      id: singer_id
      level:  professional or amateur
      name: Singer name
  wav01:
    # ... ...
language: ch or we (chinese or western language)
scene:
  english: english translation for the scene title
  original: original scene title
  phonetic: only for chinese songs, so it will be pinyin for it
song_dir: dir to this song
song_id: id for this song
title:
  english: english translation for the song title
  original: original song title
  phonetic: only for chinese songs, so it will be pinyin for it
```