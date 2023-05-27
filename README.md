# Opera singing dataset 2023
All the .wav files are ignored due to the size, to download the whole dataset, please go to this link: TODO

## Where does this dataset come from?
This data set is a combination of this two following datasets:
- Singing Voice Audio Dataset
  - Main source of this dataset.
  - > D. A. A. Black, M. Li, and M. Tian, “Automatic Identification of Emotional Cues in Chinese Opera Singing,” in 13th Int. Conf. on Music Perception and Cognition (ICMPC-2014), 2014, pp. 250–255.
- Jingju (Beijing opera) Phoneme Annotation
  - Use the csv of this dataset to correct some annotaion errors in the former one.
  - Add new audios from this dataset.
    - 2 new laosheng songs (one audio for each)
    - 6 new audios for exsiting laosheng songs
    - for dan, TODO
  - > Rong Gong, Rafael Caro Repetto, & Yile Yang. (2017). Jingju a cappella singing dataset [Data set]. Zenodo. http://doi.org/10.5281/zenodo.344932

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