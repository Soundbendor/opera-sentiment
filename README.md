# Opera singing dataset 2023

## A multi-label dataset for opera singing

All the .wav files are ignored due to the size, to download the whole dataset, please go to this link: TODO

## Where does this dataset come from?
This data set is a combination of these intelligence contributions:
- Singing Voice Audio Dataset<sup>[1]</sup>
  - Most audio source and label annotation are from this dataset.
- Jingju (Beijing opera) Phoneme Annotation<sup>[2]</sup>
  - Use the csv file of this dataset to correct some annotaion errors in the former one.
  - Add new audios from this dataset.
    - 2 new laosheng songs (one audio for each)
    - 6 new audios for exsiting laosheng songs
    - 4 new dan songs (one audio for each)
    - 2 new audio for exsiting dan songs
    - correct few mis-classified songs
- Label corrections
  - Label corrected by Shengxuan Wang ([shawn120](https://github.com/shawn120))
  - Correction rely on the knowledge from Shengxuan Wang, Rong Gong et al. <sup>[2]</sup>, and online searching.
- Only keep opera data, removed all the non-opera data from previous dataset (e.g. modern songs). 
- Future TODO: Add in more western opera to balance out the language inbalance.

> [1] D. A. A. Black, M. Li, and M. Tian, “Automatic Identification of Emotional Cues in Chinese Opera Singing,” in 13th Int. Conf. on Music Perception and Cognition (ICMPC-2014), 2014, pp. 250–255. 
> 
> [2] Rong Gong, Rafael Caro Repetto, & Yile Yang. (2017). Jingju a cappella singing dataset [Data set]. Zenodo. http://doi.org/10.5281/zenodo.344932

## explanation of the yaml meta data
```yaml
emotion:
- emotion_1
- emotion_2
- emotion_3
emotion_binary: 1 or 0 (positive or negative), -1 represents to-be-labeled
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
role_gender: the gender of the character in this song/scene
scene:
  english: english translation for the scene title
  original: original scene title (for chinese hanzi, will shown as unicode)
  phonetic: only for chinese songs, so it will be pinyin for it
singing_type:
  role: only for jingju, laosheng, dan, ... or TBD
  singing: jingju/yuju/chinese singing/... or TBD
song_dir: dir to this song
song_id: id for this song
song_size: how many audio in this one single song
title:
  english: english translation for the song title
  original: original song title (for chinese hanzi, will shown as unicode)
  phonetic: only for chinese songs, so it will be pinyin for it
```