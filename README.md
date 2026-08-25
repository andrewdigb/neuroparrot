# Project Parrot
Project in several phases to ultimately build an app that translate from animal vocalisms to human language using Neural Networks

---

## Bio-Digital Vocalism Translator (multi-species neural engine)

The project now ships a working neural-network translator that generalizes
**beyond parrots to any animal** you have data for (dogs, birds, cows, frogs, …).
It lives in the [`bio_vocalism_translator`](bio_vocalism_translator/) Python package.

### How it works
1. **Features** — each clip is turned into a normalized log-mel spectrogram, a
   representation that works across the very different frequency ranges of
   different animals.
2. **Multi-task CNN** — one shared convolutional backbone feeds two output
   heads: a **species** head (which animal) and an **intent** head (what it
   means). Sharing the backbone lets acoustic features transfer across species.
3. **Species-aware decoding** — at inference the intent head is masked to only
   the intents valid for the detected species, so translations stay coherent.
4. **Data-driven taxonomy** — species and intents live in a JSON taxonomy, not
   in code, so adding a new animal is a data change plus retraining — no
   architectural changes.

### Install
```bash
pip install -r requirements.txt
```

### Dataset layout
```
dataset/
  cat/happy/*.wav      cat/angry/*.wav ...
  dog/alert/*.wav      dog/playful/*.wav ...
  frog/mating_call/*.wav ...
```
(Filename layout `species__intent__free-text.wav` is also supported.)

### Train
```bash
python -m bio_vocalism_translator train --data ./dataset --out ./translator_model
```

### Translate any animal
```bash
python -m bio_vocalism_translator translate --model ./translator_model --audio clip.wav
```

Or from Python:
```python
from bio_vocalism_translator import Translator

translator = Translator.load("translator_model")
result = translator.translate("clip.wav")
print(result.sentence)   # e.g. "I'm hungry, feed me."
```

### Add a new animal
```python
from bio_vocalism_translator import TAXONOMY

TAXONOMY.add_intent("elephant", "greeting", "Hello, I recognize you.")
TAXONOMY.add_intent("elephant", "distress", "I'm in danger!")
# add clips under dataset/elephant/greeting/ etc., then retrain
```

### Inspect what the model knows
```bash
python -m bio_vocalism_translator taxonomy
```

---





>Help me to collect cat sounds either by recording your cat and droping the file in the drop box below or capturing sounds from Youtube videos using online [mp3fy here](https://mp3fy.com/) or any recording app on mobile (see how to name the file below)

![CuteCat](https://www.wtfclub.net/wp-content/uploads/2017/12/unnamed.jpg)


## Introduction
- Phase 1: Collection of cat sound samples in a specific format with annotation of the cat behaviour following a prescribed method for each predesignated designed behavours (eg hungry, don't bother me, etc.)
- Phase 2: Data manipulation and preparation for deep learning
- Phase 3: Machine learning on initial sample (this could branch out depending on machine learning algorith used)
- Phase 4: App development to apply the learned model to real situation and use language recognition to speak cat language

Drop your sound files here according to instructions on link above:

<iframe src="https://onedrive.live.com/embed?cid=EB02B52BD1CE79FD&resid=EB02B52BD1CE79FD%212644&authkey=AOroewqAhmmou_0" width="165" height="128" frameborder="0" scrolling="no"></iframe>


## Phase 1 Collecting cat sounds

This task is probably the most challenging task. 

![FlowChart1](flowchart1.PNG)


### Collecting raw sounds
This requires either volunteers to record their own cats, which could be fun, though time consuming or screening online resources such as Youtube and then post processing the audio files.
In any case wave files need to be collected.

Submission of cat sounds to: [CatSounds Drop Box](https://1drv.ms/f/s!Av15ztErtQLrlFSNvhIzIsAi3KJy)
 
Tools to collect cats sound : [mp3fy here](https://mp3fy.com/) or any recording app on mobile


Naming of raw cat sounds files: Cat sounds should be named as follows:
   "**SoundClass_Sex_Age_Name_FistNameOfWhoIsSubmitting_Country_Source**"


where,
  - **SoundClass** is one of the following: 

 |Sound Class       | Meaning             
 |:------------------|:--------------------:|
 |Growling | I am warning you |
 |MomoMooh | I am angry |
 |Hissing | Leave me alone |
 |Nyaaan | Want to fight? |
 | MeowMeow | I am happy | 
 | Chatting | I want to hunt/play |
 |GyaGyaGya | I want a cat-mate | 
 | KittenMiyouMiyou| Mama! Mama! | 
 | Miyoou | It hurts/hungry | 
 | Purring | I am comfortable |
 
  - ***Sex** is either M for male of F for female (if known)*
  - ***Age** is age of cat (approximately if known, eg 3m or 2Y, etc.) (if known)*
  - ***Name** is name of cat (if known)*
  - **FirstNameOfWhoIsSubmitting** is name of who is sending the file
  - ***Country** is country where the cat is. (this will be used to find out if cats in different countries understand each others) (if known)
  - **Source** should be *Web* or *Recorded*


### Classification of sounds
Hear the cats and what they say at: [Link here](https://www.mnn.com/family/pets/stories/cat-sounds-and-what-they-mean)


<iframe width="540" height="360" src="https://www.youtube.com/embed/nX1YzS_CYIw" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>


After a bit of research, it is proposed to follow the work done by JH Lee [Domestic Cat Sound Classification Using Transfer Learning](http://www.ijfis.org/journal/view.html?uid=827&&vmd=Full).
The following site shows examples of sound within each classis used as a guide for classification of site
The research proposes to classify cats sounds in 10 classes as follows:


![Cat classes](talkingcat.gif)


## Phase 2
Data filtering [HOLD: not decided yet]

It is curretly proposed to use Librosa library in python for [MFCCS](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) feature extractions


## Phase 3
It is currently proposed to use Supervised Training Convulational [Neural Networks](https://en.wikipedia.org/wiki/Neural_network) [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network) with [Tensorflow and Keras](https://www.tensorflow.org/guide/keras) given its accuracy and portability. The model will be developed in Python hand the model will be saved for further used in other languages.
The target accuracy is 70%.


## Phase 4
Simple app will be developed using [MIT App Inventor](http://appinventor.mit.edu/explore/) to record cat and translate to human and vice versa.

<!--stackedit_data:
eyJoaXN0b3J5IjpbOTYyNzg5NzU0LC0xNjEzMTQ1NTU0LDEyND
czNjc0NjIsLTYyNzY2MDE4NiwtNTQ4NDQ2MzYyLDE2MTA3OTc5
ODgsLTIzOTEyMjkxLC0xMjA0NDcxMzQ1LDc4NDI3ODQwNCwxNT
IwMDY0NzU1LC0xNjIxMjMzMywtMTc0Nzc5NTA5MSwxNjc2NzA0
NDY4XX0=
-->
