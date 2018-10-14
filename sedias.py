import numpy as np
import pandas as pd
import soundfile as sf
import os


class RAVDESS:

    def __init__(self):
        self.separator="-"
        self.audio_only="03"
        self.voice_channel="01"
        self.emotions = np.array([
            ("neutral", "01"), 
            ("calm", "02"), 
            ("happy", "03"), 
            ("sad", "04"), 
            ("angry", "05"), 
            ("fearful", "06"), 
            ("disgust", "07"), 
            ("surprised", "08")])
        self.emotion_intensities = np.array([("normal", "01"), ("strong", "02")])
        self.statements = np.array([("Kids are talking by the door", "01"), ("Dogs are sitting by the door", "02")])
        self.repetitions = np.array(["01", "02"])
        self.actors = np.array([
            ("male", "01"), 
            ("female", "02"), 
            ("male", "03"), 
            ("female", "04"), 
            ("male", "05"), 
            ("female", "06"), 
            ("male", "07"), 
            ("female", "08"),
            ("male", "09"),
            ("female", "10"),
            ("male", "11"), 
            ("female", "12"), 
            ("male", "13"), 
            ("female", "14"), 
            ("male", "15"), 
            ("female", "16"), 
            ("male", "17"), 
            ("female", "18"),
            ("male", "19"),
            ("female", "20"),
            ("male", "21"), 
            ("female", "22"), 
            ("male", "23"), 
            ("female", "24")])

    # Add noise to the audio to create more samples
    def createAudioWithNoise(self, data):
        mean, std = data.mean(), data.std()
        noised_data = []

        # Add noise based on the standard deviation
        for noise_intensity in np.arange(0.1, 0.3, 0.1):
            noise = np.random.normal(mean, std*noise_intensity, data.shape) 
            noised_data.append(data + noise)
        return noised_data

    # Creates the Data Frame for the RAVDESS
    def createDataFrame(self):
        data = []
        
        print("loading audio files and enriching them")
        for emotion in self.emotions:
            for emotion_intensity in self.emotion_intensities:
                for statement in self.statements:
                    for repetition in self.repetitions:
                        for actor in self.actors:
                            actor_gender = actor[0]
                            currentFile = "data/RAVDESS/Actor_" + actor[1] + "/" + self.addSepartors([self.audio_only, self.voice_channel, emotion[1], emotion_intensity[1], statement[1], repetition, actor[1]]) + ".wav"
                            exists = os.path.isfile(currentFile)
                            if exists:
                                audio_data, samplerate = sf.read(currentFile)
                                data.append([emotion[0], emotion_intensity[0], statement[0], repetition, actor[1], actor_gender, samplerate, audio_data])
                                
                                noised_audio_datas = self.createAudioWithNoise(audio_data)
                                for noised_audio_data in noised_audio_datas:
                                    data.append([emotion[0], emotion_intensity[0], statement[0], repetition, actor[1], actor_gender, samplerate, noised_audio_data])
                                
        print("number of samples: " + str(len(data)))
        
        pd.set_option('display.max_columns', 500)
        df = pd.DataFrame(
            { 
                'emotion'           : pd.Categorical([row[0] for row in data]),
                'emotion intensity' : pd.Categorical([row[1] for row in data]),
                'statement'         : pd.Categorical([row[2] for row in data]),
                'repetition'        : pd.Categorical([row[3] for row in data]),
                'actor'             : pd.Categorical([row[4] for row in data]),
                'actor gender'      : pd.Categorical([row[5] for row in data]),
                'samplerate'        : pd.Categorical([row[6] for row in data]),
                'audio data'        : pd.Series([row[7] for row in data])
            })
        return df

    def addSepartors(self, fragments):
        return "-".join(fragments)


ravdess = RAVDESS()
df = ravdess.createDataFrame()
print("Date frame size: " + str(len(df)))