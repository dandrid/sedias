import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import os


class RAVDESS:

    def __init__(self):
        self.separator = "-"
        self.audio_only = "03"
        self.voice_channel = "01"
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
        self.statements = np.array([("Kids_are_talking_by_the_door", "01"), ("Dogs_are_sitting_by_the_door", "02")])
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
        return

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
        
        label = 0 # label index
        suffix = 1 # suffix index
        
        print("loading audio files and enriching them")
        for emotion in self.emotions:
            for emotion_intensity in self.emotion_intensities:
                for statement in self.statements:
                    for repetition in self.repetitions:
                        for actor in self.actors:
                            actor_gender = actor[label]
                            current_file = "data/RAVDESS/Actor_" + actor[suffix] + "/" + self.addSepartors([ # construct the file name
                                    self.audio_only, 
                                    self.voice_channel, 
                                    emotion[suffix], 
                                    emotion_intensity[suffix], 
                                    statement[suffix], 
                                    repetition, 
                                    actor[suffix]]) + ".wav"
                            exists = os.path.isfile(current_file) # if the file exists
                            if exists:
                                audio_data, samplerate = sf.read(current_file) # load the audio
                                data.append([emotion[label], emotion_intensity[label], statement[label], repetition, actor[suffix], actor_gender, samplerate, audio_data]) # add original audio to the data
                                
                                noised_audio_datas = self.createAudioWithNoise(audio_data) # create noised version of the original audio
                                for noised_audio_data in noised_audio_datas:
                                    data.append([emotion[label], emotion_intensity[label], statement[label], repetition, actor[suffix], actor_gender, samplerate, noised_audio_data]) # add noised audio to enrich data
                                
        print("number of samples: " + str(len(data)))

        df = pd.DataFrame(  # contruct DataFrame from data
            { 
                'emotion'           : pd.Categorical([row[0] for row in data]),
                'emotion_intensity' : pd.Categorical([row[1] for row in data]),
                'statement'         : pd.Categorical([row[2] for row in data]),
                'repetition'        : pd.Categorical([row[3] for row in data]),
                'actor'             : pd.Categorical([row[4] for row in data]),
                'actor_gender'      : pd.Categorical([row[5] for row in data]),
                'samplerate'        : pd.Categorical([row[6] for row in data]),
                'audio_data'        : pd.Series([row[7] for row in data])
            })

        # one-hot encode columns
        df = pd.get_dummies(df, columns=["emotion", "emotion_intensity", "statement", "repetition", "actor", "actor_gender", "samplerate"])
        return df

    def addSepartors(self, fragments):
        return "-".join(fragments)


class EMODB:
    def __init__(self):
        self.actors = np.array([
            ("male", "03"), 
            ("female", "08"), 
            ("female", "09"), 
            ("male", "10"), 
            ("male", "11"), 
            ("male", "12"),
            ("female", "13"),
            ("female", "14"), 
            ("male", "15"), 
            ("female", "16")])
        self.statements = np.array([
            ("Der_Lappen_liegt_auf_dem_Eisschrank.", "a01"), 
            ("Das_will_sie_am_Mittwoch_abgeben.", "a02"),
            ("Heute_abend_könnte_ich_es_ihm_sagen.", "a04"), 
            ("Das_schwarze_Stück_Papier_befindet_sich_da_oben_neben_dem_Holzstück.", "a05"),
            ("In_sieben_Stunden_wird_es_soweit_sein.", "a07"), 
            ("Was_sind_denn_das_für_Tüten,_die_da_unter_dem_Tisch_stehen?", "b01"),
            ("Sie_haben_es_gerade_hochgetragen_und_jetzt_gehen_sie_wieder_runter.", "b02"), 
            ("An_den_Wochenenden_bin_ich_jetzt_immer_nach_Hause_gefahren_und_habe_Agnes_besucht.", "b03"),
            ("Ich_will_das_eben_wegbringen_und_dann_mit_Karl_was_trinken_gehen.", "b09"), 
            ("Die_wird_auf_dem_Platz_sein,_wo_wir_sie_immer_hinlegen.", "b10")])
        self.emotions = np.array([
            ("neutral", "N"), 
            ("anger", "W"), # Wut
            ("boredom", "L"), # Langeweile
            ("disgust", "E"), # Ekel
            ("anxiety/fear", "A"), # Angst
            ("happiness", "F"), # Freude
            ("sadness", "T"), # Trauer
            ])
        self.repetitions = np.array(["a", "b", "c", "d", "e", "f"])
        return
    
    
    def createDataFrame(self):
        data = []

        label = 0 # label index
        suffix = 1 # suffix index

        print("loading audio files and converting them")
        for actor in self.actors:
            actor_gender = actor[label]
            for statement in self.statements:
                for emotion in self.emotions:
                    for repetition in self.repetitions:

                        current_file = "data/emodb/wav/" + actor[suffix] + statement[suffix] + emotion[suffix] + repetition + ".wav"
                        exists = os.path.isfile(current_file) # if the file exists
                        if exists:
                            audio_data_original, samplerate_original = sf.read(current_file) # load the audio
                            samplerate_48k = 48000
                            audio_data_48k = librosa.resample(audio_data_original, samplerate_original, samplerate_48k)
                            data.append([emotion[label], statement[label], repetition, actor[suffix], actor_gender, samplerate_48k, audio_data_48k]) # add original audio to the data
        
        print("number of samples: " + str(len(data)))
        
        df = pd.DataFrame(  # contruct DataFrame from data
            { 
                'emotion'           : pd.Categorical([row[0] for row in data]),
                'statement'         : pd.Categorical([row[1] for row in data]),
                'repetition'        : pd.Categorical([row[2] for row in data]),
                'actor'             : pd.Categorical([row[3] for row in data]),
                'actor_gender'      : pd.Categorical([row[4] for row in data]),
                'samplerate'        : pd.Categorical([row[5] for row in data]),
                'audio_data'        : pd.Series([row[6] for row in data])
            })

        # one-hot encode columns
        df = pd.get_dummies(df, columns=["emotion", "statement", "repetition", "actor", "actor_gender", "samplerate"])
        return df


pd.set_option("max_columns" , None)

print("RAVDESS:")
ravdess = RAVDESS()
df_ravdess = ravdess.createDataFrame()
print("Date frame size: " + str(len(df_ravdess)))
print(df_ravdess.head())

print("EMODB:")

emodb = EMODB()
df_emodb = emodb.createDataFrame()
print("Date frame size: " + str(len(df_emodb)))
print(df_emodb.head())