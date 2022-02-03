
from email.mime import audio
import opensmile
from glob import glob 
import numpy as np 
from random import sample
import wave
from glob import glob
from scipy.io import wavfile
from scipy.signal import resample

audio_files = np.load("audio_files_in_order_no_neutral.npy").tolist()
audio_files = [str(item) for item in audio_files]

smile = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.eGeMAPSv01b,
                    feature_level=opensmile.FeatureLevel.LowLevelDescriptors
                )
# FeatureSet.eGeMAPSv01b FeatureLevel.LowLevelDescriptors (496, 23)

container = np.empty(shape=(0,47))
print(container.shape)
print(len(audio_files))

for audio_file in audio_files:

    # First get features
    y = smile.process_file(audio_file).to_numpy()
    mean_y = np.mean(y,axis=0)
    std_y = np.std(y,axis=0)

    # Now read wav 
    ifile = wave.open(audio_file)
    samples = ifile.getnframes()
    audio = ifile.readframes(samples)

    # Convert buffer to float32 using NumPy                                                                                 
    audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
    audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

    # Normalise float32 array so that values are between -1.0 and +1.0                                                      
    max_int16 = 2**15
    audio_normalised = audio_as_np_float32 / max_int16
    audio_normalised = resample(audio_normalised, int(samples/8))
    
    # Now calculate silence 
    Nt = audio_normalised.shape[0]
    RMS = np.mean(np.square(audio_normalised))
    th = 0.3 * RMS
    Ns = np.sum(audio_normalised < th)
    transformed_feature = np.hstack((mean_y, std_y, np.array(Ns/Nt)))

    container = np.vstack((container,np.transpose(transformed_feature)))
    print(container.shape)


np.save('MELD_test_data_no_neutral',container)
