from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os
class UrbanSoundDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate):
        self.anotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        
    def __len__(self):
        return len(self.anotations)
    
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sameple_rate = torchaudio.load(audio_sample_path)
        return signal, label
    
    def _get_audio_sample_path(self, index):
        fold = f"fold{self.anotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.anotations.iloc[index, 0])
        return path
    
    def _get_audio_sample_label(self, index):
        return self.anotations.iloc[index, 6]
    
    
    
    
if __name__ == "__main__":
    ANNOTATIONS_FILE = "./data/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "./data/UrbanSound8K/audio"
    SAMPLE_RATE = 16000
    
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = 1024,
        hop_length = 512,
        n_mels=64
    )
                
    dataset = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE)
    print(f"There are {len(dataset)} samples in the dataset")
    
    signal, label = dataset[0]
    print(signal)
    a=1