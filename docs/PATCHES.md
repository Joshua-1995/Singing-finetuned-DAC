# Compatibility patches

The DAC training code (2023) predates current PyTorch. A Blackwell GPU (sm_120) needs
the CUDA 12.8+ PyTorch build, which removed/changed some APIs the older
`descript-audiotools` / `argbind` rely on. Three small patches make training work.
Re-apply them if you reinstall the affected packages.

### 1. `argbind` — PyTorch union type annotation
`site-packages/argbind/argbind.py`, in the argument-builder loop: PyTorch 2.x annotates
optimizer `lr` as `float | torch.Tensor`, which argparse can't use as a `type`. Fall back
to the default value's type (or first concrete scalar member) when the annotation is a
`typing.Union` / PEP-604 union.

### 2. `audiotools` — torchaudio legacy I/O removed
`site-packages/audiotools/core/util.py`, `info()`: `torchaudio>=2.9` removed
`torchaudio.info` and `torchaudio.backend`. Fall back to `soundfile.info`:
```python
try:
    i = torchaudio.info(str(audio_path))
    return Info(sample_rate=i.sample_rate, num_frames=i.num_frames)
except Exception:
    import soundfile as sf
    si = sf.info(str(audio_path))
    return Info(sample_rate=si.samplerate, num_frames=si.frames)
```

### 3. DAC `train.py` — length alignment before spectral losses
When the training segment length is an exact multiple of `hop_length` (320), the decoder
can return a few fewer samples than the input, so mel/STFT losses hit a shape mismatch.
In both `train_loop` and `val_loop`, trim recon and target to the common length before
computing losses:
```python
n = min(out["audio"].shape[-1], signal.audio_data.shape[-1])
signal.audio_data = signal.audio_data[..., :n]
recons = AudioSignal(out["audio"][..., :n], signal.sample_rate)
```
