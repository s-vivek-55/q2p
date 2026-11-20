import argparse
import asyncio
import hashlib
import json
import math
import wave
from pathlib import Path

import av
import edge_tts
import numpy as np


ROOT = Path(__file__).resolve().parent
AUDIO = ROOT / "audio_revised"
SAMPLE_RATE = 48000


def load_storyboard() -> dict:
    storyboard = json.loads((ROOT / "storyboard.json").read_text(encoding="utf-8"))
    position = 0.0
    for scene in storyboard["scenes"]:
        assert abs(scene["start"] - position) < 0.001, scene["id"]
        assert scene["duration"] > 0
        position += scene["duration"]
    assert abs(position - storyboard["duration"]) < 0.001
    assert position <= 240
    return storyboard


def decode_audio(source: Path, tempo: float = 1.0) -> np.ndarray:
    resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
    chunks = []
    graph = None

    def consume(frame) -> None:
        nonlocal graph
        if tempo == 1.0:
            if frame is not None:
                chunks.append(frame.to_ndarray().reshape(-1))
            return
        if graph is None:
            if frame is None:
                return
            graph = av.filter.Graph()
            buffer = graph.add(
                "abuffer",
                f"time_base=1/{SAMPLE_RATE}:sample_rate={SAMPLE_RATE}:sample_fmt=flt:channel_layout=mono",
            )
            adjustment = graph.add("atempo", str(tempo))
            sink = graph.add("abuffersink")
            buffer.link_to(adjustment)
            adjustment.link_to(sink)
            graph.configure()
        graph.push(frame)
        while True:
            try:
                chunks.append(graph.pull().to_ndarray().reshape(-1))
            except (av.error.BlockingIOError, av.error.EOFError):
                break

    with av.open(str(source)) as container:
        for decoded in container.decode(audio=0):
            for frame in resampler.resample(decoded):
                consume(frame)
    for frame in resampler.resample(None):
        consume(frame)
    consume(None)
    return np.concatenate(chunks).astype(np.float32)


def write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1 if samples.ndim == 1 else samples.shape[1])
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm.tobytes())


async def synthesize(scene: dict, voice: str, rate: str, pitch: str, semaphore: asyncio.Semaphore) -> None:
    target = AUDIO / f"{scene['id']}.mp3"
    metadata = AUDIO / f"{scene['id']}.jsonl"
    fingerprint_path = AUDIO / f"{scene['id']}.sha256"
    fingerprint = hashlib.sha256(
        (scene["narration"] + voice + rate + "|" + pitch).encode()
    ).hexdigest()
    if (
        target.exists()
        and metadata.exists()
        and fingerprint_path.exists()
        and fingerprint_path.read_text() == fingerprint
    ):
        return
    async with semaphore:
        communicator = edge_tts.Communicate(
            scene["narration"], voice=voice, rate=rate, pitch=pitch
        )
        await communicator.save(str(target), str(metadata))
        fingerprint_path.write_text(fingerprint)
        print(f"Narrated: {scene['id']}", flush=True)


async def synthesize_all(storyboard: dict) -> None:
    AUDIO.mkdir(exist_ok=True)
    semaphore = asyncio.Semaphore(3)
    await asyncio.gather(
        *(
            synthesize(scene, storyboard["voice"], storyboard["voice_rate"], storyboard["voice_pitch"], semaphore)
            for scene in storyboard["scenes"]
        )
    )


def trim_voice(samples: np.ndarray) -> tuple[np.ndarray, float]:
    active = np.flatnonzero(np.abs(samples) > 0.0025)
    assert active.size, "Speech audio is silent."
    first = max(0, int(active[0]) - int(SAMPLE_RATE * 0.06))
    last = min(len(samples), int(active[-1]) + int(SAMPLE_RATE * 0.10))
    return samples[first:last], first / SAMPLE_RATE


def ambient_score(duration: float) -> np.ndarray:
    count = round(duration * SAMPLE_RATE)
    music = np.zeros((count, 2), dtype=np.float32)
    chords = [(48, 55, 59, 64), (45, 52, 55, 59), (41, 48, 52, 57), (43, 50, 55, 60)]
    for chord_index, start in enumerate(np.arange(-2.0, duration, 8.0)):
        length = 10.0
        time = np.arange(round(length * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
        envelope = np.sin(np.pi * time / length) ** 1.8
        chord_audio = np.zeros((len(time), 2), dtype=np.float32)
        for note_index, midi in enumerate(chords[chord_index % len(chords)]):
            frequency = 440 * 2 ** ((midi - 69) / 12)
            tone = (
                np.sin(2 * np.pi * frequency * time)
                + 0.15 * np.sin(2 * np.pi * frequency * 2 * time + 0.3)
                + 0.22 * np.sin(2 * np.pi * frequency * 1.0012 * time + 0.6)
            ) * envelope * 0.075
            pan = 0.22 + note_index * 0.18
            chord_audio[:, 0] += tone * math.sqrt(1 - pan)
            chord_audio[:, 1] += tone * math.sqrt(pan)
        offset = round(start * SAMPLE_RATE)
        source_start = max(0, -offset)
        destination_start = max(0, offset)
        amount = min(len(chord_audio) - source_start, count - destination_start)
        music[destination_start:destination_start + amount] += chord_audio[source_start:source_start + amount]
    for note_index, start in enumerate(np.arange(2, duration - 3, 2.5)):
        time = np.arange(SAMPLE_RATE * 2, dtype=np.float32) / SAMPLE_RATE
        midi = (72, 76, 79, 74, 71, 76, 72, 67)[note_index % 8]
        frequency = 440 * 2 ** ((midi - 69) / 12)
        envelope = (1 - np.exp(-time * 45)) * np.exp(-time * 2.7)
        tone = np.sin(2 * np.pi * frequency * time) * envelope * 0.028
        offset = round(start * SAMPLE_RATE)
        music[offset:offset + len(tone), 0] += tone * 0.8
        music[offset:offset + len(tone), 1] += tone * 0.6
    peak = np.max(np.abs(music))
    music *= 0.46 / max(peak, 1e-6)
    fade_in = np.minimum(np.arange(count) / (SAMPLE_RATE * 2.5), 1)
    fade_out = np.minimum((count - 1 - np.arange(count)) / (SAMPLE_RATE * 4), 1)
    music *= (fade_in * fade_out)[:, None]
    return music


def subtitle_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds_part, milliseconds_part = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds_part:02},{milliseconds_part:03}"


def assemble_audio(storyboard: dict) -> None:
    narration = np.zeros(round(storyboard["duration"] * SAMPLE_RATE), dtype=np.float32)
    report = []
    captions = []
    excessive = []
    for scene in storyboard["scenes"]:
        raw, trim_start = trim_voice(decode_audio(AUDIO / f"{scene['id']}.mp3"))
        length = len(raw) / SAMPLE_RATE
        allowance = scene["duration"] - 0.45
        tempo = max(1.0, length / allowance)
        report.append(
            {"scene": scene["id"], "slot": scene["duration"], "speech": round(length, 3), "tempo": round(tempo, 4)}
        )
        if tempo > 1.10:
            excessive.append(report[-1])
            continue
        if tempo > 1:
            temporary = AUDIO / f"{scene['id']}_trimmed.wav"
            write_wav(temporary, raw)
            samples = decode_audio(temporary, tempo)
        else:
            samples = raw
        active = samples[np.abs(samples) > 0.01]
        rms = float(np.sqrt(np.mean(active ** 2)))
        gain = min(0.155 / max(rms, 0.001), 0.82 / max(float(np.max(np.abs(samples))), 0.001))
        samples *= gain
        offset_seconds = scene["start"] + 0.25
        offset = round(offset_seconds * SAMPLE_RATE)
        narration[offset:offset + len(samples)] += samples
        for line in (AUDIO / f"{scene['id']}.jsonl").read_text(encoding="utf-8").splitlines():
            boundary = json.loads(line)
            start = offset_seconds + max(0, (boundary["offset"] / 1e7 - trim_start) / tempo)
            end = min(
                scene["start"] + scene["duration"] - 0.15,
                offset_seconds + (boundary["offset"] / 1e7 + boundary["duration"] / 1e7 - trim_start) / tempo,
            )
            if end > start:
                captions.append((start, end, boundary["text"]))
    (ROOT / "voice_timing_revised.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for entry in report:
        print(f"{entry['scene']:16} speech={entry['speech']:6.2f}s slot={entry['slot']:5.1f}s tempo={entry['tempo']:.3f}")
    if excessive:
        raise RuntimeError(f"Shorten these scenes to preserve calm pacing: {excessive}")
    score = ambient_score(storyboard["duration"])
    write_wav(AUDIO / "original_ambient_score.wav", score)
    mix = score * storyboard["music_gain"] + narration[:, None]
    transitions = [
        scene["start"]
        for index, scene in enumerate(storyboard["scenes"])
        if index > 0 and (
            scene["act"] != storyboard["scenes"][index - 1]["act"]
            or scene["id"] in ("roadmap_one", "close")
        )
    ]
    for transition in transitions:
        time = np.arange(round(0.7 * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
        cue = np.sin(2 * np.pi * 660 * time) * np.sin(np.pi * time / 0.7) ** 2 * np.exp(-time * 6) * 0.012
        offset = round(transition * SAMPLE_RATE)
        mix[offset:offset + len(cue)] += cue[:, None]
    assert float(np.max(np.abs(mix))) < 0.98, "Audio would clip."
    write_wav(AUDIO / "narration.wav", narration)
    write_wav(AUDIO / "final_mix.wav", mix)
    subtitle_blocks = []
    for index, (start, end, text) in enumerate(captions, start=1):
        subtitle_blocks.append(f"{index}\n{subtitle_time(start)} --> {subtitle_time(end)}\n{text}\n")
    (ROOT / f"{storyboard['output_stem']}.srt").write_text("\n".join(subtitle_blocks), encoding="utf-8")
    print(f"Audio verified: {len(mix) / SAMPLE_RATE:.3f}s; peak={float(np.max(np.abs(mix))):.3f}; music gain={storyboard['music_gain']:.0%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["check", "render"])
    arguments = parser.parse_args()
    storyboard = load_storyboard()
    if arguments.command == "check":
        print(f"Valid storyboard: {len(storyboard['scenes'])} scenes, {storyboard['duration']} seconds.")
        print(f"Narration: {sum(len(scene['narration'].split()) for scene in storyboard['scenes'])} written words.")
        preview = decode_audio(ROOT / "voice_preview.mp3")
        adjusted = decode_audio(ROOT / "voice_preview.mp3", 1.05)
        assert np.max(np.abs(preview)) > 0.02
        assert 1.03 < len(preview) / len(adjusted) < 1.07
        probe = ROOT / "encoder_probe.mp4"
        with av.open(str(probe), "w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width = 320
            stream.height = 180
            stream.pix_fmt = "yuv420p"
            for index in range(12):
                pixels = np.full((180, 320, 3), 30 + index * 10, dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        with av.open(str(probe)) as container:
            assert sum(1 for frame in container.decode(video=0)) == 12
        print(f"Female neural speech decoded: {len(preview) / SAMPLE_RATE:.2f}s.")
        print("Pitch-preserving tempo filter and H.264 encoding verified.")
    else:
        asyncio.run(synthesize_all(storyboard))
        assemble_audio(storyboard)


if __name__ == "__main__":
    main()