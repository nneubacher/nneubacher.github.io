---
id: nn-1
title: "Local Call Transcription with IBM Granite Speech and MLX on Apple Silicon"
date: 2026-05-08 01:01:00 +0100
author: 'Nikolas Neubacher'
layout: post
guid: 'nikolasneubacher.com/nn-1'
permalink: /article/local-call-transcription-granite-speech-mlx/
custom_permalink:
    - article/local-call-transcription-granite-speech-mlx/
image:
pin: false
---

*IBM Granite Speech 4.1 running locally via MLX can transcribe both sides of a call in real time — microphone and system audio — without sending anything to the cloud. This post shows how to set it up on Apple Silicon.*

Teams, Zoom, and Meet all have built-in transcription, but they send your audio to a cloud service. For calls involving customers, internal strategy, or anything sensitive, that is not always acceptable. The alternative is to run the speech model locally so nothing leaves the machine.

With IBM Granite Speech 4.1 and Apple's MLX framework this is surprisingly practical — fast, private, and the output drops straight into your favorite text editor, in my case Obsidian.

## The Stack

* **IBM Granite Speech 4.1 2B** — IBM's open speech model on Hugging Face. Built on a Granite LM core, so it supports instruction-following over audio: transcribe, summarize, translate, or answer questions about a recording.
* **MLX** — Apple's inference framework for Apple Silicon. Runs model weights directly in unified memory, which makes it significantly faster than PyTorch on the same hardware.
* **mlx-audio** — open-source library wrapping STT and TTS models for MLX, with a simple `load` / `generate` API.

The 2B variant uses around 4.3 GB of memory and handles mixed-language audio (German and English in the same session, for example) without any configuration.

## Audio Routing on macOS

Capturing the microphone is straightforward. Capturing system audio — what plays through your speakers — requires a loopback driver. **BlackHole** is the standard free solution for macOS.

```bash
brew install blackhole-2ch
```

Then in **Audio MIDI Setup** (find it with Spotlight):

1. `(+)` → **Create Multi-Output Device**
2. Tick your normal output (AirPods, Built-in, etc.) and **BlackHole 2ch**
3. **System Settings → Sound → Output** → select the new Multi-Output Device

Your audio still plays normally. BlackHole taps a silent copy and makes it available as an input device.

## Running the Script

```bash
python transcribe.py --list-devices
python transcribe.py --mic "AirPods" --system "BlackHole"
```

Device names are matched by substring, so `"AirPods"` finds `"AirPods Pro von Nikolas"` automatically. Numeric indices work too.

```
--you      Speaker label for mic           (default: Microphone)
--them     Speaker label for system audio  (default: System Audio)
--title    Names the transcript and file
-o         Output path or directory
-t         VAD sensitivity threshold       (default: 0.01)
```

Two capture threads run concurrently, one per device. Each buffers audio when speech is detected and flushes the utterance after 800 ms of silence. Transcription runs sequentially on the main thread and writes each result to the output file immediately.

A live status bar shows recording time, queue depth, memory usage, and RTF while it runs:

```
● REC  00:03:42  ·  12 utterances  ·  queue: 0  ·  mem: 4.3 GB  ·  last RTF: 0.16×
```

## Output

Every session produces a Markdown file with YAML front-matter — compatible with Obsidian, Logseq, and importable into Notion:

```
---
title: Call — 2026-05-08 14:14
date: 2026-05-08
time: 14:14
participants:
  - Microphone (AirPods Pro von Nikolas)
  - System Audio (BlackHole 2ch)
model: ibm-granite/granite-speech-4.1-2b
---

# Call — 2026-05-08 14:14

**[14:14:58] Microphone:** marcus smart hat ein schlechtes spiel.

**[14:15:02] Microphone:** und es fällt einfach bei den scoring options auf.

**[14:15:11] Microphone:** oh man

**[14:15:29] Microphone:** ich teste gerade unser gespräch live mit der transkription. ich gucke mal, wie das hier funktioniert.

**[14:15:44] Microphone:** das habe ich gesehen, das sehen wir auf die rast.

**[14:15:47] Microphone:** gegangen sind warum auch immer.

**[14:15:48] System Audio:** wir wollen die kryptowährung anders besteuern. das ist dann die verantwortung des finanzministeriums. auch dafür sorgen muss die staatlichen einnahmen gestärkt werden. so einfach kann der finanzminister in deutschland leider kein gesetz einführen, das die kryptos besteuert, weil er offensichtlich ein urteil vom bundesverfassungsgericht vom siebten juli zweitausendzehn übersehen hat, was genau auf den fall hier anwendung findet. deswegen sage ich jetzt drei steuerliche grundprinzipien zur besteuerung von kryptobesteuerung. und das dritte prinzip betrifft alle gesetzesänderungen zu ihrem vorteil. sie müssen also keine steuern zahlen.

**[14:15:50] Microphone:** free throw merchant


---
*Recording ended at 14:16:13 — 00:01:22, 11 utterances, 141 words*
```

Because Obsidian live-reloads open files, you can leave the transcript open during the call and watch it update as the conversation progresses. Pointing `-o` to your vault folder is all the integration needed.

## Performance

On Apple Silicon, Granite Speech 4.1 2B transcribes well under real-time. An RTF of 0.16 means each second of audio takes about 160 ms to process — fast enough that the queue never builds up during normal conversation.

```
13:50:35  System Audio  ↳  9.7s audio · 1.6s · RTF 0.16× · 4.3 GB
13:51:17  System Audio  ↳ 42.1s audio · 6.8s · RTF 0.16× · 4.3 GB
```

A few honest caveats: the VAD is energy-based, so a noisy environment will produce some false utterances. Speaker separation works by device, not by voice — multiple remote participants all appear under the same label. And this was tested with microphone input and browser audio from a YouTube video, not a full multi-party call, so real-world results may vary. I'm very interested to hear from your experiences using Granite Speech 4.1! See below for further references

## The Script

One thing worth calling out: the script includes a monkey-patch for a bug in the current mlx-audio release. The Granite Speech model's weight loader doesn't correctly transpose the convolutional layer tensors when converting from PyTorch layout, which causes silent shape mismatches at inference time. The fix patches the `sanitize` method at import. I've submitted a [pull request to the mlx-audio repo](https://github.com/Blaizzy/mlx-audio/pull/715) — once merged the workaround can be removed.

<a href="https://gist.github.com/nneubacher/892e1805f283b244dd416c5114fd5d73/raw/transcribe.py" download="transcribe.py">⬇ Download transcribe.py</a>

<details id="gist-details">
<summary>View transcribe.py</summary>
<pre style="overflow:auto;max-height:520px;border-radius:6px;font-size:0.8em;margin-top:0.5em;"><code id="gist-code" class="language-python">Loading…</code></pre>
</details>
<script>
  document.getElementById('gist-details').addEventListener('toggle', function () {
    if (!this.open) return;
    var code = document.getElementById('gist-code');
    if (code.dataset.loaded) return;
    code.dataset.loaded = '1';

    function applyHighlight() {
      if (window.hljs) { hljs.highlightElement(code); }
    }

    function loadHljs(cb) {
      if (window.hljs) { cb(); return; }
      if (!document.getElementById('hljs-css')) {
        var link = document.createElement('link');
        link.id = 'hljs-css';
        link.rel = 'stylesheet';
        link.href = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css';
        document.head.appendChild(link);
      }
      var s = document.createElement('script');
      s.src = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js';
      s.onload = cb;
      document.head.appendChild(s);
    }

    fetch('https://gist.github.com/nneubacher/892e1805f283b244dd416c5114fd5d73/raw/transcribe.py')
      .then(function (r) { return r.text(); })
      .then(function (t) {
        code.textContent = t;
        loadHljs(applyHighlight);
      });
  });
</script>

## Get Started

* [IBM Granite Speech 4.1 on Hugging Face](https://huggingface.co/ibm-granite/granite-speech-4.1-2b)
* [mlx-audio on GitHub](https://github.com/Blaizzy/mlx-audio)
