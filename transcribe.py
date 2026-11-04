import sys, os, glob, time, subprocess
shard, N, lo, hi = map(int, sys.argv[1:5])
from faster_whisper import WhisperModel
os.makedirs("out", exist_ok=True)
files = sorted(glob.glob("audio/*.ogg"))
mine = [f for f in files if lo <= int(os.path.basename(f)[:-4]) <= hi and int(os.path.basename(f)[:-4]) % N == shard]
def save():
    for _ in range(8):
        subprocess.run("git pull --rebase -q origin main", shell=True)
        subprocess.run("git add out && git commit -q -m 'transcripts [skip ci]'", shell=True)
        if subprocess.run("git push -q origin HEAD:main", shell=True).returncode == 0: return
        time.sleep(7 + shard)
m = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8", cpu_threads=4)
done = 0
for f in mine:
    n = os.path.basename(f)[:-4]; out = f"out/{n}.txt"
    if os.path.exists(out): continue
    t = time.time()
    segs, info = m.transcribe(f, language="ur", beam_size=5, vad_filter=True,
        initial_prompt="ڈاکٹر اسرار احمد، بیان القرآن، سورۃ الفاتحہ، سورۃ البقرہ، آیات، اللہ تعالیٰ، رسول اللہ صلی اللہ علیہ وسلم")
    with open(out + ".tmp", "w", encoding="utf-8") as o:
        for s in segs: o.write(f"[{int(s.start//60)}:{int(s.start%60):02d}] {s.text.strip()}\n")
    os.rename(out + ".tmp", out)
    print(n, round(time.time()-t), "s", flush=True)
    done += 1
    if done % 5 == 0: save()
save()
