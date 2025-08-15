// KeygenStyleMusic.cpp
// Build: cl /O2 /EHsc KeygenStyleMusic.cpp
// Output: keygen_style.wav (44.1kHz, 16-bit PCM, stereo)
// 设计要点：不同种子 => 风格DNA(调式/BPM/鼓型/音色/FX/律动) 完全不同

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static constexpr double PI = 3.14159265358979323846;


// ---------------- RNG ----------------
struct Rng {
    std::mt19937 rng;
    explicit Rng(uint64_t seed) : rng(seed) {}
    int ri(int a, int b) { std::uniform_int_distribution<int> d(a, b); return d(rng); }
    double rf() { std::uniform_real_distribution<double> d(0.0, 1.0); return d(rng); }
    double rfn(double mu = 0.0, double sigma = 1.0) { std::normal_distribution<double> d(mu, sigma); return d(rng); }
    template<class T>
    T choice(const std::vector<T>& v) { return v[ri(0, (int)v.size() - 1)]; }
};

static inline double clampd(double x, double lo, double hi) { return std::max(lo, std::min(hi, x)); }

// ---------------- WAV ----------------
static void writeWav(const std::string& path, const std::vector<int16_t>& interleaved, int sr, int ch) {
    uint32_t dataSize = (uint32_t)(interleaved.size() * sizeof(int16_t));
    uint32_t fmtChunkSize = 16; uint16_t audioFormat = 1; uint16_t numChannels = (uint16_t)ch;
    uint32_t byteRate = sr * numChannels * 2; uint16_t blockAlign = numChannels * 2; uint16_t bps = 16;
    uint32_t riffSize = 36 + dataSize;
    std::ofstream f(path, std::ios::binary); if (!f) throw std::runtime_error("open wav fail");
    f.write("RIFF", 4); f.write((char*)&riffSize, 4); f.write("WAVE", 4);
    f.write("fmt ", 4); f.write((char*)&fmtChunkSize, 4);
    f.write((char*)&audioFormat, 2); f.write((char*)&numChannels, 2);
    f.write((char*)&sr, 4); f.write((char*)&byteRate, 4);
    f.write((char*)&blockAlign, 2); f.write((char*)&bps, 2);
    f.write("data", 4); f.write((char*)&dataSize, 4);
    f.write((char*)interleaved.data(), dataSize);
}

// ---------------- Music primitives ----------------
static double midiToFreq(double midi) { return 440.0 * std::pow(2.0, (midi - 69.0) / 12.0); }

enum class Scale { Major, NaturalMinor, HarmonicMinor, Dorian, Mixolydian, Phrygian, Aeolian, PentatonicMajor, PentatonicMinor };
static std::vector<int> scaleIntervals(Scale s) {
    switch (s) {
    case Scale::Major:            return { 0,2,4,5,7,9,11 };
    case Scale::NaturalMinor:     return { 0,2,3,5,7,8,10 };
    case Scale::HarmonicMinor:    return { 0,2,3,5,7,8,11 };
    case Scale::Dorian:           return { 0,2,3,5,7,9,10 };
    case Scale::Mixolydian:       return { 0,2,4,5,7,9,10 };
    case Scale::Phrygian:         return { 0,1,3,5,7,8,10 };
    case Scale::Aeolian:          return { 0,2,3,5,7,8,10 };
    case Scale::PentatonicMajor:  return { 0,2,4,7,9 };
    case Scale::PentatonicMinor:  return { 0,3,5,7,10 };
    }
    return { 0,2,4,5,7,9,11 };
}

enum class Osc { Sine, Square, Triangle, Saw, Noise, Pulse };

static inline double frac(double x) { return x - std::floor(x); }
static inline double oscSine(double ph) { return std::sin(2 * PI * frac(ph)); }
static inline double oscTri(double ph) { double x = frac(ph); return 2.0 * std::abs(2.0 * x - 1.0) - 1.0; }
static inline double oscSaw(double ph) { return 2.0 * frac(ph) - 1.0; }
static inline double oscSquare(double ph, double duty = 0.5) { return frac(ph) < duty ? 1.0 : -1.0; }

// cheap noise
static inline double lcgNoise(uint32_t& s) { s = s * 1664525u + 1013904223u; return ((s >> 1) & 0x7FFFFFFF) / double(0x7FFFFFFF) * 2.0 - 1.0; }

struct ADSR {
    double a{ 0.01 }, d{ 0.1 }, s{ 0.6 }, r{ 0.2 }, sr{ 44100.0 };
    void setup(double A, double D, double S, double R, double SR) { a = A; d = D; s = S; r = R; sr = SR; }
    // t: since note on; off >=0 means note-off at that time
    double env(double t, double off) const {
        if (off >= 0.0 && t >= off) {
            double tr = t - off; double start;
            if (off < a) start = off / a;
            else if (off < a + d) { double u = (off - a) / d; start = 1.0 + (s - 1.0) * u; }
            else start = s;
            return std::max(0.0, start * (1.0 - tr / r));
        }
        if (t < a) return t / a;
        if (t < a + d) { double u = (t - a) / d; return 1.0 + (s - 1.0) * u; }
        return s;
    }
};

// ---------------- FX ----------------
struct BitCrusher {
    int hold = 1, cnt = 0; double last = 0.0;
    void setup(int h) { hold = std::max(1, h); cnt = 0; }
    double process(double x) { if (cnt <= 0) { last = x; cnt = hold; } --cnt; return last; }
};
struct Delay {
    std::vector<double> L, R; size_t idx = 0; double fb = 0.3, mix = 0.35;
    void init(int sr, double sec) { size_t n = std::max(1, int(sr * sec)); L.assign(n, 0.0); R.assign(n, 0.0); idx = 0; }
    void set(double feedback, double wet) { fb = clampd(feedback, 0.0, 0.95); mix = clampd(wet, 0.0, 1.0); }
    void process(double& l, double& r) {
        double dl = L[idx], dr = R[idx];
        L[idx] = l + dl * fb; R[idx] = r + dr * fb;
        l = l * (1.0 - mix) + dl * mix;
        r = r * (1.0 - mix) + dr * mix;
        idx = (idx + 1) % L.size();
    }
};
struct Chorus {
    // simple mono -> stereo chorus via two short modulated delays
    std::vector<double> buf; size_t w = 0; int sr = 44100;
    double depthSamp = 8.0, rateHz = 0.6, mix = 0.22, phase = 0.0;
    void init(int SR, double maxMs = 30.0) { sr = SR; size_t n = (size_t)(SR * maxMs / 1000.0) + 2; buf.assign(n, 0.0); w = 0; phase = 0.0; }
    void set(double depthMs, double rate, double m) { depthSamp = clampd(depthMs * sr / 1000.0, 1.0, (double)buf.size() - 2); rateHz = rate; mix = clampd(m, 0.0, 1.0); }
    void process(double& l, double& r) {
        double x = (l + r) * 0.5;
        phase += rateHz / sr;
        double mod = (std::sin(2 * PI * phase) + 1.0) * 0.5; // 0..1
        double d = 4.0 + mod * depthSamp; // samples
        int rd = (int)d;
        int N = (int)buf.size();
        int ri = (int)w - rd; if (ri < 0) ri += N;
        int ri2 = ri - 1; if (ri2 < 0) ri2 += N;
        double fracp = d - rd;
        double y = buf[ri] * (1.0 - fracp) + buf[ri2] * fracp;
        buf[w] = x;
        w = (w + 1) % N;
        double wetL = x * 0.6 + y * 0.4;
        double wetR = x * 0.6 - y * 0.4;
        l = x * (1.0 - mix) + wetL * mix;
        r = x * (1.0 - mix) + wetR * mix;
    }
};

// ---------------- Style DNA ----------------
struct StyleDNA {
    std::string name;
    int bpm;
    double swing;          // 0..0.5 (swing amount for off 8th/16th)
    Scale scale;
    int rootMidi;          // e.g., 57 = A3
    std::vector<int> progression; // scale degrees (0=tonic) per bar
    Osc leadOsc, bassOsc, padOsc;
    double leadDuty, bassDuty, padDuty; // for square/pulse
    ADSR leadEnv, bassEnv, padEnv;
    int drumPattern;       // 0=four,1=break,2=halfTrap,3=shuffleFunk,4=euro
    bool arpLead;          // tracker-style arpeggio
    bool useChorus;
    double delaySec, delayFb, delayMix;
    int crushHold;         // bitcrusher hold samples
    double drive;          // master drive before tanh
    int bars;              // song length bars
};

// helpers
static int degToMidi(int root, const std::vector<int>& scaleI, int degree, int octaveOffset = 0) {
    // degree can exceed scale length; wrap with octave steps
    int n = (int)scaleI.size();
    int o = degree / n;
    int di = degree % n; if (di < 0) { di += n; o -= 1; }
    return root + scaleI[di] + 12 * (o + octaveOffset);
}

static std::vector<int> chooseProgression(Rng& r, const std::vector<int>& degrees, int bars) {
    // make a looped progression with some repeats
    std::vector<int> base = degrees;
    // random rotate
    int rot = r.ri(0, (int)base.size() - 1);
    std::rotate(base.begin(), base.begin() + rot, base.end());
    std::vector<int> out; out.reserve(bars);
    while ((int)out.size() < bars) {
        for (int d : base) {
            out.push_back(d);
            if ((int)out.size() >= bars) break;
        }
        // maybe insert a secondary cadence
        if (r.rf() < 0.35 && (int)out.size() + 2 <= bars) {
            out.push_back(r.choice(std::vector<int>{1, 4})); // subdominant or dominant
            out.push_back(0); // back to tonic
        }
    }
    return out;
}

static StyleDNA makeStyle(uint32_t seed, int sampleRate, Rng& r) {
    (void)sampleRate;
    StyleDNA s{};
    // choose family
    enum Fam { BrightTracker, DarkCyber, EuroDance, LoFiCrunch, BreakBeat };
    Fam fam = (Fam)(seed % 5);

    // common scales to choose per family
    auto S = [&](std::initializer_list<Scale> L) { return r.choice(std::vector<Scale>(L)); };
    auto O = [&](std::initializer_list<Osc> L) { return r.choice(std::vector<Osc>(L)); };

    s.bars = 16 + r.ri(0, 8); // 16~24 bars
    s.swing = 0.0;
    s.leadDuty = 0.5; s.bassDuty = 0.5; s.padDuty = 0.5;
    s.arpLead = false;
    s.useChorus = false;
    s.crushHold = 1 + r.ri(0, 4);
    s.delaySec = 0.22; s.delayFb = 0.3; s.delayMix = 0.3;
    s.drive = 0.85;
    s.rootMidi = 48 + r.ri(0, 12); // C3..B3
    s.leadOsc = O({ Osc::Saw, Osc::Pulse, Osc::Noise });  // 增加FM或双Saw

    switch (fam) {
    case BrightTracker: {
        s.name = "Bright Tracker";
        s.bpm = 160 + r.ri(-20, 10);
        s.scale = S({ Scale::Major, Scale::Mixolydian, Scale::PentatonicMajor });
        s.progression = chooseProgression(r, { 0,4,5,3 }, s.bars); // I–V–VI–IV variant
        s.leadOsc = O({ Osc::Square,Osc::Pulse,Osc::Saw });
        s.bassOsc = O({ Osc::Triangle,Osc::Square });
        s.padOsc = O({ Osc::Saw,Osc::Triangle });
        s.leadDuty = 0.5 + (r.rf() - 0.5) * 0.2;
        s.bassDuty = 0.5;
        s.padDuty = 0.5;
        s.leadEnv.setup(0.003, 0.03, 0.55, 0.06, 44100);
        s.bassEnv.setup(0.002, 0.05, 0.7, 0.05, 44100);
        s.padEnv.setup(0.02, 0.25, 0.8, 0.4, 44100);
        s.drumPattern = 0; // four-on-floor
        s.arpLead = true;
        s.useChorus = true;
        s.delaySec = 0.26; s.delayFb = 0.32; s.delayMix = 0.35;
        s.crushHold = 2 + r.ri(0, 2);
        s.drive = 0.9;
    } break;
    case DarkCyber: {
        s.name = "Dark Cyber";
        s.bpm = 100 + r.ri(-10, 15);
        s.scale = S({ Scale::HarmonicMinor, Scale::Phrygian, Scale::NaturalMinor });
        s.progression = chooseProgression(r, { 0,6,5,4 }, s.bars);
        s.leadOsc = O({ Osc::Saw,Osc::Square });
        s.bassOsc = O({ Osc::Square,Osc::Sine });
        s.padOsc = O({ Osc::Triangle,Osc::Saw });
        s.leadDuty = 0.5 + (r.rf() - 0.5) * 0.1;
        s.bassDuty = 0.5;
        s.padDuty = 0.5;
        s.leadEnv.setup(0.004, 0.06, 0.5, 0.12, 44100);
        s.bassEnv.setup(0.005, 0.09, 0.65, 0.15, 44100);
        s.padEnv.setup(0.05, 0.35, 0.9, 0.6, 44100);
        s.drumPattern = 2; // half-time trap-ish
        s.swing = 0.04;
        s.useChorus = false;
        s.delaySec = 0.33; s.delayFb = 0.38; s.delayMix = 0.32;
        s.crushHold = 3 + r.ri(0, 4);
        s.drive = 1.05;
    } break;
    case EuroDance: {
        s.name = "Eurodance";
        s.bpm = 140 + r.ri(-10, 20);
        s.scale = S({ Scale::Major, Scale::Mixolydian });
        s.progression = chooseProgression(r, { 0,4,1,5 }, s.bars); // I–V–ii–VI (变体)
        s.leadOsc = Osc::Pulse; s.leadDuty = 0.33 + r.rf() * 0.3;
        s.bassOsc = Osc::Square; s.bassDuty = 0.5;
        s.padOsc = Osc::Saw;
        s.leadEnv.setup(0.002, 0.02, 0.55, 0.06, 44100);
        s.bassEnv.setup(0.002, 0.03, 0.75, 0.08, 44100);
        s.padEnv.setup(0.02, 0.20, 0.85, 0.4, 44100);
        s.drumPattern = 4; // euro 四踩+开闭镲滚动
        s.arpLead = true;
        s.swing = 0.02;
        s.useChorus = true;
        s.delaySec = 0.24; s.delayFb = 0.28; s.delayMix = 0.28;
        s.crushHold = 2;
        s.drive = 0.95;
    } break;
    case LoFiCrunch: {
        s.name = "Lo-Fi Crunch";
        s.bpm = 90 + r.ri(-10, 10);
        s.scale = S({ Scale::PentatonicMinor, Scale::Dorian, Scale::Aeolian });
        s.progression = chooseProgression(r, { 0,3,4,2 }, s.bars);
        s.leadOsc = O({ Osc::Triangle,Osc::Square,Osc::Noise });
        s.bassOsc = O({ Osc::Sine,Osc::Triangle });
        s.padOsc = O({ Osc::Triangle,Osc::Square });
        s.leadDuty = 0.5 + (r.rf() - 0.5) * 0.2;
        s.bassDuty = 0.5;
        s.padDuty = 0.5;
        s.leadEnv.setup(0.01, 0.08, 0.6, 0.18, 44100);
        s.bassEnv.setup(0.005, 0.10, 0.7, 0.2, 44100);
        s.padEnv.setup(0.03, 0.40, 0.85, 0.6, 44100);
        s.drumPattern = 3; // shuffle/funk
        s.swing = 0.12;
        s.useChorus = false;
        s.delaySec = 0.28; s.delayFb = 0.35; s.delayMix = 0.25;
        s.crushHold = 5 + r.ri(0, 6);
        s.drive = 0.8;
    } break;
    case BreakBeat: {
        s.name = "Breakbeat";
        s.bpm = 170 + r.ri(-10, 10);
        s.scale = S({ Scale::NaturalMinor, Scale::Dorian });
        s.progression = chooseProgression(r, { 0,5,4,3 }, s.bars);
        s.leadOsc = O({ Osc::Saw,Osc::Square });
        s.bassOsc = O({ Osc::Square,Osc::Triangle });
        s.padOsc = O({ Osc::Saw,Osc::Triangle });
        s.leadDuty = 0.5 + (r.rf() - 0.5) * 0.15;
        s.bassDuty = 0.5;
        s.padDuty = 0.5;
        s.leadEnv.setup(0.003, 0.03, 0.52, 0.07, 44100);
        s.bassEnv.setup(0.003, 0.05, 0.7, 0.08, 44100);
        s.padEnv.setup(0.02, 0.22, 0.82, 0.38, 44100);
        s.drumPattern = 1; // break
        s.swing = 0.0;
        s.useChorus = true;
        s.delaySec = 0.21; s.delayFb = 0.28; s.delayMix = 0.32;
        s.crushHold = 2 + r.ri(0, 2);
        s.drive = 1.1;
    } break;
    }
    return s;
}

// ---------------- Sequencing ----------------
struct Note { int midi; double startBeat; double lenBeats; double vel; };

struct Song {
    int sampleRate = 44100;
    double bpm = 150.0, secPerBeat = 0.4;
    int bars = 16; double beatsPerBar = 4.0;
    std::vector<Note> lead, bass, pad;
    struct Hit { double beat; int type; }; // 0 kick,1 snare,2 hatClosed,3 hatOpen,4 clap
    std::vector<Hit> drums;
};

static void fillLead(const StyleDNA& s, Rng& r, Song& song) {
    auto scaleI = scaleIntervals(s.scale);
    // pattern building：按每小节 16 分
    int stepsPerBar = 16;
    for (int bar = 0; bar < song.bars; ++bar) {
        int degRoot = s.progression[bar % s.progression.size()];
        // 生成一组可用音 (三和弦 + 7th/9th 随机)
        std::vector<int> degs = { degRoot, degRoot + 2, degRoot + 4 };
        if (r.rf() < 0.5) degs.push_back(degRoot + 6); // 7th
        if (r.rf() < 0.25) degs.push_back(degRoot + 7); // 9th
        // 旋律八度随机：高八度/中八度
        int oct = r.choice(std::vector<int>{1, 2});
        for (int s16 = 0; s16 < stepsPerBar; ++s16) {
            double beat = bar * song.beatsPerBar + s16 * 0.25; // 16分
            double len = 0.22 + r.rf() * 0.08;
            int pick;
            if (s.arpLead) {
                // tracker 风格：琶音序列
                pick = (s16 * 2 + bar) % (int)degs.size();
                if (r.rf() < 0.08) pick = r.ri(0, (int)degs.size() - 1);
            }
            else {
                // 步进或级进
                if (s16 % 4 == 0) pick = r.ri(0, (int)degs.size() - 1);
                else pick = clampd((double)pick + (r.ri(-1, 1)), 0, (double)degs.size() - 1);
            }
            int midi = degToMidi(s.rootMidi, scaleI, degs[pick], oct);
            double vel = 0.8 + r.rf() * 0.2;
            song.lead.push_back({ midi, beat, len, vel });
            // 偶尔做装饰音
            if (r.rf() < 0.12) {
                int m2 = degToMidi(s.rootMidi, scaleI, degs[r.ri(0, (int)degs.size() - 1)], oct + (r.rf() < 0.5 ? 0 : 1));
                song.lead.push_back({ m2, beat + 0.125, 0.1, 0.6 });
            }
        }
    }
}

static void fillBass(const StyleDNA& s, Rng& r, Song& song) {
    auto scaleI = scaleIntervals(s.scale);
    for (int bar = 0; bar < song.bars; ++bar) {
        int degRoot = s.progression[bar % s.progression.size()];
        int rootMidi = degToMidi(s.rootMidi, scaleI, degRoot, -1);
        int fifthMidi = degToMidi(s.rootMidi, scaleI, degRoot + 4, -1);
        // 八分/四分随风格
        double step = (s.bpm >= 150 ? 0.5 : 1.0); // fast => 8th
        for (double beat = bar * song.beatsPerBar; beat < (bar + 1) * song.beatsPerBar - 1e-6; beat += step) {
            int use = (((int)((beat - bar * song.beatsPerBar) / step)) % 2 == 0) ? rootMidi : fifthMidi;
            double len = step * 0.9;
            double vel = 0.7 + r.rf() * 0.25;
            song.bass.push_back({ use, beat, len, vel });
            // passing note
            if (r.rf() < 0.15) {
                int pdeg = degRoot + (r.rf() < 0.5 ? -1 : 1);
                int pm = degToMidi(s.rootMidi, scaleI, pdeg, -1);
                song.bass.push_back({ pm, beat + step * 0.5, step * 0.45, 0.65 });
            }
        }
    }
}

static void fillPad(const StyleDNA& s, Rng& r, Song& song) {
    auto scaleI = scaleIntervals(s.scale);
    for (int bar = 0; bar < song.bars; ++bar) {
        int d = s.progression[bar % s.progression.size()];
        // 和弦叠三度（可能加7）
        std::vector<int> chordDeg = { d, d + 2, d + 4 };
        if (r.rf() < 0.4) chordDeg.push_back(d + 6);
        for (int i = 0; i < (int)chordDeg.size(); ++i) {
            int midi = degToMidi(s.rootMidi, scaleI, chordDeg[i], 0);
            double beat = bar * song.beatsPerBar + 0.0 + i * 0.02; // 轻微错位
            double len = song.beatsPerBar * (0.95 + r.rf() * 0.05);
            double vel = 0.45 + r.rf() * 0.15;
            song.pad.push_back({ midi, beat, len, vel });
        }
    }
}

static void fillDrums(const StyleDNA& s, Rng& r, Song& song) {
    // patterns
    for (int bar = 0; bar < song.bars; ++bar) {
        double base = bar * song.beatsPerBar;
        switch (s.drumPattern) {
        case 0: { // four-on-floor
            for (int s16 = 0; s16 < 16; ++s16) {
                double b = base + 0.25 * s16;
                if (s16 % 4 == 0) song.drums.push_back({ b,0 });        // kick on 1/2/3/4
                if (s16 == 4 || s16 == 12) song.drums.push_back({ b,1 }); // snare on 2/4
                if (s16 % 2 == 0) song.drums.push_back({ b,2 });        // closed hat 8th
                if (r.rf() < 0.12) song.drums.push_back({ b,3 });     // open hat occasional
                if (r.rf() < 0.15 && s16 % 4 != 0) song.drums.push_back({ b,4 }); // clap ghost
            }
        } break;
        case 1: { // breakbeat: ghost snares & syncopated kicks
            for (int s16 = 0; s16 < 16; ++s16) {
                double b = base + 0.25 * s16;
                if (s16 == 0 || s16 == 7 || s16 == 8 || (s16 == 10 && r.rf() < 0.6)) song.drums.push_back({ b,0 });
                if (s16 == 4 || s16 == 12) song.drums.push_back({ b,1 });
                if (s16 % 1 == 0 && r.rf() < 0.85) song.drums.push_back({ b,2 });
                if (s16 == 14 && r.rf() < 0.8) song.drums.push_back({ b,1 }); // late snare
                if ((s16 == 3 || s16 == 11) && r.rf() < 0.6) song.drums.push_back({ b,4 });
            }
        } break;
        case 2: { // half-time trap-ish
            for (int s16 = 0; s16 < 16; ++s16) {
                double b = base + 0.25 * s16;
                if (s16 == 0 || (s16 == 8 && r.rf() < 0.6)) song.drums.push_back({ b,0 });
                if (s16 == 8) song.drums.push_back({ b,1 }); // backbeat on 3
                if (s16 % 1 == 0 && (r.rf() < 0.5 || s16 % 2 == 0)) song.drums.push_back({ b,2 });
                if (r.rf() < 0.2 && s16 % 4 == 2) song.drums.push_back({ b,3 });
            }
        } break;
        case 3: { // shuffle/funk
            for (int s12 = 0; s12 < 12; ++s12) {
                double triBeat = base + (song.beatsPerBar / 12.0) * s12; // triplet grid
                if (s12 == 0 || s12 == 6) song.drums.push_back({ triBeat,0 });
                if (s12 == 4 || s12 == 10) song.drums.push_back({ triBeat,1 });
                if (s12 % 2 == 0) song.drums.push_back({ triBeat,2 });
                if (r.rf() < 0.15 && s12 % 3 == 2) song.drums.push_back({ triBeat,3 });
            }
        } break;
        case 4: { // eurodance: strong 4/4 + rolling hats
            for (int s16 = 0; s16 < 16; ++s16) {
                double b = base + 0.25 * s16;
                if (s16 % 4 == 0) song.drums.push_back({ b,0 });
                if (s16 == 4 || s16 == 12) song.drums.push_back({ b,1 });
                song.drums.push_back({ b,2 }); // continuous 16th hats
                if (s16 % 8 == 4) song.drums.push_back({ b,3 }); // open on offbeat
                if (r.rf() < 0.1 && s16 % 4 == 2) song.drums.push_back({ b,4 });
            }
        } break;
        }
    }
}

// swing timing: return time offset seconds for off-beat steps (8th/16th)
static double swingOffset(double beatPosWithinBar, double swingAmount, bool isTriplet = false, double bpm = 120.0) {
    if (swingAmount <= 1e-6) return 0.0;
    // simple: push the "and" of the beat later
    double secPerBeat = 60.0 / bpm;
    // 16th notes: offset every odd 16th slightly
    double f = beatPosWithinBar * 4.0; // 4 = beats per bar => 16th indices 0..16
    int idx16 = (int)std::round(f * 4.0);
    bool isOff = (idx16 % 2 == 1);
    return isOff ? swingAmount * 0.5 * secPerBeat * (isTriplet ? 0.6 : 1.0) : 0.0;
}

// ---------------- Rendering ----------------
struct SynthState { double phase = 0.0; uint32_t noise = 0x12345678; };
static double oscSample(Osc osc, double& phase, uint32_t& ns, double f, double sr, double duty) {
    phase += f / sr;
    switch (osc) {
    case Osc::Sine:     return oscSine(phase);
    case Osc::Triangle: return oscTri(phase);
    case Osc::Square:   return oscSquare(phase, duty);
    case Osc::Pulse:    return oscSquare(phase, duty);
    case Osc::Saw:      return oscSaw(phase);
    case Osc::Noise:    return lcgNoise(ns);
    }
    return 0.0;
}

static void synthTrack(const std::vector<Note>& notes, const ADSR& env, Osc osc, double duty,
    double gain, int sr, double bpm, double swing, std::vector<double>& L, std::vector<double>& R, double pan = 0.0) {
    // precompute note ranges to reduce CPU
    SynthState st{};
    for (const auto& n : notes) {
        size_t start = (size_t)std::round(n.startBeat * (60.0 / bpm) * sr);
        size_t durS = (size_t)std::max(1.0, n.lenBeats * (60.0 / bpm) * sr);
        size_t offS = (size_t)(durS * 0.85);
        for (size_t i = 0; i < durS; ++i) {
            size_t idx = start + i; if (idx >= L.size()) break;
            double t = i / (double)sr;
            double offTime = (i >= offS) ? (offS / (double)sr) : -1.0;
            double e = env.env(t, offTime);
            double f = midiToFreq(n.midi);
            // 微小vibrato（不统一，避免千篇一律）
            double vibr = 0.06 * std::sin(2 * PI * 5.0 * (idx / (double)sr));
            double s = oscSample(osc, st.phase, st.noise, f * std::pow(2.0, vibr / 12.0), sr, duty);
            double v = std::tanh(s * e * n.vel * gain);
            double l = v * (0.5 - pan * 0.5);
            double r = v * (0.5 + pan * 0.5);
            L[idx] += l; R[idx] += r;
        }
    }
}

static double kickSample(double t) {
    double f = 120.0 * std::exp(-t * 8.0) + 40.0;
    double amp = std::exp(-t * 7.0);
    return std::sin(2 * PI * f * t) * amp;
}
static double snareSample(double t, uint32_t& ns) {
    double tone = std::sin(2 * PI * 180.0 * t) * std::exp(-t * 12.0);
    double n = lcgNoise(ns) * 0.9 * std::exp(-t * 25.0);
    return tone * 0.3 + n * 0.7;
}
static double hatClosedSample(double t) { double env = std::exp(-t * 90.0); double a = std::sin(2 * PI * 6500 * t), b = std::sin(2 * PI * 9000 * t); return (a + b) * 0.25 * env; }
static double hatOpenSample(double t) { double env = std::exp(-t * 18.0);  double a = std::sin(2 * PI * 6500 * t), b = std::sin(2 * PI * 9000 * t); return (a + b) * 0.25 * env; }
static double clapSample(double t, uint32_t& ns) { double n = 0.0; for (int i = 0; i < 4; ++i) { n += std::abs(lcgNoise(ns)); } n /= 4.0; return n * std::exp(-t * 25.0); }

static void renderDrums(const std::vector<Song::Hit>& hits, int sr, double bpm, double swing,
    std::vector<double>& L, std::vector<double>& R) {
    uint32_t ns = 0x87654321;
    for (const auto& h : hits) {
        size_t start = (size_t)std::round(h.beat * (60.0 / bpm) * sr);
        size_t durS = (size_t)(sr * 0.6);
        for (size_t i = 0; i < durS; ++i) {
            size_t idx = start + i; if (idx >= L.size()) break;
            double t = i / (double)sr;
            double s = 0.0;
            if (h.type == 0) s = kickSample(t) * 1.0;
            else if (h.type == 1) s = snareSample(t, ns) * 0.7;
            else if (h.type == 2) s = hatClosedSample(t) * 0.4;
            else if (h.type == 3) s = hatOpenSample(t) * 0.5;
            else if (h.type == 4) s = clapSample(t, ns) * 0.5;
            double l = s * 0.9, r = s * 0.9;
            L[idx] += l; R[idx] += r;
        }
    }
}

static void applyMaster(std::vector<double>& L, std::vector<double>& R, double drive,
    int crushHold, double delaySec, double delayFb, double delayMix, bool chorusOn, int sr) {
    // bitcrush
    BitCrusher bcL, bcR; bcL.setup(std::max(1, crushHold)); bcR.setup(std::max(1, crushHold + 1));
    for (size_t i = 0; i < L.size(); ++i) { L[i] = bcL.process(L[i]); R[i] = bcR.process(R[i]); }
    // pre-drive soft clip
    for (size_t i = 0; i < L.size(); ++i) { L[i] = std::tanh(L[i] * drive); R[i] = std::tanh(R[i] * drive); }
    // chorus
    Chorus ch; if (chorusOn) { ch.init(sr, 30.0); ch.set(8.0, 0.5, 0.25); for (size_t i = 0; i < L.size(); ++i) { double l = L[i], r = R[i]; ch.process(l, r); L[i] = l; R[i] = r; } }
    // delay
    Delay d; d.init(sr, delaySec); d.set(delayFb, delayMix);
    for (size_t i = 0; i < L.size(); ++i) { double l = L[i], r = R[i]; d.process(l, r); L[i] = l; R[i] = r; }
    // width + normalize
    double peak = 1e-9;
    for (size_t i = 0; i < L.size(); ++i) {
        double mid = (L[i] + R[i]) * 0.5, side = (L[i] - R[i]) * 0.5;
        side *= 1.1; L[i] = mid + side; R[i] = mid - side;
        peak = std::max(peak, std::abs(L[i])); peak = std::max(peak, std::abs(R[i]));
    }
    double norm = 0.92 / peak;
    for (size_t i = 0; i < L.size(); ++i) { L[i] *= norm; R[i] *= norm; }
}

// ---------------- Main ----------------
int main(int argc, char** argv) {
    const int SR = 44100; const int CH = 2;
    uint64_t seed;
    if (argc >= 2) try { seed = (uint64_t)std::stoull(argv[1]); }
    catch (...) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint64_t> dis_int(0, UINT_MAX);
        seed = dis_int(gen);
    }
    else seed = (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    int lengthSec = (argc >= 3) ? std::max(10, std::atoi(argv[2])) : 0; // 0 => auto by bars

    Rng rng(seed);
    StyleDNA style = makeStyle(seed, SR, rng);
    if (lengthSec > 0.0) {
        int barsNeeded = (int)std::ceil(lengthSec / (60.0 / style.bpm) / 4.0);
        style.bars = barsNeeded;
    }
    Song song; song.sampleRate = SR; song.bpm = style.bpm; song.secPerBeat = 60.0 / style.bpm; song.bars = style.bars;
    std::cout << "Seed: " << seed << "\n";
    std::cout << "Style: " << style.name << "\n";
    std::cout << "BPM: " << style.bpm << ", Scale size: " << scaleIntervals(style.scale).size()
        << ", Bars: " << style.bars << "\n";

    // Build parts
    fillLead(style, rng, song);
    fillBass(style, rng, song);
    fillPad(style, rng, song);
    fillDrums(style, rng, song);

    // Length
    double songBeats = song.bars * song.beatsPerBar;
    double songSeconds = songBeats * song.secPerBeat;
    if (lengthSec > 0) songSeconds = (double)lengthSec;

    size_t N = (size_t)(songSeconds * SR);
    std::vector<double> L(N, 0.0), R(N, 0.0);

    // Render tracks
    synthTrack(song.pad, style.padEnv, style.padOsc, style.padDuty, 0.22, SR, style.bpm, style.swing, L, R, 0.0);
    synthTrack(song.bass, style.bassEnv, style.bassOsc, style.bassDuty, 0.32, SR, style.bpm, style.swing, L, R, -0.1);
    synthTrack(song.lead, style.leadEnv, style.leadOsc, style.leadDuty, 0.28, SR, style.bpm, style.swing, L, R, 0.1);

    renderDrums(song.drums, SR, style.bpm, style.swing, L, R);

    // FX & master
    applyMaster(L, R, style.drive, style.crushHold, style.delaySec, style.delayFb, style.delayMix, style.useChorus, SR);

    // Dither + write
    Rng ditherRng(seed ^ 0xA5A5A5A5u);
    std::vector<int16_t> out; out.resize(N * CH);
    for (size_t i = 0; i < N; ++i) {
        double d1 = (ditherRng.rf() - ditherRng.rf()) * (1.0 / 32768.0);
        double d2 = (ditherRng.rf() - ditherRng.rf()) * (1.0 / 32768.0);
        int16_t l = (int16_t)clampd((L[i] + d1) * 32767.0, -32768.0, 32767.0);
        int16_t r = (int16_t)clampd((R[i] + d2) * 32767.0, -32768.0, 32767.0);
        out[i * 2 + 0] = l; out[i * 2 + 1] = r;
    }

    try {
        writeWav("keygen_style.wav", out, SR, CH);
        std::cout << "Generated keygen_style.wav (" << (N / (double)SR) << " s)\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n"; return 1;
    }
    return 0;
}
