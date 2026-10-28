# Teng-Tang — a keyboard that speaks Tenglish and Tanglish

**Teng-Tang** brings next-word prediction and word completion to **romanized code-mixed Telugu–English (Tenglish)** and **Tamil–English (Tanglish)** — the way people actually text:

> *"repu class undi kada"* · *"nenu inka ready avvaledhu ra"* · *"machi naalaiku party irukku da"*

Mainstream keyboards are trained on English or on native-script Telugu/Tamil, so they "correct" `cheppu` into `chew` and never predict `naku` after `repu`. Teng-Tang fixes that with a language model built from real code-mixed chat, running **fully offline on your phone**.

---

## Contents

- [Two ways to use it](#two-ways-to-use-it)
- [Option 1 — Teng-Tang keyboard (recommended)](#option-1--teng-tang-keyboard-recommended)
- [Option 2 — Add the words to Gboard](#option-2--add-the-words-to-gboard)
- [How it works](#how-it-works)
- [Results](#results)
- [Repository layout](#repository-layout)
- [Building from source](#building-from-source)
- [Privacy](#privacy)
- [License](#license)

---

## Two ways to use it

| | **Option 1: Teng-Tang keyboard** | **Option 2: Gboard word list** |
|---|---|---|
| What you install | A keyboard app (looks like Gboard) | One zip file imported into Gboard |
| Completes words (`che` → `cheppu`) | ✅ | ✅ |
| Stops "correcting" Tenglish/Tanglish into English | ✅ | ✅ |
| **Predicts the next word** (`repu ` → `naku`, `exam`) | ✅ | ❌ — Gboard can't import word pairs |
| Ranks words by how common they are | ✅ | ❌ — all imported words count equally |
| Separate Tenglish and Tanglish models | ✅ switch with a long-press on space | ❌ one shared list |
| Learns from your own typing | ✅ per language | ✅ Gboard's own learning |
| Effort | Install + enable (2 min) | Import a file (30 s) |

**Option 1 is the full product. Option 2 is a quick way to try it inside the keyboard you already use.**

---

## Option 1 — Teng-Tang keyboard (recommended)

A Gboard-style keyboard (rounded keys, Material You colours on Android 12+, emoji key) with built-in **Tenglish** and **Tanglish** languages. Plain English QWERTY keys — only the suggestions change.

**Requirements:** Android 8.0 or newer.

### Install

1. On your phone, open the **[Releases](../../releases)** page and download the latest **`.apk`** file.
2. Open the downloaded file. If Android asks, allow **"Install unknown apps"** for your browser or file manager, then tap **Install**.
   *(This is needed for any app installed outside the Play Store.)*
3. Open the keyboard app once from your app drawer.

### Turn it on

4. Go to **Settings → System → Languages & input → On-screen keyboard → Manage keyboards** *(the exact path varies a little by phone; searching Settings for "keyboard" works too)*.
5. Switch the Teng-Tang keyboard **on**. Android shows a standard warning that keyboards can read what you type — the keyboard is fully offline and has **no internet permission**, so nothing you type leaves your phone.
6. Open any app with a text box (Messages, WhatsApp, Notes…), tap the **keyboard icon** at the bottom right of the screen (or pull down notifications → *Change keyboard*), and choose the Teng-Tang keyboard.

### Use it

- The **space bar shows the active language** — *Tenglish* or *Tanglish*. Both are enabled out of the box.
- **Switch language:** **long-press the space bar** and pick Tenglish or Tanglish.
- **Next word:** type a word and a space — the top 3 likely next words appear above the keyboard (`repu ` → `exam · naku · morning`). The middle one is the best guess.
- **Complete a word:** start typing — `che` → `cheppu · chestha · chesina`. Tap a suggestion to insert it.
- **It learns:** words you type often move up, and new words (names, slang) get remembered — separately for each language.
- Swipe left/right on the space bar to move the cursor, as in Gboard.
- Theme, key height, number row, etc. can be changed in the keyboard's **Settings → Appearance / Preferences**.

### Going back to your old keyboard

Tap the keyboard icon at the bottom right of any text box and choose your previous keyboard. To remove Teng-Tang completely, uninstall the app like any other.

---

## Option 2 — Add the words to Gboard

Adds thousands of Tenglish or Tanglish words to Gboard's personal dictionary, so Gboard **suggests and stops autocorrecting** them. No new app.

1. Download **`Tenglish_Gboard.zip`** and/or **`Tanglish_Gboard.zip`** from the **[Releases](../../releases)** page. **Don't unzip it.**
2. Open **Gboard settings** (long-press the comma key → ⚙, or **Settings → Languages & input → Gboard**).
3. Go to **Dictionary → Personal dictionary → All languages**.
4. Tap **⋮** (top right) → **Import**, and pick the zip from your **Downloads**.
   *If the file isn't listed, open the picker's ☰ menu and choose **Downloads**.*
5. Wait for *"Import finished"*. Type `che` (Tenglish) or `naa` (Tanglish) to see the new words.

**Tips**
- The Tenglish and Tanglish files have almost the same name — check you picked the one you want.
- Import each file **once**; importing again creates duplicates.
- To undo, remove words individually in the same Personal dictionary screen, or clear Gboard's storage in *Settings → Apps → Gboard → Storage* (this also resets Gboard's settings).

**Limit:** Gboard has no way to import word-pair statistics, so it won't predict *the next word* the way Option 1 does.

---

## How it works

```
 WhatsApp chat exports ──► prepare_tenglish.py ──► cleaned code-mixed sentences
                                                         │
                       ┌─────────────────────────────────┼──────────────────────────────┐
                       ▼                                 ▼                              ▼
            Fine-tuned DistilGPT2           Word + word-pair statistics         Word list (seen ≥ 2×)
            + trigram n-gram hybrid         (unigram/bigram probabilities)              │
            (research model, notebook)                   │                              ▼
                       │                                 ▼                     Gboard personal-
                       ▼                      binary keyboard dictionary          dictionary zip
             FastAPI demo web app          (te-Latn / ta-Latn .dict files)         (Option 2)
                                                         │
                                                         ▼
                                       Teng-Tang keyboard (HeliBoard-based, Option 1)
```

### 1. Data
Romanized code-mixed text is scarce, so the Tenglish data was collected from **real WhatsApp chats** (with the participants' permission). `prepare_tenglish.py` parses Android / iOS / WhatsApp-Web export formats, drops media notices, links, phone numbers, native-script lines and names, and keeps only clauses that contain Telugu words (≈8,600 sentences, ≈8,000-word vocabulary). The Tanglish model uses a separately built dataset (≈2,000 sentences).

### 2. Research model
`tenglish_autocomplete.ipynb` (runs on a free Kaggle GPU) fine-tunes **DistilGPT2** on the sentences and blends it with an **interpolated trigram n-gram**: every vocabulary word is scored whole, and the two scores are combined as `p = α·p_LM + (1−α)·p_ngram` with α tuned on validation data. `app/` serves both models through a small FastAPI web app for experimentation.

### 3. On-device keyboard
Running a neural model on every keystroke is too slow and battery-hungry for a phone keyboard, and on this data the n-gram carries almost all of the accuracy (see Results). The keyboard therefore ships the **statistical part** of the model:

- `heliboard_dict/make_tsv.py` converts the training sentences into word probabilities **P(w)** and word-pair probabilities **P(w | previous word)**, on one shared log-probability scale.
- `heliboard_dict/TenglishDictCompiler.java` writes them into the standard binary dictionary format used by AOSP-based keyboards.
- The keyboard is a fork of the open-source **[HeliBoard](https://github.com/HeliBorg/HeliBoard)**. `heliboard-tenglish-tanglish.patch` adds two languages — `te-Latn` (Tenglish) and `ta-Latn` (Tanglish), modelled on HeliBoard's existing Hinglish support — plus Gboard-style default appearance.
- HeliBoard's built-in **user-history dictionary** then learns from each user's own typing, separately per language — so suggestions keep improving the more you type.

### 4. Gboard option
`gboard_dict/make_gboard_zip.py` exports the frequent words in Gboard's personal-dictionary import format (`shortcut<TAB>word<TAB>locale`).

---

## Results

Held-out test set of **real chat sentences** (Tenglish, 4,175 positions). Accuracy = the correct word is among the top 1 / 3 / 5 suggestions.

| Model | nothing typed (1/3/5) | 1 letter typed (1/3/5) | 2 letters typed (1/3/5) |
|---|---|---|---|
| Most-frequent baseline | .024 / .051 / .069 | .143 / .268 / .343 | .272 / .474 / .558 |
| N-gram only | .068 / .125 / .159 | .203 / .340 / .407 | .316 / .511 / .588 |
| DistilGPT2 only | .057 / .113 / .150 | .190 / .328 / .394 | .302 / .484 / .558 |
| **Hybrid (α = 0.5)** | **.069 / .133 / .171** | **.209 / .348 / .417** | **.321 / .505 / .582** |

- Both learned models clearly beat the frequency baseline.
- The hybrid is best overall, but only slightly ahead of the n-gram alone — which is why the phone keyboard uses the n-gram statistics and leaves the neural model to research/demo use.
- Over half the vocabulary appears only once and many test words never appear in training, which caps achievable accuracy; more data is the biggest lever.

---

## Repository layout

```
├── prepare_tenglish.py          WhatsApp exports → cleaned sentence CSV
├── audit_dataset.py             quality / privacy audit of that CSV
├── tenglish_autocomplete.ipynb  Kaggle notebook: DistilGPT2 + n-gram hybrid, evaluation
├── tenglish_synthetic.csv       small hand-written example set (shows the CSV format)
├── app/                         FastAPI demo serving both models (+ web UI)
├── heliboard_dict/
│   ├── heliboard-tenglish-tanglish.patch   keyboard changes (apply to HeliBoard)
│   ├── make_tsv.py                         sentences → word / word-pair probabilities
│   ├── TenglishDictCompiler.java           probabilities → binary .dict
│   ├── stubs/                              tiny Android stand-ins so the compiler runs on a PC
│   └── build_dict.sh                       one-command dictionary build (+ full rebuild steps)
└── gboard_dict/
    └── make_gboard_zip.py       word list → Gboard personal-dictionary zip
```

---

## Building from source

Datasets and trained models are **not** in this repository (see [Privacy](#privacy)); build your own from your own chats.

### 1. Make a dataset

Export chats from the WhatsApp **phone app** (*chat → ⋮ → More → Export chat → Without media*) into a `chats/` folder, then:

```bash
python3 prepare_tenglish.py "chats/*.txt" --names "Your Name" "Friend A" "Friend B" --out tenglish_train.csv
python3 audit_dataset.py tenglish_train.csv --names "Your Name" "Friend A" "Friend B"
```

Pass every name and nickname to `--names` so they are removed, and read the audit for leftover names or numbers. Get your friends' permission before using their messages.

### 2. Train the research model (optional)

Upload `tenglish_train.csv` as a **private** Kaggle dataset, import `tenglish_autocomplete.ipynb`, enable **GPU** and **Internet**, attach the dataset and run all. Download the output zip and extract it to `app/models/tenglish/` (it contains `tenglish_meta.json`, which the next steps read).

```bash
cd app && pip install -r requirements.txt && uvicorn app:app      # demo at http://localhost:8000
```

### 3. Build the keyboard

Needs JDK 17, Python 3, and the Android SDK (the Gradle build fetches the right platform and NDK).

```bash
git clone https://github.com/HeliBorg/HeliBoard && cd HeliBoard
git checkout 44ddb1a
git apply ../heliboard_dict/heliboard-tenglish-tanglish.patch
./gradlew assembleDebug                              # first build: compiles the dictionary tooling

../heliboard_dict/build_dict.sh . ../app/models/tenglish/tenglish_meta.json main_te-Latn.dict te-Latn "Tenglish"
../heliboard_dict/build_dict.sh . ../app/models/tanglish/tanglish_meta.json main_ta-Latn.dict ta-Latn "Tanglish"
cp ../heliboard_dict/main_*-Latn.dict app/src/main/assets/dicts/

./gradlew assembleDebug                              # APK in app/build/outputs/apk/debug/
adb install app/build/outputs/apk/debug/*.apk
```

For a shareable build, use `./gradlew assembleRelease` with your own signing key, and give the app its own name and package ID.

### 4. Build the Gboard zip

```bash
python3 gboard_dict/make_gboard_zip.py app/models/tenglish/tenglish_meta.json Tenglish_Gboard.zip
```

---

## Privacy

- The training data is **private chat messages**. Raw exports, cleaned datasets, trained models and anything derived from them (dictionaries, word lists) are excluded from this repository via `.gitignore`.
- The keyboard runs **entirely on the phone** and has no internet permission; what you type and what it learns never leave the device.
- Before sharing any build, check its word lists for personal names.

---

## License

The keyboard is based on **[HeliBoard](https://github.com/HeliBorg/HeliBoard)** (GPL-3.0); the patch and any keyboard builds are distributed under **GPL-3.0**. HeliBoard is in turn based on the AOSP LatinIME keyboard (Apache-2.0).
