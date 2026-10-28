#!/usr/bin/env bash
# Full rebuild of the Tenglish/Tanglish HeliBoard from scratch:
#   git clone https://github.com/HeliBorg/HeliBoard && cd HeliBoard && git checkout 44ddb1a
#   git apply ../heliboard_dict/heliboard-tenglish-tanglish.patch
#   ./gradlew assembleDebug                      # once, so the makedict classes exist
#   <this script> . ../app/models/tenglish/tenglish_meta.json main_te-Latn.dict te-Latn "Tenglish (Telugu-English)"
#   <this script> . ../app/models/tanglish/tanglish_meta.json main_ta-Latn.dict ta-Latn "Tanglish (Tamil-English)"
#   cp main_*-Latn.dict app/src/main/assets/dicts/ && ./gradlew assembleDebug
# Usage: ./build_dict.sh <path/to/HeliBoard> <path/to/xxx_meta.json> <out.dict> <locale e.g. te-Latn> <description>
# Needs one prior `./gradlew assembleDebug` in HeliBoard (for its compiled makedict classes).
set -euo pipefail
HB=$(realpath "$1"); META=$(realpath "$2"); OUT=$3; LOC=$4; DESC=$5
cd "$(dirname "$0")"
SDK=${ANDROID_HOME:-$HOME/Android/Sdk}
CLS="$HB/app/build/intermediates/javac/debug/compileDebugJavaWithJavac/classes:$HB/app/build/intermediates/built_in_kotlinc/debug/compileDebugKotlin/classes"
JAR="$SDK/platforms/android-37.0/android.jar"
KT=$(ls ~/.gradle/wrapper/dists/gradle-*/*/gradle-*/lib/kotlin-stdlib-2*.jar | tail -1)
python3 make_tsv.py "$META"
javac -d stubsout stubs/android/util/Log.java stubs/android/os/Build.java
javac -cp "$CLS:$JAR" -d out TenglishDictCompiler.java
# stubsout must come before android.jar: the SDK jar's methods throw "Stub!" at runtime
java -cp "out:stubsout:$CLS:$JAR:$KT" TenglishDictCompiler unigrams.tsv bigrams.tsv "$OUT" "$LOC" "$DESC"
