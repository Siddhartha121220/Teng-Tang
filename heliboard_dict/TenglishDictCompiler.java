import helium314.keyboard.latin.makedict.FusionDictionary;
import helium314.keyboard.latin.makedict.FusionDictionary.PtNodeArray;
import helium314.keyboard.latin.makedict.FormatSpec;
import helium314.keyboard.latin.makedict.FormatSpec.DictionaryOptions;
import helium314.keyboard.latin.makedict.FormatSpec.FormatOptions;
import helium314.keyboard.latin.makedict.ProbabilityInfo;
import helium314.keyboard.latin.makedict.Ver2DictEncoder;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.HashMap;

public class TenglishDictCompiler {
    public static void main(String[] args) throws Exception {
        String uniPath = args.length > 0 ? args[0] : "unigrams.tsv";
        String biPath = args.length > 1 ? args[1] : "bigrams.tsv";
        String outPath = args.length > 2 ? args[2] : "tenglish.dict";

        String locale = args.length > 3 ? args[3] : "te-Latn";
        String description = args.length > 4 ? args[4] : "Tenglish";

        HashMap<String, String> attrs = new HashMap<>();
        // "dictionary" is required by the reader (DictionaryHeader), even though the writer doesn't check it
        attrs.put("dictionary", "main:" + locale);
        attrs.put("locale", locale);
        attrs.put("description", description);
        attrs.put("date", String.valueOf(System.currentTimeMillis() / 1000));
        attrs.put("version", "1");
        FusionDictionary dict = new FusionDictionary(new PtNodeArray(), new DictionaryOptions(attrs));

        int nWords = 0;
        try (BufferedReader r = new BufferedReader(new FileReader(uniPath))) {
            String line;
            while ((line = r.readLine()) != null) {
                if (line.isEmpty()) continue;
                String[] parts = line.split("\t");
                dict.add(parts[0], new ProbabilityInfo(Integer.parseInt(parts[1])), null, false, false);
                nWords++;
            }
        }
        System.out.println("Added " + nWords + " unigrams.");

        int nBigrams = 0;
        try (BufferedReader r = new BufferedReader(new FileReader(biPath))) {
            String line;
            while ((line = r.readLine()) != null) {
                if (line.isEmpty()) continue;
                String[] parts = line.split("\t");
                dict.setBigram(parts[0], parts[1], new ProbabilityInfo(Integer.parseInt(parts[2])));
                nBigrams++;
            }
        }
        System.out.println("Added " + nBigrams + " bigrams.");

        FormatOptions formatOptions = new FormatOptions(FormatSpec.VERSION202, false);
        Ver2DictEncoder encoder = new Ver2DictEncoder(new File(outPath), Ver2DictEncoder.CODE_POINT_TABLE_OFF);
        encoder.writeDictionary(dict, formatOptions);

        File out = new File(outPath);
        System.out.println("Wrote " + outPath + " (" + out.length() + " bytes).");
    }
}
