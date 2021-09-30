import re
import json
import random
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str)
    parser.add_argument("outfile", type=str)
    return parser.parse_args()


mods = {"AUX": ["might", "may"],
        "ADV": ["maybe", "probably", "likely"]}

aux_patterns = [(re.compile(r'\bis not\b'), "<AUX> not be"),
                (re.compile(r'\bare not\b'), "<AUX> not be"),
                (re.compile(r'\bwas not\b'), "<AUX> not have been"),
                (re.compile(r'\bwere not\b'), "<AUX> not have been"),
                (re.compile(r'\b(is)\b|\b(are)\b'), "<AUX> be"),
                (re.compile(r'\bwas\b'), "<AUX> have been"),
                (re.compile(r'\bdoes not\b'), "<AUX> not"),
                (re.compile(r'\bdo not\b'), "<AUX> not"),
                (re.compile(r'\bdid not\b'), "<AUX> not have"),
                (re.compile(r'\bhas not\b'), "<AUX> not have"),
                (re.compile(r'\bhave not\b'), "<AUX> not have"),
                (re.compile(r'\bhas\b'), "<AUX> have"),
                (re.compile(r'\bhave\b'), "<AUX> have"),
                (re.compile(r'\brequires?\b'), "<AUX> require"),
                (re.compile(r'\bcauses?\b'), "<AUX> cause"),
                (re.compile(r'\bwants?\b'), "<AUX> want"),
                (re.compile(r'\bneeds?\b'), "<AUX> need"),
                (re.compile(r'\bindicates?\b'), "<AUX> indicate"),
                (re.compile(r'\bcontains?\b'), "<AUX> contain"),
                (re.compile(r'\bcontains?\b'), "<AUX> indicate"),
                (re.compile(r'\blikes?\b'), "<AUX> like"),
                (re.compile(r'\bexhibits?\b'), "<AUX> exhibit"),
                (re.compile(r'\btastes?\b'), "<AUX> taste"),
                (re.compile(r'\bproduces?\b'), "<AUX> produce"),
                (re.compile(r'\bruns?\b'), "<AUX> run"),
                (re.compile(r'\bsmells?\b'), "<AUX> smell"),
                (re.compile(r'\bfounded\b'), "<AUX> have founded")
                ]

adv_patterns = [(re.compile(r'\bis not\b'), "is <ADV> not"),
                (re.compile(r'\bare not\b'), "are <ADV> not"),
                (re.compile(r'\bwas not\b'), "was <ADV> not"),
                (re.compile(r'\bwere not\b'), "were <ADV> not"),
                (re.compile(r'\bdoes not\b'), "<ADV> does not"),
                (re.compile(r'\bdo not\b'), "<ADV> do not"),
                (re.compile(r'\bcan be\b'), "<ADV> can be"),
                (re.compile(r'\bcannot be\b'), "<ADV> cannot be"),
                (re.compile(r'\bcan\b'), "can <ADV>"),
                (re.compile(r'\bcannot\b'), "<ADV> cannot"),
                (re.compile(r'\bhas not\b'), "<ADV> do not have"),
                (re.compile(r'\bhave not\b'), "<ADV> not have"),
                (re.compile(r'\bhas\b'), "<ADV> have"),
                (re.compile(r'\bhave\b'), "<ADV> have"),
                (re.compile(r'\brequires\b'), "<ADV> requires"),
                (re.compile(r'\brequire\b'), "<ADV> require"),
                (re.compile(r'\bcauses\b'), "<ADV> causes"),
                (re.compile(r'\bcause\b'), "<ADV> cause"),
                (re.compile(r'\bwants\b'), "<ADV> wants"),
                (re.compile(r'\bwant\b'), "<ADV> want"),
                (re.compile(r'\bneeds\b'), "<ADV> needs"),
                (re.compile(r'\bneed\b'), "<ADV> need"),
                (re.compile(r'\bindicates\b'), "<ADV> indicates"),
                (re.compile(r'\bindicate\b'), "<ADV> indicate"),
                (re.compile(r'\bcontains\b'), "<ADV> contains"),
                (re.compile(r'\bcontain\b'), "<ADV> contain"),
                (re.compile(r'\blikes\b'), "<ADV> likes"),
                (re.compile(r'\blike\b'), "<ADV> like"),
                (re.compile(r'\bexhibits\b'), "<ADV> exhibits"),
                (re.compile(r'\bexhibit\b'), "<ADV> exhibit"),
                (re.compile(r'\btastes\b'), "<ADV> tastes"),
                (re.compile(r'\btaste\b'), "<ADV> taste"),
                (re.compile(r'\bproduces\b'), "<ADV> produces"),
                (re.compile(r'\bproduce\b'), "<ADV> produce"),
                (re.compile(r'\bruns\b'), "<ADV> runs"),
                (re.compile(r'\brun\b'), "<ADV> run"),
                (re.compile(r'\bsmells\b'), "<ADV> smells"),
                (re.compile(r'\bsmell\b'), "<ADV> smell"),
                (re.compile(r'\bfounded\b'), "<ADV> founded")
                ]


def main(infile, outfile):
    data = [json.loads(line) for line in open(infile)]

    new_data = []
    for datum in data:
        if datum["polarity"] == 1:
            datum["polarity"] = "positive"
        else:
            datum["polarity"] = "negative"
        if check_already_modal(datum) is True:
            datum["uncertainty"] = "uncertain"
            new_data.append(datum)
            continue

        modalized_datum = modalize(datum)
        datum["uncertainty"] = "certain"
        modalized_datum["uncertainty"] = "uncertain"
        new_data.extend([datum, modalized_datum])

    with open(outfile, 'w') as outF:
        for datum in new_data:
            json.dump(datum, outF)
            outF.write('\n')


def check_already_modal(datum):
    keywords = mods["AUX"] + mods["ADV"]
    for keyword in keywords:
        if re.search(rf'\b{keyword}\b', datum["sentence"]) is not None:
            return True
    return False


def modalize(datum):
    sentence = datum["sentence"]
    patterns_set = [aux_patterns, adv_patterns]
    rev = random.choice([True, False])
    if rev is True:
        patterns_set = patterns_set[::-1]
    for patterns in patterns_set:
        for (pattern, repl) in patterns:
            new_sent = pattern.sub(repl, sentence)
            if new_sent != sentence:
                if "<ADV>" in new_sent:
                    mod = random.choice(mods["ADV"])
                    new_sent = new_sent.replace("<ADV>", mod)
                elif "<AUX>" in new_sent:
                    mod = random.choice(mods["AUX"])
                    new_sent = new_sent.replace("<AUX>", mod)
                else:
                    raise ValueError(f"Unknown mod type in '{new_sent}'")
                break
        if new_sent != sentence:
            break
    if new_sent == sentence:
        print(f"ALERT! No rule found for modalizing '{sentence}'")
    new_datum = dict(datum)
    new_datum["sentence"] = new_sent
    return new_datum


if __name__ == "__main__":
    args = parse_args()
    main(args.infile, args.outfile)
